"""KG構築・チャンク更新・増分更新・ジョブ管理エンドポイント。"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.deps import require_ready
from api.jobs import JobBusy
from api.sse import sse_response
from api.state import AppState
from graphrag_core.services.build import BuildOptions, build_knowledge_base, update_chunks_only
from graphrag_core.services.ingest import load_csv_edges_from_bytes, load_documents_from_bytes

router = APIRouter(prefix="/api", tags=["build"])


def _read_uploads(files: List[UploadFile]) -> list[tuple[str, bytes]]:
    out = []
    for f in files:
        data = f.file.read()
        out.append((f.filename or "unnamed", data))
    return out


@router.post("/build", status_code=202)
async def build(
    files: List[UploadFile] = File(default=[]),
    csv_file: Optional[UploadFile] = File(default=None),
    mode: str = Form("new"),
    enable_knowledge_graph: bool = Form(True),
    enable_entity_vector: bool = Form(True),
    enable_japanese_search: bool = Form(True),
    st: AppState = Depends(require_ready),
) -> dict:
    """KG構築ジョブを投入する（multipart）。202 + job_id を返す。"""
    if mode not in ("new", "resume"):
        raise HTTPException(400, "mode は new | resume")
    if not files and csv_file is None:
        raise HTTPException(400, "files または csv_file が必要です")

    raw_files = _read_uploads(files)
    csv_bytes = await csv_file.read() if csv_file is not None else None
    options = BuildOptions(
        enable_knowledge_graph=enable_knowledge_graph,
        enable_entity_vector=enable_entity_vector,
        enable_japanese_search=enable_japanese_search,
    )
    settings = st.settings

    def run(progress, should_cancel):
        # 重い処理（PDF解析含む）はすべてジョブスレッド内で行う
        source_docs = load_documents_from_bytes(raw_files) if raw_files else []
        csv_edges = load_csv_edges_from_bytes(csv_bytes) if csv_bytes else []
        stats = build_knowledge_base(
            source_docs, csv_edges, mode=mode, options=options, settings=settings,
            progress=progress, should_cancel=should_cancel,
        )
        st.invalidate_retrieval()
        return stats

    try:
        job = st.jobs.submit("build", run)
    except JobBusy as e:
        raise HTTPException(409, {"message": str(e), "running_job_id": e.running_job_id})
    return {"job_id": job.id}


@router.post("/build/chunks-only", status_code=202)
async def build_chunks_only(
    files: List[UploadFile] = File(...),
    enable_japanese_search: bool = Form(True),
    st: AppState = Depends(require_ready),
) -> dict:
    """チャンクのみ更新ジョブ（グラフ再構築なし・高速）。"""
    raw_files = _read_uploads(files)
    options = BuildOptions(enable_japanese_search=enable_japanese_search)
    settings = st.settings

    def run(progress, should_cancel):
        source_docs = load_documents_from_bytes(raw_files)
        result = update_chunks_only(source_docs, options=options, settings=settings,
                                    progress=progress)
        st.invalidate_retrieval()
        return result

    try:
        job = st.jobs.submit("chunks_only", run)
    except JobBusy as e:
        raise HTTPException(409, {"message": str(e), "running_job_id": e.running_job_id})
    return {"job_id": job.id}


@router.post("/documents/{doc_id}/update", status_code=202)
async def update_document_incremental(
    doc_id: str,
    file: UploadFile = File(...),
    st: AppState = Depends(require_ready),
) -> dict:
    """1文書の増分更新ジョブ（内容ハッシュ差分→剪定→added再抽出→全ストア同期）。"""
    data = await file.read()
    filename = file.filename or doc_id
    settings = st.settings
    graph = st.graph
    embeddings = st.embeddings

    def run(progress, should_cancel):
        import itertools
        from graphrag_core.llm.factory import create_chat_llm
        from graphrag_core.graph.incremental import update_document
        from graphrag_core.services.documents import build_add_chunk_fn
        from graphrag_core.services.ingest import load_document_from_bytes
        from graphrag_core.services.progress import JobCancelled, ProgressEvent
        from graphrag_core.text.chunking import create_markdown_chunks

        progress(ProgressEvent(stage="load", message=f"{filename} を解析中..."))
        doc = load_document_from_bytes(filename, data)
        doc.metadata["source"] = doc_id
        chunks = create_markdown_chunks([doc], chunk_size=1024, chunk_overlap=100)
        for c in chunks:
            c.metadata["source"] = doc_id

        # timeout は factory 引数で（後付け代入は効かない）。永続失敗チャンクは fail-fast
        llm = create_chat_llm(temperature=0, timeout=90, max_retries=1)
        base_add = build_add_chunk_fn(graph, llm)
        counter = itertools.count(1)

        def counting_add(chunk):
            if should_cancel():
                raise JobCancelled()
            r = base_add(chunk)
            progress(ProgressEvent(stage="extract", current=next(counter), total=None,
                                   message=f"チャンク抽出中... ({chunk.metadata.get('id', '')[:8]})"))
            return r

        def run_post():
            from graphrag_core.graph.enrichment import enrich_post_update
            progress(ProgressEvent(stage="post", message="軽量後処理（mention_count/search_keys）..."))
            enrich_post_update(graph)

        result = update_document(
            graph, doc_id, chunks, counting_add,
            pg_conn=settings.pg_conn, pg_collection=settings.pg_collection,
            embeddings=embeddings, run_post=run_post,
        )
        st.invalidate_retrieval()
        return result

    try:
        job = st.jobs.submit("doc_update", run)
    except JobBusy as e:
        raise HTTPException(409, {"message": str(e), "running_job_id": e.running_job_id})
    return {"job_id": job.id, "doc_id": doc_id}


@router.delete("/documents/{doc_id}")
def delete_document_endpoint(doc_id: str, st: AppState = Depends(require_ready)) -> dict:
    """文書をグラフ・PGVector・エンティティベクトルから完全削除する（同期）。"""
    from graphrag_core.graph.incremental import delete_document
    result = delete_document(
        st.graph, doc_id,
        pg_conn=st.settings.pg_conn, pg_collection=st.settings.pg_collection,
        embeddings=st.embeddings,
    )
    st.invalidate_retrieval()
    return result


@router.post("/build/edc-sync", status_code=202)
async def edc_sync(
    payload: Optional[dict] = None,
    st: AppState = Depends(require_ready),
) -> dict:
    """EDCスキーマ同期ジョブ。現コレクションの文書サンプル→EDC /extract→スキーマJSON。

    body(任意): {"endpoint": str, "docs": int, "chunks_per_doc": int, "out": str}
    """
    p = payload or {}
    settings = st.settings

    def run(progress, should_cancel):
        from graphrag_core.services.schema_sync import sync_edc_schema
        return sync_edc_schema(
            settings.pg_conn, settings.pg_collection,
            out_path=p.get("out"),
            endpoint=p.get("endpoint"),
            n_docs=int(p.get("docs") or 4),
            chunks_per_doc=int(p.get("chunks_per_doc") or 6),
            progress=progress, should_cancel=should_cancel,
        )

    try:
        job = st.jobs.submit("edc_sync", run)
    except JobBusy as e:
        raise HTTPException(409, {"message": str(e), "running_job_id": e.running_job_id})
    return {"job_id": job.id}


# ── ジョブ管理 ─────────────────────────────────────────────────────
@router.get("/jobs")
def list_jobs(st: AppState = Depends(require_ready)) -> list:
    return st.jobs.list()


@router.get("/jobs/{job_id}")
def get_job(job_id: str, st: AppState = Depends(require_ready)) -> dict:
    job = st.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "ジョブが見つかりません")
    return job.snapshot()


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str, st: AppState = Depends(require_ready)):
    """ジョブ進捗のSSEストリーム（過去分replay + live、終了stateで閉じる）。"""
    if st.jobs.get(job_id) is None:
        raise HTTPException(404, "ジョブが見つかりません")
    return sse_response((ev["type"], ev.get("data", {})) for ev in st.jobs.subscribe(job_id))


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, st: AppState = Depends(require_ready)) -> dict:
    ok = st.jobs.cancel(job_id)
    if not ok:
        raise HTTPException(400, "キャンセルできません（未実行または終了済み）")
    return {"ok": True}
