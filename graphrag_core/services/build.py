"""KG構築サービス（ui/system.py build_rag_system / update_chunks_only の st非依存版）。

- 進捗は ProgressFn コールバックで通知（Streamlit: st.progress へ、API: ジョブSSEへ）
- should_cancel() が True を返すとチャンクループを協調中断（JobCancelled）
- 戻り値 BuildStats（旧 st.session_state.last_build_stats の戻り値化）
- QAチェーン構築は含まない（Streamlit 側は services.qa.make_qa_chain で組む）
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

from graphrag_core.config import get_settings
from graphrag_core.graph.incremental import make_chunk_id
from graphrag_core.llm.factory import create_chat_llm, create_embeddings
from graphrag_core.llm.langfuse_utils import is_langfuse_enabled
from graphrag_core.prompts import KG_SYSTEM_PROMPT, KG_USER_PROMPT
from graphrag_core.services.progress import JobCancelled, ProgressEvent, ProgressFn, noop_progress
from graphrag_core.text.chunking import create_markdown_chunks
from graphrag_core.text.japanese import get_japanese_processor
from graphrag_core.db.utils import (
    add_connection_timeout,
    batch_pgvector_from_documents,
    batch_update_tokenized,
    ensure_embedding_id_unique,
    ensure_hnsw_index,
    ensure_schema_compatibility,
    ensure_tokenized_schema,
)


@dataclass
class BuildOptions:
    enable_knowledge_graph: bool = True
    enable_entity_vector: bool = True
    enable_japanese_search: bool = True


def _make_chunks(source_docs: List) -> List:
    """2段階Markdownチャンキング + 図チャンク展開 + 内容ハッシュID採番 + 重複除去。"""
    from graphrag_core.text.chunking import expand_figure_chunks
    # 図チャンクを先に展開（内部で metadata["figures"] を pop するため、
    # create_markdown_chunks より前に呼ぶこと＝本文チャンクへの伝播防止）
    figure_chunks = []
    for doc in source_docs:
        figure_chunks.extend(expand_figure_chunks(doc))
    all_chunks = create_markdown_chunks(source_docs, chunk_size=1024, chunk_overlap=100)
    all_chunks.extend(figure_chunks)
    deduped, seen = [], set()
    for chunk in all_chunks:
        digest = make_chunk_id(chunk.metadata.get("source", ""), chunk.page_content)
        if digest in seen:
            continue
        seen.add(digest)
        chunk.metadata["id"] = digest
        deduped.append(chunk)
    return deduped


def _ingest_csv_edges(graph, csv_edges: List[Dict]) -> None:
    for edge in csv_edges:
        graph.query(
            f"""
            MERGE (s:CSVNode {{id: $src}})
            MERGE (t:CSVNode {{id: $tgt}})
            MERGE (s)-[r:`{edge['label']}`]->(t)
            """,
            params={"src": edge["source"], "tgt": edge["target"]},
        )


def _vectorize_entities(graph, pg_conn, embeddings, graph_docs, progress: ProgressFn) -> None:
    progress(ProgressEvent(stage="entity_vector", message="エンティティをベクトル化中..."))
    try:
        from graphrag_core.retrieval.entity_vector import EntityVectorizer
        ev = EntityVectorizer(pg_conn, embeddings)
        entities = ev.extract_entities_from_graph(graph)
        num_saved = ev.add_entities(entities, graph_docs)
        progress(ProgressEvent(stage="entity_vector",
                               message=f"{num_saved}個のエンティティをベクトル化しました"))
    except Exception as e:
        progress(ProgressEvent(stage="entity_vector", level="warning",
                               message=f"エンティティベクトル化エラー: {e}"))


def build_knowledge_base(
    source_docs: List,
    csv_edges: Optional[List[Dict]] = None,
    *,
    mode: str = "new",                     # "new" = ProcessedChunk クリア / "resume" = スキップ再開
    options: Optional[BuildOptions] = None,
    settings=None,
    progress: ProgressFn = noop_progress,
    should_cancel: Callable[[], bool] = lambda: False,
    session_id: Optional[str] = None,
    llm=None,
) -> Dict:
    """ナレッジベース構築（Neo4j KG + PGVector + エンティティベクトル + provenance）。

    Returns BuildStats: {"ok", "err", "total", "chunks", "skipped", "cancelled"}
    """
    from langchain_neo4j import Neo4jGraph
    from graphrag_core.graph.provenance import stamp_graph_provenance

    s = settings or get_settings()
    opts = options or BuildOptions()
    csv_edges = csv_edges or []
    stats = {"ok": 0, "err": 0, "total": 0, "chunks": 0, "skipped": 0, "cancelled": False}

    embeddings = create_embeddings()

    progress(ProgressEvent(stage="chunk", message="チャンキング中..."))
    chunks = _make_chunks(source_docs)
    stats["chunks"] = len(chunks)

    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user,
                       password=s.neo4j_pw, enhanced_schema=True)

    if mode == "new":
        try:
            graph.query("MATCH (c:ProcessedChunk) DELETE c")
            progress(ProgressEvent(stage="prepare", message="処理済みデータをクリアしました"))
        except Exception as e:
            progress(ProgressEvent(stage="prepare", level="warning",
                                   message=f"クリア処理でエラー（続行します）: {e}"))

    graph_docs = []
    if opts.enable_knowledge_graph and chunks:
        progress(ProgressEvent(stage="kg_extract", message="ナレッジグラフ生成中..."))

        llm_provider = s.llm_provider.lower()
        _llm = llm or create_chat_llm(temperature=0, timeout=120, max_retries=2)
        if is_langfuse_enabled() and session_id:
            from langfuse.langchain import CallbackHandler as LangfuseHandler
            _llm.callbacks = [LangfuseHandler(trace_context={
                "name": "kg_building",
                "session_id": f"{session_id}_kg_{datetime.now().strftime('%H%M%S')}"})]

        from langchain_experimental.graph_transformers import LLMGraphTransformer
        from langchain_core.prompts import ChatPromptTemplate
        from graphrag_core.graph.schema import (
            entity_naming_instructions, get_allowed_node_types, get_allowed_relations)

        _kg_kwargs = dict(
            llm=_llm,
            allowed_nodes=get_allowed_node_types(),
            allowed_relationships=get_allowed_relations(),
            strict_mode=False,
            ignore_tool_usage=(llm_provider == "vllm"),
        )
        if llm_provider == "vllm":
            # VLLM はデフォルトprompt（JSON出力指示付き）+ additional_instructions
            _kg_kwargs["additional_instructions"] = (
                "抽出する: 技術用語、概念、固有名詞、プロセス名、規格名。"
                "抽出しない: 一般的な名詞（「こと」「もの」「方法」）、代名詞、動詞。"
                "RELATED_TOは他に適切な関係がない場合の最終手段として使用。"
                + entity_naming_instructions()
            )
        else:
            _kg_kwargs["prompt"] = ChatPromptTemplate.from_messages(
                [("system", KG_SYSTEM_PROMPT), ("user", KG_USER_PROMPT)])

        transformer = LLMGraphTransformer(**_kg_kwargs)

        try:
            processed = graph.query("MATCH (c:ProcessedChunk) RETURN c.hash AS hash")
            processed_hashes = {r["hash"] for r in processed} if processed else set()
        except Exception:
            processed_hashes = set()

        pending = [c for c in chunks if c.metadata.get("id") not in processed_hashes]
        stats["skipped"] = len(chunks) - len(pending)
        stats["total"] = len(pending)
        if stats["skipped"]:
            progress(ProgressEvent(stage="kg_extract",
                                   message=f"処理対象 {len(pending)}/{len(chunks)} チャンク（{stats['skipped']}件スキップ）"))

        for i, chunk in enumerate(pending):
            if should_cancel():
                stats["cancelled"] = True
                raise JobCancelled()
            # 図チャンク（キャプション文）はKG抽出しない。ProcessedChunkだけ
            # 記録して resume/差分検出の整合を保つ
            if chunk.metadata.get("type") == "figure":
                if chunk.metadata.get("id"):
                    graph.query(
                        "MERGE (c:ProcessedChunk {hash: $hash}) SET c.processed_at = datetime()",
                        {"hash": chunk.metadata["id"]})
                stats["ok"] += 1
                continue
            try:
                chunk_docs = transformer.convert_to_graph_documents([chunk])
                graph.add_graph_documents(chunk_docs, include_source=True)
                graph_docs.extend(chunk_docs)
                chunk_hash = chunk.metadata.get("id")
                if chunk_hash:
                    graph.query(
                        "MERGE (c:ProcessedChunk {hash: $hash}) SET c.processed_at = datetime()",
                        {"hash": chunk_hash})
                stats["ok"] += 1
            except Exception as e:
                stats["err"] += 1
                progress(ProgressEvent(stage="kg_extract", level="warning",
                                       current=i + 1, total=len(pending),
                                       ok=stats["ok"], err=stats["err"],
                                       message=f"チャンク {i+1} の処理でエラー: {e}"))
                continue
            progress(ProgressEvent(stage="kg_extract", current=i + 1, total=len(pending),
                                   ok=stats["ok"], err=stats["err"],
                                   message=f"ナレッジグラフ生成中... {i+1}/{len(pending)}"))

    if csv_edges:
        progress(ProgressEvent(stage="csv_edges", message=f"CSVから{len(csv_edges)}件のエッジを追加中..."))
        _ingest_csv_edges(graph, csv_edges)

    # Documentノードに source 名を付与（増分更新の差分検出に必須）
    from graphrag_core.graph.schema import chunk_label
    for chunk in chunks:
        chunk_id = chunk.metadata.get("id")
        if chunk_id:
            graph.query(
                "MATCH (d:" + chunk_label() + " {id: $chunk_id}) SET d.source = $doc_name",
                params={"chunk_id": chunk_id,
                        "doc_name": chunk.metadata.get("source", "Unknown")})

    if opts.enable_entity_vector and (opts.enable_knowledge_graph or csv_edges):
        _vectorize_entities(graph, s.pg_conn, embeddings, graph_docs, progress)

    if not opts.enable_knowledge_graph:
        try:
            from graphrag_core.graph.schema import stamp_schema_metadata
            stamp_schema_metadata(graph)
        except Exception:
            pass

    # 日本語トークン化
    jp = get_japanese_processor()
    if jp and opts.enable_japanese_search and chunks:
        progress(ProgressEvent(stage="tokenize", message="日本語トークン化中..."))
        for chunk in chunks:
            try:
                chunk.metadata["tokenized_content"] = jp.tokenize(chunk.page_content)
            except Exception:
                chunk.metadata["tokenized_content"] = None

    # PGVector保存
    vector_store = None
    if chunks:
        progress(ProgressEvent(stage="pgvector", message="PGVectorへ保存中..."))
        ensure_embedding_id_unique(s.pg_conn)
        ensure_schema_compatibility(s.pg_conn)
        ensure_hnsw_index(s.pg_conn)
        pg = add_connection_timeout(s.pg_conn, timeout=30)
        vector_store = batch_pgvector_from_documents(
            chunks, embeddings, connection=pg,
            collection_name=s.pg_collection, pre_delete_collection=True)
        if jp and opts.enable_japanese_search:
            try:
                ensure_tokenized_schema(s.pg_conn)
                batch_update_tokenized(s.pg_conn, chunks)
            except Exception as e:
                progress(ProgressEvent(stage="pgvector", level="warning",
                                       message=f"トークン化データのDB保存エラー: {e}"))
    else:
        progress(ProgressEvent(stage="pgvector", level="warning",
                               message="チャンクが0件のためベクトルストア保存をスキップしました"))

    # provenance刻印（グラフ↔コレクション整合ゲート用）
    from graphrag_core.graph.provenance import stamp_graph_provenance
    stamp_graph_provenance(graph, s.pg_collection, doc_count=len(chunks))

    progress(ProgressEvent(stage="done", ok=stats["ok"], err=stats["err"],
                           message="構築が完了しました"))
    return stats


def update_chunks_only(
    source_docs: List,
    *,
    options: Optional[BuildOptions] = None,
    settings=None,
    progress: ProgressFn = noop_progress,
) -> Dict:
    """チャンクのみ更新（グラフ再構築スキップ、PGVector を作り直す）。"""
    s = settings or get_settings()
    opts = options or BuildOptions()

    embeddings = create_embeddings()
    progress(ProgressEvent(stage="chunk", message="チャンキング中..."))
    chunks = _make_chunks(source_docs)

    jp = get_japanese_processor()
    if jp and opts.enable_japanese_search:
        progress(ProgressEvent(stage="tokenize", message="日本語トークン化中..."))
        for chunk in chunks:
            try:
                chunk.metadata["tokenized_content"] = jp.tokenize(chunk.page_content)
            except Exception:
                chunk.metadata["tokenized_content"] = None

    if not chunks:
        progress(ProgressEvent(stage="pgvector", level="warning", message="チャンクが0件のためスキップしました"))
        return {"chunks": 0}

    progress(ProgressEvent(stage="pgvector", message="PGVectorへ保存中..."))
    ensure_embedding_id_unique(s.pg_conn)
    ensure_schema_compatibility(s.pg_conn)
    ensure_hnsw_index(s.pg_conn)
    pg = add_connection_timeout(s.pg_conn, timeout=30)
    batch_pgvector_from_documents(
        chunks, embeddings, connection=pg,
        collection_name=s.pg_collection, pre_delete_collection=True)
    if jp and opts.enable_japanese_search:
        try:
            ensure_tokenized_schema(s.pg_conn)
            batch_update_tokenized(s.pg_conn, chunks)
        except Exception as e:
            progress(ProgressEvent(stage="pgvector", level="warning",
                                   message=f"トークン化データのDB保存エラー: {e}"))

    progress(ProgressEvent(stage="done", message="チャンク更新が完了しました"))
    return {"chunks": len(chunks)}
