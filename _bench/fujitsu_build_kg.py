#!/usr/bin/env python
"""Fujitsu PDFs を Neo4j にナレッジグラフ化

page-aware なチャンク (fujitsu_ingest.py と同じ作り方) を入力にして、
LLMGraphTransformer で各チャンクから Term/関係を抽出して Neo4j に投入。
edge には source_chunks (チャンクhash) を付与し、triple→chunk リンクを正確にする。

Usage:
    python _bench/fujitsu_build_kg.py --fresh                    # Neo4j全クリアして全PDF
    python _bench/fujitsu_build_kg.py --limit 100                # 先頭100チャンクのみ
    python _bench/fujitsu_build_kg.py --workers 8                # 並列度上げる
    python _bench/fujitsu_build_kg.py --resume                   # 既処理スキップ (--fresh無し時のデフォルト)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
import time
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="../Fujitsu-RAG-Hard-Benchmark/dataset/PDFs")
    ap.add_argument("--fresh", action="store_true", help="Neo4j を MATCH (n) DETACH DELETE n で全削除してから始める")
    ap.add_argument("--workers", type=int, default=4, help="並列ワーカー数 (default=4)")
    ap.add_argument("--limit", type=int, default=None, help="先頭N チャンクのみ (デバッグ用)")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = (_proj / args.pdf_dir).resolve()
    if not pdf_dir.exists():
        print(f"PDFフォルダが見つかりません: {pdf_dir}"); sys.exit(1)
    print(f"PDF dir: {pdf_dir}")
    print(f"Workers: {args.workers}")
    print(f"--fresh: {args.fresh}")

    # fujitsu_ingest と同じ page-aware chunker を使う
    from _bench.fujitsu_ingest import load_page_chunks
    print("ページチャンク抽出中...")
    t0 = time.time()
    page_chunks = load_page_chunks(pdf_dir)
    print(f"  {len(page_chunks)} chunks ({time.time()-t0:.1f}s)")

    if args.limit:
        page_chunks = page_chunks[: args.limit]
        print(f"  --limit で {len(page_chunks)} chunks に絞り")

    # langchain Document に変換 (metadata.id でNeo4j上でも一意に追跡)
    from langchain_core.documents import Document
    docs = [
        Document(
            page_content=c["text"],
            metadata={
                "id": c["id"], "source": c["source"],
                "page": c["page"], "sub": c.get("sub", 0),
            },
        )
        for c in page_chunks
    ]

    # Neo4j 接続
    from graphrag_core.config import reset_settings, get_settings
    reset_settings()
    s = get_settings()
    from langchain_neo4j import Neo4jGraph
    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw, enhanced_schema=True)

    if args.fresh:
        print("\n🗑️ Neo4j を全削除中...")
        graph.query("MATCH (n) DETACH DELETE n")
        print("  完了")

    # 処理済みhash 取得
    processed = graph.query("MATCH (c:ProcessedChunk) RETURN c.hash AS hash")
    processed_hashes = {r["hash"] for r in processed} if processed else set()
    pending = [d for d in docs if d.metadata["id"] not in processed_hashes]
    skipped = len(docs) - len(pending)
    print(f"\n  pending: {len(pending)} / total: {len(docs)} (skipped: {skipped})")

    if not pending:
        print("全チャンク処理済み → 終了")
        return

    # LLMGraphTransformer 準備
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from graphrag_core.llm.factory import create_chat_llm
    from graphrag_core.graph.schema import get_allowed_node_types, get_allowed_relations
    from graphrag_core.graph.enrichment import attach_source_chunks

    llm = create_chat_llm(temperature=0)
    additional = (
        "抽出する: 技術用語、概念、固有名詞、プロセス名、規格名、組織、人物、製品。"
        "抽出しない: 一般的な名詞（『こと』『もの』『方法』）、代名詞、動詞。"
        "RELATED_TOは他に適切な関係がない場合の最終手段として使用。"
        "カテゴリ分類関係には BELONGS_TO_CATEGORY を、IS-A関係には IS_A を使う。"
    )
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=get_allowed_node_types(),
        allowed_relationships=get_allowed_relations(),
        strict_mode=False,
        ignore_tool_usage=True,
        additional_instructions=additional,
    )

    # ビルド時 entity ID 正規化 (Stage A 相当)
    # 「一般社団法人日本自動車...」と「一般社団法人 日本自動車...」を MERGE 時点で統合
    import unicodedata as _ud
    try:
        import neologdn as _neologdn  # type: ignore
        _has_neologdn = True
    except Exception:
        _has_neologdn = False
        print("⚠️  neologdn 未インストール → NFKCのみで正規化")

    def _normalize_id(name):
        if not name: return name
        s = _ud.normalize("NFKC", str(name))
        if _has_neologdn:
            s = _neologdn.normalize(s)
        s = re.sub(r"\s+", "", s).strip() if 're' in dir() else s.replace(' ', '').replace('　', '').strip()
        return s or str(name)

    success, error = 0, 0
    t_start = time.time()

    def process_chunk(doc):
        chunk_docs = transformer.convert_to_graph_documents([doc])

        # ★ Stage A 正規化: 全Node/Relationship endpoint の id を NFKC + neologdn 統一
        for gd in chunk_docs:
            for node in gd.nodes:
                orig = node.id
                node.id = _normalize_id(orig)
                if orig != node.id:
                    if not getattr(node, "properties", None):
                        node.properties = {}
                    existing = node.properties.get("surface_forms", [])
                    if isinstance(existing, str): existing = [existing]
                    if orig not in existing:
                        node.properties["surface_forms"] = existing + [orig]
            for rel in gd.relationships:
                rel.source.id = _normalize_id(rel.source.id)
                rel.target.id = _normalize_id(rel.target.id)

        graph.add_graph_documents(chunk_docs, include_source=True)
        cid = doc.metadata["id"]
        attach_source_chunks(graph, chunk_docs, cid)
        graph.query(
            "MERGE (c:ProcessedChunk {hash: $hash}) SET c.processed_at = datetime()",
            {"hash": cid},
        )
        # Document に page/source も付与しておく (KG source chunk取得時に使える)
        graph.query(
            "MATCH (d:Document {id: $id}) SET d.source = $src, d.page = $page",
            {"id": cid, "src": doc.metadata["source"], "page": doc.metadata["page"]},
        )
        return cid

    print(f"\n🕸️ KG構築 開始: {len(pending)} chunks × {args.workers} workers")
    print(f"   見積もり時間: {len(pending) * 6 / args.workers / 60:.0f}分 (avg 6s/chunk仮定)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_chunk, d): d for d in pending}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            try:
                fut.result(); success += 1
            except Exception as e:
                error += 1
                if error <= 5:
                    print(f"  ⚠️  err: {type(e).__name__}: {str(e)[:100]}")
            done += 1
            if done % 50 == 0 or done == len(pending):
                avg = (time.time() - t_start) / done
                eta = avg * (len(pending) - done) / 60
                print(f"  {done}/{len(pending)}  avg={avg:.1f}s/chunk  ETA={eta:.0f}min  ok={success} err={error}")

    print(f"\n✅ KG構築完了: 成功{success}, エラー{error}, {(time.time()-t_start)/60:.1f}分")

    # 統計
    nr = graph.query("MATCH (n) RETURN count(n) AS c")
    er = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")
    print(f"   Neo4j: ノード={nr[0]['c']}, エッジ={er[0]['c']}")

    # 後付けプロパティ
    print("\n🔧 mention_count + pagerank 計算中...")
    from graphrag_core.graph.enrichment import enrich_post_build
    stats = enrich_post_build(graph)
    print(f"   mention_count: {stats['mention_count']} Term")
    print(f"   pagerank: {stats['pagerank']} Term")

    # エンティティベクトル化 (myapp_entities ではなく fjrag_entities に分ける)
    print("\n🔍 エンティティベクトル化中...")
    try:
        from graphrag_core.retrieval.entity_vector import EntityVectorizer
        from graphrag_core.llm.factory import create_embeddings
        embeddings = create_embeddings()
        # collection_name を指定するためカスタムで初期化
        ev = EntityVectorizer(s.pg_conn, embeddings)
        # 念のため EntityVectorizer の collection_name 上書き
        # (実装によって変わるので env も差し替える方が確実)
        entities = ev.extract_entities_from_graph(graph)
        print(f"   抽出 Term: {len(entities)}")
        n_saved = ev.add_entities(entities, [])
        print(f"   保存: {n_saved}")
    except Exception as e:
        print(f"   ⚠️  EntityVectorizer 失敗: {e}")

    print("\n完了")


if __name__ == "__main__":
    main()
