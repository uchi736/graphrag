#!/usr/bin/env python
"""Fujitsu preprocessed markdown chunks → Neo4j KG (拡張スキーマ + source_chunks)

fujitsu_build_kg.py の markdown版。
- 入力: _bench/_pp/{pdf}/extracted_text.txt
- ingester: fujitsu_ingest_md.load_all_chunks
- スキーマ: SHARED_SCHEMA_PATH (拡張36関係)
- 出力: Neo4j (--fresh で全クリア)

Usage:
    python _bench/fujitsu_build_kg_md.py --fresh --workers 8
"""
from __future__ import annotations
import argparse, concurrent.futures, sys, time
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pp-dir', default='_bench/_pp')
    ap.add_argument('--fresh', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--max-chars', type=int, default=1000)
    ap.add_argument('--overlap', type=int, default=120)
    args = ap.parse_args()

    pp_dir = (_proj / args.pp_dir).resolve()
    if not pp_dir.exists():
        print(f'PP dir not found: {pp_dir}'); sys.exit(1)
    print(f'PP dir: {pp_dir}'); print(f'Workers: {args.workers}'); print(f'--fresh: {args.fresh}')

    from _bench.fujitsu_ingest_md import load_all_chunks
    print('Markdown chunks 構築中...')
    t0 = time.time()
    page_chunks = load_all_chunks(pp_dir, args.max_chars, args.overlap)
    print(f'  {len(page_chunks)} chunks ({time.time()-t0:.1f}s)')
    if args.limit:
        page_chunks = page_chunks[: args.limit]
        print(f'  --limit で {len(page_chunks)}')

    from langchain_core.documents import Document
    docs = [
        Document(page_content=c['text'],
                 metadata={'id': c['id'], 'source': c['source'],
                           'page': c['page'], 'sub': c['sub'],
                           'section': c.get('section', '')})
        for c in page_chunks
    ]

    from graphrag_core.config import reset_settings, get_settings
    reset_settings(); s = get_settings()
    from langchain_neo4j import Neo4jGraph
    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw, enhanced_schema=True)

    if args.fresh:
        print('\n🗑️ Neo4j 全削除中...')
        graph.query('MATCH (n) DETACH DELETE n')
        print('  完了')

    processed = graph.query('MATCH (c:ProcessedChunk) RETURN c.hash AS hash')
    processed_hashes = {r['hash'] for r in processed} if processed else set()
    pending = [d for d in docs if d.metadata['id'] not in processed_hashes]
    skipped = len(docs) - len(pending)
    print(f'\n  pending: {len(pending)} / total: {len(docs)} (skipped: {skipped})')
    if not pending:
        print('全チャンク処理済み → 終了'); return

    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from graphrag_core.llm.factory import create_chat_llm
    from graphrag_core.graph.schema import get_allowed_node_types, get_allowed_relations, describe_schema
    from graphrag_core.graph.enrichment import attach_source_chunks

    print(f'\n  Schema: {describe_schema()}')
    llm = create_chat_llm(temperature=0)
    additional = (
        "抽出する: 技術用語、概念、固有名詞、プロセス名、規格名、組織、人物、製品、指標、文書、法令。"
        "抽出しない: 一般的な名詞（『こと』『もの』『方法』）、代名詞、動詞。"
        "RELATED_TOは他に適切な関係がない場合の最終手段として使用。"
        "カテゴリ分類関係には BELONGS_TO_CATEGORY、IS-A関係には IS_A を使う。"
        "財務/統計の数値は Indicator として HAS_VALUE + MEASURED_IN で構造化。"
        "手順は HAS_STEP + FOLLOWS/PRECEDES で順序情報も付与。"
        "予防/軽減関係は PREVENTS/MITIGATES を使い分ける。"
    )
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=get_allowed_node_types(),
        allowed_relationships=get_allowed_relations(),
        strict_mode=False,
        ignore_tool_usage=True,
        additional_instructions=additional,
    )

    success, error = 0, 0
    t_start = time.time()

    def process_chunk(doc):
        chunk_docs = transformer.convert_to_graph_documents([doc])
        graph.add_graph_documents(chunk_docs, include_source=True)
        cid = doc.metadata['id']
        attach_source_chunks(graph, chunk_docs, cid)
        graph.query('MERGE (c:ProcessedChunk {hash: $hash}) SET c.processed_at = datetime()', {'hash': cid})
        graph.query(
            "MATCH (d:Document {id: $id}) SET d.source = $src, d.page = $page, d.section = $section",
            {'id': cid, 'src': doc.metadata['source'], 'page': doc.metadata['page'],
             'section': doc.metadata.get('section', '')},
        )
        return cid

    print(f'\n🕸️ KG構築 開始: {len(pending)} chunks × {args.workers} workers')
    print(f'   見積もり: {len(pending) * 3 / args.workers / 60:.0f}分 (avg 3s/chunk仮定)')

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_chunk, d): d for d in pending}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            try: fut.result(); success += 1
            except Exception as e:
                error += 1
                if error <= 5: print(f'  ⚠️  {type(e).__name__}: {str(e)[:100]}')
            done += 1
            if done % 50 == 0 or done == len(pending):
                avg = (time.time() - t_start) / done
                eta = avg * (len(pending) - done) / 60
                print(f'  {done}/{len(pending)}  avg={avg:.1f}s  ETA={eta:.0f}min  ok={success} err={error}')

    print(f'\n✅ KG構築完了: 成功{success}, エラー{error}, {(time.time()-t_start)/60:.1f}分')

    nr = graph.query('MATCH (n) RETURN count(n) AS c')
    er = graph.query('MATCH ()-[r]->() RETURN count(r) AS c')
    print(f'   Neo4j: ノード={nr[0]["c"]}, エッジ={er[0]["c"]}')

    print('\n🔧 mention_count + pagerank 計算中...')
    from graphrag_core.graph.enrichment import enrich_post_build
    stats = enrich_post_build(graph)
    print(f"   mention_count: {stats['mention_count']} Term")
    print(f"   pagerank: {stats['pagerank']} Term")

    print('\n🔍 エンティティベクトル化中...')
    try:
        from graphrag_core.retrieval.entity_vector import EntityVectorizer
        from graphrag_core.llm.factory import create_embeddings
        embeddings = create_embeddings()
        ev = EntityVectorizer(s.pg_conn, embeddings)
        entities = ev.extract_entities_from_graph(graph)
        print(f'   抽出 Term: {len(entities)}')
        n = ev.add_entities(entities, [])
        print(f'   保存: {n}')
    except Exception as e:
        print(f'   ⚠️ {e}')

    print('\n完了')


if __name__ == '__main__':
    main()
