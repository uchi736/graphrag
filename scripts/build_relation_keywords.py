#!/usr/bin/env python
"""build_relation_keywords.py - エッジのテーマキーワード注釈＋関係ベクトル索引の構築

LightRAG dual-level retrieval の高レベル側を既存グラフに後付けする
（再抽出・再ビルド不要。グラフをフル再構築したら再実行すること）。

使用例:
    PG_COLLECTION=synth_v1 python scripts/build_relation_keywords.py
    python scripts/build_relation_keywords.py --limit 40 --batch 20   # スモーク
    python scripts/build_relation_keywords.py --skip-annotate         # 索引のみ再構築
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv()


def main():
    ap = argparse.ArgumentParser(description="エッジキーワード注釈＋関係ベクトル索引")
    ap.add_argument("--batch", type=int, default=20, help="LLM1回あたりのエッジ数")
    ap.add_argument("--limit", type=int, default=None, help="先頭Nエッジのみ（スモーク用）")
    ap.add_argument("--skip-annotate", action="store_true", help="注釈をスキップし索引のみ")
    ap.add_argument("--skip-index", action="store_true", help="索引をスキップし注釈のみ")
    args = ap.parse_args()

    from langchain_neo4j import Neo4jGraph
    from graphrag_core.config import get_settings
    from graphrag_core.graph.relation_keywords import (
        annotate_relation_keywords, build_relation_vector_index,
        relation_collection_name)
    from graphrag_core.llm.factory import create_chat_llm, create_embeddings

    s = get_settings()
    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw,
                       enhanced_schema=False)
    print(f"=== relation keywords === collection={s.pg_collection} neo4j={s.neo4j_uri}")

    if not args.skip_annotate:
        llm = create_chat_llm(temperature=0, timeout=120, max_retries=2)
        t0 = time.time()

        def prog(b, nb, done):
            if b % 5 == 0 or b == nb:
                print(f"  注釈 {b}/{nb}バッチ (付与済み {done})", flush=True)

        r = annotate_relation_keywords(graph, llm, pg_conn=s.pg_conn,
                                       batch_size=args.batch, limit=args.limit,
                                       progress=prog)
        print(f"✅ 注釈: {r['annotated']}/{r['total']}エッジ "
              f"(失敗バッチ {r['failed_batches']}, {time.time()-t0:.0f}s)")

    if not args.skip_index:
        emb = create_embeddings()
        t0 = time.time()
        r = build_relation_vector_index(graph, emb, s.pg_conn, s.pg_collection)
        print(f"✅ 索引: {r['indexed']}件 → {relation_collection_name(s.pg_collection)} "
              f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
