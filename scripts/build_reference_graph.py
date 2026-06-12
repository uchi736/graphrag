#!/usr/bin/env python
"""既存のNeo4jグラフに参照グラフ + 照応解決を後付け適用する

build_kg.py の再構築なしで、現行グラフのDocumentチャンクから
REFERS_TOエッジ・ref_docs・照応フラグを生成する。

Usage:
    python scripts/build_reference_graph.py
"""
import logging
import sys
import time
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main():
    from graphrag_core.config import get_settings
    from langchain_neo4j import Neo4jGraph
    from graphrag_core.graph.references import build_reference_graph
    from graphrag_core.graph.consolidate import resolve_anaphora_nodes

    s = get_settings()
    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user,
                       password=s.neo4j_pw, enhanced_schema=False)

    t0 = time.time()
    print("=== 参照グラフ構築 ===")
    ref_stats = build_reference_graph(graph)
    print(f"  chunks: {ref_stats['chunks']}")
    print(f"  titles: {ref_stats['titles']}")
    print(f"  参照エッジ: {ref_stats['edges_written']}本")
    print(f"  文書名参照チャンク: {ref_stats['doc_ref_chunks']}件")
    print(f"  未解決参照: {ref_stats['unresolved']}件")
    print(f"  略称定義: {sum(len(v) for v in ref_stats['alias_maps'].values())}件 "
          f"({len(ref_stats['alias_maps'])}文書)")

    print("\n=== 照応解決 ===")
    ana_stats = resolve_anaphora_nodes(graph, ref_stats["alias_maps"])
    print(f"  解決（正式名称へ紐づけ）: {ana_stats['resolved']}件")
    print(f"  検索除外フラグ(is_anaphor): {ana_stats['flagged']}件")

    # kind別エッジ統計
    rows = graph.query(
        "MATCH ()-[r:REFERS_TO]->() RETURN r.kind AS kind, count(*) AS c ORDER BY c DESC")
    print("\n=== REFERS_TO kind別 ===")
    for r in rows:
        print(f"  {r['kind']}: {r['c']}")

    print(f"\n完了: {time.time()-t0:.0f}s")
    print("REFERENCE GRAPH COMPLETE")


if __name__ == "__main__":
    main()
