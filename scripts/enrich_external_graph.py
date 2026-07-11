# -*- coding: utf-8 -*-
"""外部構築グラフ（llm-graph-builder等）に graphrag の後段パスを適用するバッチ。

llm-graph-builder + EDC が構築した共有グラフ（:Chunk / HAS_ENTITY / __Entity__）に対し、
Graph-First 検索に必要な派生プロパティを付与する:

  1. consolidate_post_build : 値ノードflag / 同一id分裂ノードのマージ /
                              かな揺れマージ / 関係名の正規化
                              ※ 同一idマージは gemma4/EDC のラベル違い分裂
                                （例: IHI:Organization と IHI:企業）の統合にもなる
  2. enrich_post_build      : mention_count / pagerank / search_keys
  3. GraphProvenance 刻印    : services.qa の KGゲートを通すために必須
                              （未刻印グラフは安全側でKGスキップされる）

冪等性: 1,2 とも再実行安全（マージ済みノードは再マージ対象にならない、
集計プロパティは上書き）。edge.extraction_count はこのバッチでは触らない。

使い方（GBの後処理 /post_processing 完了後に実行）:
  python scripts/enrich_external_graph.py \
      --neo4j-uri neo4j://192.168.0.250:7688 --neo4j-user neo4j --neo4j-pw *** \
      --pg-collection gb_mirror
  # 接続情報を省略すると .env (NEO4J_URI/NEO4J_USER/NEO4J_PW, PG_COLLECTION) を使う
  # --dry-run で対象件数の確認のみ / --skip-consolidate でマージ系をスキップ
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("enrich_external_graph")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", choices=["gb", "native"], default="gb",
                   help="gb: KG_CHUNK_LABEL=Chunk / KG_CHUNK_EDGE=HAS_ENTITY を設定（既定）")
    p.add_argument("--neo4j-uri", default=None, help="省略時は NEO4J_URI")
    p.add_argument("--neo4j-user", default=None, help="省略時は NEO4J_USER")
    p.add_argument("--neo4j-pw", default=None, help="省略時は NEO4J_PW")
    p.add_argument("--pg-collection", default=None,
                   help="GraphProvenance に刻印するPGコレクション名（省略時は PG_COLLECTION）")
    p.add_argument("--skip-consolidate", action="store_true",
                   help="マージ系（非可逆）をスキップし enrich のみ実行")
    p.add_argument("--skip-provenance", action="store_true",
                   help="GraphProvenance 刻印をスキップ")
    p.add_argument("--dry-run", action="store_true", help="対象件数を表示して終了（書き込みなし）")
    return p.parse_args()


def main():
    args = parse_args()

    # ── env はimportより先に確定させる（プロンプト等がimport時に評価されるため）──
    if args.profile == "gb":
        os.environ.setdefault("KG_CHUNK_LABEL", "Chunk")
        os.environ.setdefault("KG_CHUNK_EDGE", "HAS_ENTITY")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from langchain_neo4j import Neo4jGraph
    from graphrag_core.config import get_settings
    from graphrag_core.graph.schema import chunk_label, chunk_edge, entity_node_predicate
    from graphrag_core.graph.consolidate import consolidate_post_build
    from graphrag_core.graph.enrichment import enrich_post_build
    from graphrag_core.graph.provenance import stamp_graph_provenance

    s = get_settings()
    uri = args.neo4j_uri or s.neo4j_uri
    user = args.neo4j_user or s.neo4j_user
    pw = args.neo4j_pw or s.neo4j_pw
    pg_collection = args.pg_collection or s.pg_collection

    logger.info("profile=%s chunk_label=%s chunk_edge=%s", args.profile, chunk_label(), chunk_edge())
    logger.info("Neo4j: %s / provenance collection: %s", uri, pg_collection)

    graph = Neo4jGraph(url=uri, username=user, password=pw, enhanced_schema=False)

    # ── 対象の健全性チェック ──
    counts = {}
    counts["chunks"] = graph.query(
        f"MATCH (d:{chunk_label()}) RETURN count(d) AS c")[0]["c"]
    counts["chunks_with_text"] = graph.query(
        f"MATCH (d:{chunk_label()}) WHERE d.text IS NOT NULL RETURN count(d) AS c")[0]["c"]
    counts["chunk_edges"] = graph.query(
        "MATCH ()-[r:" + chunk_edge() + "]->() RETURN count(r) AS c")[0]["c"]
    counts["entities"] = graph.query(
        f"MATCH (n) WHERE {entity_node_predicate('n')} RETURN count(n) AS c")[0]["c"]
    counts["semantic_edges"] = graph.query(
        "MATCH (a)-[r]->(b) WHERE type(r) <> '" + chunk_edge() + "' "
        f"AND {entity_node_predicate('a')} AND {entity_node_predicate('b')} "
        "RETURN count(r) AS c")[0]["c"]
    logger.info("対象グラフ: %s", json.dumps(counts, ensure_ascii=False))

    if counts["chunks"] == 0:
        logger.error("チャンクノード (:%s) が0件です。プロファイル/接続先を確認してください", chunk_label())
        sys.exit(1)
    if counts["chunks_with_text"] == 0:
        logger.warning("チャンクに text プロパティがありません。KGソースチャンク取得が機能しません")

    if args.dry_run:
        logger.info("--dry-run のため終了（書き込みなし）")
        return

    # ── 1. consolidate（非可逆・冪等）──
    if args.skip_consolidate:
        logger.info("consolidate をスキップ")
    else:
        result = consolidate_post_build(graph)
        logger.info("consolidate 完了: %s", json.dumps(result, ensure_ascii=False, default=str))

    # ── 2. enrich（上書き再計算・冪等）──
    result = enrich_post_build(graph)
    logger.info("enrich 完了: %s", json.dumps(result, ensure_ascii=False, default=str))

    # ── 3. GraphProvenance 刻印（KGゲート通過に必須）──
    if args.skip_provenance:
        logger.info("provenance 刻印をスキップ")
    else:
        stamp_graph_provenance(graph, pg_collection, doc_count=counts["chunks"])
        logger.info("GraphProvenance 刻印完了: pg_collection=%s", pg_collection)

    logger.info("done.")


if __name__ == "__main__":
    main()
