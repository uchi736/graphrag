"""グラフ出自(provenance)の刻印と整合性チェック

Neo4j は1個の共有グラフDBで「どのPGVectorコレクションから作られたか」を
追跡しない。そのため `.env` の PG_COLLECTION を変えてもグラフは据え置きで、
ベクトル(コレクション) と グラフ が別コーパスになる不整合が起き得る。

ここではグラフに `(:GraphProvenance {kind:'active'})` ノードで「出自コレクション」
を刻印し、実行時に現在の PG_COLLECTION と照合する。照合結果が match 以外なら
呼び出し側は KG 探索をスキップする（不整合ハイブリッドを構造的に回避）。

GraphProvenance ノードは `id` を持たせないため、entity_node_predicate
(`id IS NOT NULL`) およびハッシュID/管理ノード除外により検索対象に入らない。
schema.entity_node_predicate にもラベル除外を明示済み。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

PROVENANCE_LABEL = "GraphProvenance"


def stamp_graph_provenance(graph, pg_collection: str, doc_count: Optional[int] = None) -> bool:
    """現在のグラフに出自コレクションを刻印する（構築/チャンク更新の直後に呼ぶ）。

    冪等: `kind:'active'` の単一ノードを MERGE して上書きする。
    """
    try:
        graph.query(
            """
            MERGE (p:GraphProvenance {kind: 'active'})
            SET p.pg_collection = $pg_collection,
                p.doc_count = $doc_count,
                p.stamped_at = datetime($stamped_at)
            """,
            params={
                "pg_collection": pg_collection,
                "doc_count": doc_count,
                "stamped_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info("Stamped GraphProvenance: pg_collection=%s", pg_collection)
        return True
    except Exception as e:
        logger.warning("Failed to stamp GraphProvenance: %s", e)
        return False


def get_graph_provenance(graph) -> Optional[dict]:
    """刻印済みの出自を返す。未刻印なら None。"""
    try:
        result = graph.query(
            "MATCH (p:GraphProvenance {kind: 'active'}) "
            "RETURN p.pg_collection AS pg_collection, p.doc_count AS doc_count, "
            "toString(p.stamped_at) AS stamped_at "
            "LIMIT 1"
        )
        if result:
            return {
                "pg_collection": result[0].get("pg_collection"),
                "doc_count": result[0].get("doc_count"),
                "stamped_at": result[0].get("stamped_at"),
            }
    except Exception as e:
        logger.warning("Failed to read GraphProvenance: %s", e)
    return None


def graph_collection_status(graph, pg_collection: str) -> dict:
    """グラフの出自と現在の PG_COLLECTION の整合性を返す。

    Returns:
        {"status": "match"|"mismatch"|"unknown", "graph_collection": str|None}
        - match    : 刻印された出自 == 現コレクション → KG を使ってよい
        - mismatch : 刻印された出自 != 現コレクション → KG をスキップすべき
        - unknown  : 未刻印（出自不明、レガシー/手動構築） → 安全側で KG をスキップすべき
    """
    if graph is None:
        return {"status": "unknown", "graph_collection": None}
    prov = get_graph_provenance(graph)
    if not prov or not prov.get("pg_collection"):
        return {"status": "unknown", "graph_collection": None}
    gc = prov["pg_collection"]
    if gc == pg_collection:
        return {"status": "match", "graph_collection": gc}
    return {"status": "mismatch", "graph_collection": gc}


def graph_matches_collection(graph, pg_collection: str) -> bool:
    """KG を使ってよいか（status == 'match' のときだけ True）。"""
    return graph_collection_status(graph, pg_collection)["status"] == "match"
