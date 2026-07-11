"""
Unified CRUD Dispatcher
=======================
Neo4j グラフバックエンド用の統一 CRUD ディスパッチャ。
neo4j_ops の関数を薄くラップし、共通シグネチャを提供する。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from graphrag_core.graph.neo4j_ops import (
    neo4j_add_node,
    neo4j_update_node,
    neo4j_delete_node,
    neo4j_get_node_info,
    neo4j_add_edge,
    neo4j_update_edge,
    neo4j_delete_edge,
    neo4j_get_edge_info,
    neo4j_list_all_nodes,
    neo4j_list_all_edges,
)


# ------------------------------------------------------------------
# ノード操作
# ------------------------------------------------------------------

def graph_add_node(
    graph,
    node_id: str,
    node_type: str,
    properties: Optional[Dict[str, Any]],
) -> bool:
    """ノード追加"""
    return neo4j_add_node(graph, node_id, node_type, properties)


def graph_update_node(
    graph,
    node_id: str,
    node_type: Optional[str],
    properties: Optional[Dict[str, Any]],
) -> bool:
    """ノード更新"""
    return neo4j_update_node(graph, node_id, node_type, properties)


def graph_delete_node(
    graph,
    node_id: str,
) -> bool:
    """ノード削除"""
    return neo4j_delete_node(graph, node_id)


def graph_get_node_info(
    graph,
    node_id: str,
) -> Optional[Dict[str, Any]]:
    """ノード情報取得"""
    return neo4j_get_node_info(graph, node_id)


# ------------------------------------------------------------------
# エッジ操作
# ------------------------------------------------------------------

def graph_add_edge(
    graph,
    source: str,
    target: str,
    rel_type: str,
    properties: Optional[Dict[str, Any]],
) -> Optional[int]:
    """エッジ追加"""
    return neo4j_add_edge(graph, source, target, rel_type, properties)


def graph_update_edge(
    graph,
    source: str,
    target: str,
    edge_key: int,
    rel_type: Optional[str],
    properties: Optional[Dict[str, Any]],
) -> bool:
    """エッジ更新"""
    return neo4j_update_edge(graph, source, target, edge_key, rel_type, properties)


def graph_delete_edge(
    graph,
    source: str,
    target: str,
    edge_key: Optional[int],
) -> bool:
    """エッジ削除"""
    return neo4j_delete_edge(graph, source, target, edge_key)


def graph_get_edge_info(
    graph,
    source: str,
    target: str,
    edge_key: int,
) -> Optional[Dict[str, Any]]:
    """エッジ情報取得"""
    return neo4j_get_edge_info(graph, source, target, edge_key)


# ------------------------------------------------------------------
# リスト取得
# ------------------------------------------------------------------

def graph_list_all_nodes(graph) -> List[Dict[str, Any]]:
    """全ノード一覧"""
    return neo4j_list_all_nodes(graph)


def graph_list_all_edges(graph) -> List[Dict[str, Any]]:
    """全エッジ一覧"""
    return neo4j_list_all_edges(graph)


# ------------------------------------------------------------------
# データ取得 (キャッシュ / 可視化用)
# ------------------------------------------------------------------

def graph_get_data_for_cache(graph) -> List[Dict[str, Any]]:
    """グラフデータ取得（キャッシュ用）"""
    try:
        from graphrag_core.graph.schema import chunk_edge
        result = graph.query(
            "MATCH (s)-[r]->(t) "
            "WHERE type(r) <> '" + chunk_edge() + "' "
            "AND NOT s.id =~ '[0-9a-f]{32,}' AND NOT t.id =~ '[0-9a-f]{32,}' "
            "RETURN s.id AS source, type(r) AS relation, t.id AS target "
            "LIMIT 200"
        )
        return [
            {"source": r["source"], "relation": r["relation"], "target": r["target"], "edge_key": 0}
            for r in (result or [])
        ]
    except Exception:
        return []
