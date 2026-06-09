"""
graphrag_core.graph - グラフ操作パッケージ (Neo4j)
"""
from graphrag_core.graph.base import GraphBackend
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
    export_graph_json,
)
from graphrag_core.graph.crud import (
    graph_add_node,
    graph_update_node,
    graph_delete_node,
    graph_get_node_info,
    graph_add_edge,
    graph_update_edge,
    graph_delete_edge,
    graph_get_edge_info,
    graph_list_all_nodes,
    graph_list_all_edges,
    graph_get_data_for_cache,
)

__all__ = [
    # Protocol
    "GraphBackend",
    # Neo4j ops
    "neo4j_add_node",
    "neo4j_update_node",
    "neo4j_delete_node",
    "neo4j_get_node_info",
    "neo4j_add_edge",
    "neo4j_update_edge",
    "neo4j_delete_edge",
    "neo4j_get_edge_info",
    "neo4j_list_all_nodes",
    "neo4j_list_all_edges",
    "export_graph_json",
    # Unified CRUD
    "graph_add_node",
    "graph_update_node",
    "graph_delete_node",
    "graph_get_node_info",
    "graph_add_edge",
    "graph_update_edge",
    "graph_delete_edge",
    "graph_get_edge_info",
    "graph_list_all_nodes",
    "graph_list_all_edges",
    "graph_get_data_for_cache",
]
