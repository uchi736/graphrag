"""グラフ探索・CRUD・Cypher・ステータス系エンドポイント。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.deps import require_ready
from api.state import AppState
from graphrag_core.services.admin import graph_stats
from graphrag_core.services.graph_explore import (
    WriteQueryRejected,
    execute_readonly_cypher,
    get_enhanced_graph_data,
    get_enhanced_subgraph_data,
    natural_language_to_cypher,
    search_node_ids,
)
from graphrag_core.graph.crud import (
    graph_add_edge,
    graph_add_node,
    graph_delete_edge,
    graph_delete_node,
    graph_get_edge_info,
    graph_get_node_info,
    graph_update_edge,
    graph_update_node,
)
from graphrag_core.graph.provenance import graph_collection_status, stamp_graph_provenance

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/status")
def status(st: AppState = Depends(require_ready)) -> dict:
    stats = graph_stats(st.graph)
    prov = graph_collection_status(st.graph, st.settings.pg_collection)
    return {"graph": stats, "provenance": prov, "collection": st.settings.pg_collection}


@router.post("/provenance")
def stamp_provenance(st: AppState = Depends(require_ready)) -> dict:
    ok = stamp_graph_provenance(st.graph, st.settings.pg_collection)
    return {"ok": bool(ok), "collection": st.settings.pg_collection}


@router.get("/overview")
def overview(
    limit: int = Query(200, ge=1, le=5000),
    st: AppState = Depends(require_ready),
) -> list:
    """全体エッジリスト（EdgeRecord[]。可視化・データテーブル共用）。"""
    return get_enhanced_graph_data(st.graph, limit=limit)


@router.get("/subgraph")
def subgraph(
    center: List[str] = Query(..., min_length=1),
    hop: int = Query(1, ge=1, le=3),
    limit: int = Query(500, ge=1, le=2000),
    st: AppState = Depends(require_ready),
) -> list:
    """指定ノード群を中心とした N-hop サブグラフ。"""
    return get_enhanced_subgraph_data(st.graph, center_nodes=center, hop=hop, limit=limit)


@router.get("/node-ids")
def node_ids(
    q: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    st: AppState = Depends(require_ready),
) -> dict:
    """ノードID部分一致検索（中心ノード選択用）。"""
    ids = search_node_ids(st.graph, q=q or "", limit=limit)
    return {"ids": ids}


# =====================================================================
# CRUD（ノード/エッジ。idは日本語を含むため query param / body で受ける）
# =====================================================================
class NodeBody(BaseModel):
    id: str = Field(min_length=1)
    type: str = "Term"
    properties: Dict[str, Any] = {}


class NodeUpdateBody(BaseModel):
    id: str = Field(min_length=1)
    type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None


class EdgeBody(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    rel_type: str = "RELATED"
    properties: Dict[str, Any] = {}


class EdgeUpdateBody(BaseModel):
    source: str
    target: str
    edge_key: int = 0
    rel_type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None


@router.get("/node")
def get_node(id: str = Query(...), st: AppState = Depends(require_ready)) -> dict:
    info = graph_get_node_info(st.graph, id)
    if info is None:
        raise HTTPException(404, f"ノードが見つかりません: {id}")
    return info


@router.post("/node")
def add_node(body: NodeBody, st: AppState = Depends(require_ready)) -> dict:
    ok = graph_add_node(st.graph, body.id, body.type, body.properties)
    if not ok:
        raise HTTPException(400, "ノード追加に失敗しました")
    return {"ok": True}


@router.put("/node")
def update_node(body: NodeUpdateBody, st: AppState = Depends(require_ready)) -> dict:
    ok = graph_update_node(st.graph, body.id, body.type, body.properties)
    if not ok:
        raise HTTPException(400, "ノード更新に失敗しました")
    return {"ok": True}


@router.delete("/node")
def delete_node(id: str = Query(...), st: AppState = Depends(require_ready)) -> dict:
    ok = graph_delete_node(st.graph, id)
    if not ok:
        raise HTTPException(400, "ノード削除に失敗しました")
    return {"ok": True}


@router.get("/edge")
def get_edge(source: str = Query(...), target: str = Query(...),
             edge_key: int = Query(0), st: AppState = Depends(require_ready)) -> dict:
    info = graph_get_edge_info(st.graph, source, target, edge_key)
    if info is None:
        raise HTTPException(404, "エッジが見つかりません")
    return info


@router.post("/edge")
def add_edge(body: EdgeBody, st: AppState = Depends(require_ready)) -> dict:
    key = graph_add_edge(st.graph, body.source, body.target, body.rel_type, body.properties)
    if key is None:
        raise HTTPException(400, "エッジ追加に失敗しました（両端ノードの存在を確認してください）")
    return {"ok": True, "edge_key": key}


@router.put("/edge")
def update_edge(body: EdgeUpdateBody, st: AppState = Depends(require_ready)) -> dict:
    ok = graph_update_edge(st.graph, body.source, body.target, body.edge_key,
                           body.rel_type, body.properties)
    if not ok:
        raise HTTPException(400, "エッジ更新に失敗しました")
    return {"ok": True}


@router.delete("/edge")
def delete_edge(source: str = Query(...), target: str = Query(...),
                edge_key: Optional[int] = Query(None),
                st: AppState = Depends(require_ready)) -> dict:
    ok = graph_delete_edge(st.graph, source, target, edge_key)
    if not ok:
        raise HTTPException(400, "エッジ削除に失敗しました")
    return {"ok": True}


# =====================================================================
# Cypher（NL変換 + 参照専用実行）
# =====================================================================
class CypherGenerateBody(BaseModel):
    query: str = Field(min_length=1, description="自然言語クエリ")


class CypherExecuteBody(BaseModel):
    cypher: str = Field(min_length=1)


@router.post("/cypher/generate")
def cypher_generate(body: CypherGenerateBody, st: AppState = Depends(require_ready)) -> dict:
    cypher = natural_language_to_cypher(body.query, llm=st.llm)
    return {"cypher": cypher}


@router.post("/cypher/execute")
def cypher_execute(body: CypherExecuteBody, st: AppState = Depends(require_ready)) -> dict:
    try:
        result = execute_readonly_cypher(st.graph, body.cypher)
    except WriteQueryRejected as e:
        raise HTTPException(400, str(e))
    rows = result["rows"]
    columns = list(rows[0].keys()) if rows else []
    return {
        "columns": columns,
        "rows": rows,
        "applied_limit": result["applied_limit"],
        "executed_query": result["executed_query"],
    }
