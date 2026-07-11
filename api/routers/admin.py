"""ヘルスチェック・設定・管理系エンドポイント。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.deps import get_state, require_ready
from api.state import AppState
from graphrag_core.services.admin import clear_database, clear_graph, health_report

router = APIRouter(prefix="/api", tags=["admin"])


@router.get("/health")
def health(st: AppState = Depends(get_state)) -> dict:
    """総合ヘルス（degraded でも 200 で内容を返す。フロントの HealthGate が消費）。"""
    report = health_report(graph=st.graph, settings=st.settings)
    report["startup_error"] = st.startup_error
    return report


@router.get("/settings")
def settings_info(st: AppState = Depends(get_state)) -> dict:
    """実効設定（秘密除外ホワイトリスト）+ QA検索設定の既定値。"""
    s = st.settings
    from graphrag_core.llm.factory import get_llm_provider_info
    from graphrag_core.text.japanese import SUDACHI_AVAILABLE

    return {
        "collection": s.pg_collection,
        "llm": get_llm_provider_info(),
        "embedding_provider": s.embedding_provider,
        "reranker_enabled": s.reranker_enabled,
        "sudachi_available": bool(SUDACHI_AVAILABLE),
        # UI検索設定9キーの既定値（フロント settingsStore の初期値）
        "qa_defaults": {
            "retrieval_top_k": s.retrieval_top_k,
            "enable_rerank": s.enable_rerank,
            "enable_japanese_search": s.enable_japanese_search,
            "search_mode": s.search_mode,
            "enable_knowledge_graph": s.enable_knowledge_graph,
            "include_kg_source_chunks": s.include_kg_source_chunks,
            "graph_hop_count": s.graph_hop_count,
            "enable_entity_vector": s.enable_entity_vector_search,
            "entity_similarity_threshold": s.entity_similarity_threshold,
        },
    }


# ── コレクション切替（実行時・再起動不要） ─────────────────────────
class SwitchCollectionBody(BaseModel):
    name: str


@router.get("/admin/collections")
def list_collections(st: AppState = Depends(require_ready)) -> dict:
    """PGVectorの全コレクション一覧（切替UI用）。現在の選択と出自グラフも返す。"""
    from graphrag_core.services.admin import list_pg_collections
    from graphrag_core.graph.provenance import graph_collection_status
    cols = list_pg_collections(st.settings.pg_conn)
    prov = graph_collection_status(st.graph, st.settings.pg_collection)
    return {
        "current": st.settings.pg_collection,
        "graph_collection": prov.get("graph_collection"),
        "collections": cols,
    }


@router.post("/admin/collection")
def switch_collection(body: SwitchCollectionBody,
                      st: AppState = Depends(require_ready)) -> dict:
    """検索対象コレクションを切り替える（.graphrag_runtime.json に永続化）。"""
    from graphrag_core.services.admin import list_pg_collections
    from graphrag_core.graph.provenance import graph_collection_status
    names = {c["name"] for c in list_pg_collections(st.settings.pg_conn)}
    if body.name not in names:
        raise HTTPException(400, f"コレクションが存在しません: {body.name}")
    result = st.switch_collection(body.name)
    prov = graph_collection_status(st.graph, body.name)
    return {**result, "provenance": prov}


# ── 破壊的操作（サーバ側でも確認を担保） ──────────────────────────
class ClearDatabaseBody(BaseModel):
    confirm: str  # 誤操作防止: 現コレクション名の完全一致が必須


@router.post("/admin/clear-graph")
def admin_clear_graph(st: AppState = Depends(require_ready)) -> dict:
    """Neo4j グラフのみ全削除（PGVector は残す）。"""
    result = clear_graph(st.graph)
    st.invalidate_retrieval()
    return result


@router.post("/admin/clear-database")
def admin_clear_database(body: ClearDatabaseBody, st: AppState = Depends(require_ready)) -> dict:
    """Neo4j 全削除 + PGVector 現コレクション削除。confirm=コレクション名 必須。"""
    if body.confirm != st.settings.pg_collection:
        raise HTTPException(
            400,
            f"確認文字列が一致しません（現在のコレクション名 '{st.settings.pg_collection}' を入力してください）",
        )
    result = clear_database(graph=st.graph, settings=st.settings)
    st.invalidate_retrieval()
    return result
