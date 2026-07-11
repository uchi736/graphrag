"""管理系サービス: ヘルスチェック / グラフ統計 / クリア操作。

ui/sidebar.py のハードガード（環境変数・Neo4j接続）と _clear_database、
ui/state.py の check_existing_graph を st 非依存化して集約。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from graphrag_core.config import get_settings

logger = logging.getLogger(__name__)


# ── 必須環境変数チェック（sidebar.py:52-71 移植） ──────────────────────
def required_env_check(settings=None) -> Dict[str, Any]:
    """プロバイダ構成に応じた必須環境変数の充足を検査する。

    Returns: {"ok": bool, "missing": [env名,...]}
    """
    s = settings or get_settings()
    required = {"PG_CONN": s.pg_conn}

    llm_provider = (s.llm_provider or "").lower()
    if llm_provider == "azure_openai":
        required["AZURE_OPENAI_API_KEY"] = s.azure_openai_api_key
        required["AZURE_OPENAI_ENDPOINT"] = s.azure_openai_endpoint
        required["AZURE_OPENAI_API_VERSION"] = s.azure_openai_api_version
        required["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = s.azure_openai_chat_deployment
    elif llm_provider == "vllm":
        required["VLLM_ENDPOINT"] = s.vllm_endpoint

    emb_provider = (s.embedding_provider or "").lower()
    if emb_provider == "azure_openai":
        required["AZURE_OPENAI_API_KEY"] = s.azure_openai_api_key
        required["AZURE_OPENAI_ENDPOINT"] = s.azure_openai_endpoint
        required["AZURE_OPENAI_API_VERSION"] = s.azure_openai_api_version
        required["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"] = s.azure_openai_embedding_deployment
    elif emb_provider == "vllm":
        required["VLLM_EMBEDDING_ENDPOINT"] = s.vllm_embedding_endpoint

    if not all([s.neo4j_uri, s.neo4j_user, s.neo4j_pw]):
        for name, val in (("NEO4J_URI", s.neo4j_uri), ("NEO4J_USER", s.neo4j_user),
                          ("NEO4J_PW", s.neo4j_pw)):
            required[name] = val

    missing = [name for name, value in required.items() if not value]
    return {"ok": not missing, "missing": missing}


def pdf_processor_status(settings=None) -> Dict[str, Any]:
    """PDF前処理経路のステータス（sidebar.py:110-127 移植）。"""
    s = settings or get_settings()
    processor = (s.pdf_processor or "").lower()
    info: Dict[str, Any] = {"processor": processor or "pymupdf", "backend": None,
                            "endpoint": None, "ok": True, "note": ""}
    if processor == "onprem":
        backend = (s.pdf_backend or "vllm").lower()
        info["backend"] = backend
        if backend == "paddleocr_remote":
            info["endpoint"] = s.paddlex_endpoint
            info["ok"] = bool(s.paddlex_endpoint)
        elif backend == "vllm":
            info["endpoint"] = s.vllm_vision_endpoint
            info["ok"] = bool(s.vllm_vision_endpoint)
        elif backend == "none":
            info["note"] = "画像解析なし"
        if not info["ok"]:
            info["note"] = "エンドポイント未設定の可能性"
    elif processor == "azure_di":
        info["ok"] = bool(s.azure_di_endpoint)
        if not info["ok"]:
            info["note"] = "AZURE_DI_ENDPOINT 未設定"
    else:
        info["note"] = "PyMuPDF プレーンテキスト抽出"
    return info


# ── グラフ統計（state.py check_existing_graph の純粋版） ──────────────
def graph_stats(graph) -> Dict[str, Any]:
    """Neo4j のノード/リレーション数。接続エラーは exists=False + error で返す。"""
    try:
        result = graph.query("MATCH (n) RETURN count(n) AS node_count")
        node_count = result[0]["node_count"] if result else 0
        rel_count = 0
        if node_count > 0:
            rel = graph.query("MATCH ()-[r]->() RETURN count(r) AS rel_count")
            rel_count = rel[0]["rel_count"] if rel else 0
        return {"exists": node_count > 0, "node_count": node_count,
                "rel_count": rel_count, "error": None}
    except Exception as e:
        logger.warning("graph_stats failed: %s", e)
        return {"exists": False, "node_count": 0, "rel_count": 0, "error": str(e)[:200]}


# ── ヘルスレポート（/api/health の中身） ──────────────────────────────
def health_report(graph=None, settings=None) -> Dict[str, Any]:
    """環境変数・Neo4j・LLM/embedding 構成・Sudachi の総合ステータス。

    graph に既存接続を渡すと再接続せず統計を取る。None なら接続試行。
    起動不能レベルの不足でも例外にせず degraded 応答を返す。
    """
    s = settings or get_settings()
    env = required_env_check(s)

    neo4j: Dict[str, Any] = {"ok": False, "error": None}
    g_stats: Dict[str, Any] = {"exists": False, "node_count": 0, "rel_count": 0}
    provenance: Dict[str, Any] = {"status": "unknown", "graph_collection": None}
    close_after = False
    try:
        if graph is None:
            from langchain_neo4j import Neo4jGraph
            graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user,
                               password=s.neo4j_pw, enhanced_schema=False)
            close_after = True
        g_stats = graph_stats(graph)
        neo4j["ok"] = g_stats.get("error") is None
        neo4j["error"] = g_stats.get("error")
        if neo4j["ok"]:
            from graphrag_core.graph.provenance import graph_collection_status
            provenance = graph_collection_status(graph, s.pg_collection)
    except Exception as e:
        neo4j["error"] = str(e)[:200]
    finally:
        if close_after and graph is not None:
            try:
                graph._driver.close()
            except Exception:
                pass

    from graphrag_core.llm.factory import get_llm_provider_info
    from graphrag_core.text.japanese import SUDACHI_AVAILABLE

    llm_info = get_llm_provider_info()
    ok = env["ok"] and neo4j["ok"]
    return {
        "ok": ok,
        "status": "ok" if ok else "degraded",
        "checks": {
            "env": env,
            "neo4j": neo4j,
            "llm": {"provider": llm_info.get("provider"), "model": llm_info.get("model"),
                    "status": llm_info.get("status")},
            "embedding": {"provider": s.embedding_provider},
            "pdf": pdf_processor_status(s),
            "sudachi_available": bool(SUDACHI_AVAILABLE),
        },
        "graph": g_stats,
        "provenance": provenance,
        "collection": s.pg_collection,
    }


# ── クリア操作（sidebar.py:150-181 / build_tab.py:69 移植） ───────────
def clear_graph(graph) -> Dict[str, Any]:
    """Neo4j のみ全削除（「クリアして新規作成」相当）。"""
    graph.query("MATCH (n) DETACH DELETE n")
    return {"ok": True}


def clear_database(graph=None, settings=None) -> Dict[str, Any]:
    """Neo4j 全削除 + PGVector 現コレクション削除。

    警告（PG側の失敗）は例外にせず warnings で返す（現行挙動踏襲）。
    """
    s = settings or get_settings()
    warnings = []

    if graph is None:
        from langchain_neo4j import Neo4jGraph
        graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user,
                           password=s.neo4j_pw, enhanced_schema=False)
    graph.query("MATCH (n) DETACH DELETE n")

    if s.pg_conn:
        try:
            import psycopg
            from graphrag_core.db.utils import normalize_pg_connection_string
            raw_conn = normalize_pg_connection_string(s.pg_conn)
            with psycopg.connect(raw_conn) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM langchain_pg_embedding e
                        USING langchain_pg_collection c
                        WHERE e.collection_id = c.uuid AND c.name = %s
                    """, (s.pg_collection,))
                    cur.execute("DELETE FROM langchain_pg_collection WHERE name = %s",
                                (s.pg_collection,))
                conn.commit()
        except Exception as e:
            warnings.append(f"PGVectorクリアで警告: {e}")
            logger.warning("clear_database PG side failed: %s", e)

    return {"ok": True, "warnings": warnings}


def list_pg_collections(pg_conn: str) -> list:
    """PGVector の全コレクションとチャンク数を返す（コレクション切替UI用）。

    `_entities` 系（エンティティベクトル格納用）は文書コレクションではないため除外。
    """
    import psycopg
    from graphrag_core.db.utils import normalize_pg_connection_string
    raw_conn = normalize_pg_connection_string(pg_conn)
    with psycopg.connect(raw_conn) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.name, count(e.id)
                FROM langchain_pg_collection c
                LEFT JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
                GROUP BY c.name ORDER BY 2 DESC
            """)
            rows = cur.fetchall()
    return [{"name": r[0], "chunks": r[1]} for r in rows
            if not r[0].endswith("_entities")]
