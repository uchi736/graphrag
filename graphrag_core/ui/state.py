"""
graphrag_core/ui/state.py
Central session-state init + the shared UIContext.

Live, rerun-changing objects (chain, graph) live in st.session_state, NOT in the
dataclass; ctx exposes live read-only properties for them. The dataclass carries
env constants, the Settings object, stateless factory callables, the graph CRUD
callbacks, and the two dialog wrappers — assembled ONCE per script run.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import streamlit as st
from langchain_neo4j import Neo4jGraph
from langchain_postgres import PGVector

from graphrag_core.config import get_settings, build_pipeline_config
from graphrag_core.llm.factory import create_embeddings as _factory_create_embeddings
from graphrag_core.db.utils import (
    normalize_pg_connection_string,
    add_connection_timeout,
    retry_on_timeout,
)
from graphrag_core.graph.crud import (
    graph_add_node, graph_update_node, graph_delete_node, graph_get_node_info,
    graph_add_edge, graph_update_edge, graph_delete_edge, graph_get_edge_info,
    graph_get_data_for_cache,
)
from graphrag_core.ui.dialogs import (
    edit_node_dialog as _edit_node_dialog,
    edit_edge_dialog as _edit_edge_dialog,
)

# ParentDocumentRetriever version shim (community -> langchain fallback)
try:
    from langchain_community.retrievers.parent_document import ParentDocumentRetriever
    HAS_PARENT = True
except ImportError:
    try:
        from langchain.retrievers.parent_document import ParentDocumentRetriever
        HAS_PARENT = True
    except ImportError:
        HAS_PARENT = False


# =====================================================================
# graph CRUD callback bundle (passed to data_tables / dialog wrappers)
# =====================================================================
CRUD_CALLBACKS: dict = {
    "graph_add_node": graph_add_node,
    "graph_update_node": graph_update_node,
    "graph_delete_node": graph_delete_node,
    "graph_get_node_info": graph_get_node_info,
    "graph_add_edge": graph_add_edge,
    "graph_update_edge": graph_update_edge,
    "graph_delete_edge": graph_delete_edge,
    "graph_get_edge_info": graph_get_edge_info,
    "graph_get_data_for_cache": graph_get_data_for_cache,
}


# =====================================================================
# dialog wrappers (CRUD already bound) — moved from app._wrapped_*
# =====================================================================
def wrapped_edit_node_dialog(graph, node_info):
    _edit_node_dialog(
        graph, node_info,
        graph_update_node=graph_update_node,
        graph_add_node=graph_add_node,
        graph_get_data_for_cache=graph_get_data_for_cache,
    )


def wrapped_edit_edge_dialog(graph, edge_info, all_nodes=None):
    _edit_edge_dialog(
        graph, edge_info, all_nodes,
        graph_update_edge=graph_update_edge,
        graph_add_edge=graph_add_edge,
        graph_get_data_for_cache=graph_get_data_for_cache,
    )


# =====================================================================
# shared helpers (used by build_tab/system AND graph_tab)
# =====================================================================
def create_embeddings():
    return _factory_create_embeddings()


def create_graph_instance(enhanced_schema: bool = False) -> Neo4jGraph:
    s = get_settings()
    return Neo4jGraph(
        url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw,
        enhanced_schema=enhanced_schema,
    )


def create_vector_retriever(vector_store, top_k: int):
    """バージョン差異を吸収してRetrieverを構築する。"""
    if vector_store is None:
        return None
    if HAS_PARENT:
        try:
            return ParentDocumentRetriever(
                vectorstore=vector_store, search_kwargs={"k": top_k},
            )
        except Exception as e:
            print(f"[Retriever] ParentDocumentRetriever unavailable. "
                  f"fallback=vector_store.as_retriever ({e})")
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def check_existing_graph(graph) -> dict:
    """Neo4jに既存のグラフデータがあるかチェック"""
    try:
        result = graph.query("MATCH (n) RETURN count(n) AS node_count")
        node_count = result[0]["node_count"] if result else 0
        if node_count > 0:
            rel = graph.query("MATCH ()-[r]->() RETURN count(r) AS rel_count")
            rel_count = rel[0]["rel_count"] if rel else 0
            return {"exists": True, "node_count": node_count, "rel_count": rel_count}
        return {"exists": False, "node_count": 0, "rel_count": 0}
    except Exception as e:
        st.error(f"グラフ接続エラー: {e}")
        return {"exists": False, "node_count": 0, "rel_count": 0}


def _build_config_from_session_state() -> dict:
    """Settings 由来の既定に st.session_state の対話値を overlay する。"""
    log = st.session_state.get("path_rerank_log", [])
    st.session_state["path_rerank_log"] = log
    s = get_settings()
    overrides = {}
    for key in (
        "graph_hop_count", "retrieval_top_k", "enable_japanese_search",
        "enable_rerank", "enable_entity_vector", "entity_similarity_threshold",
        "search_mode", "include_kg_source_chunks",
    ):
        if key in st.session_state:
            overrides[key] = st.session_state[key]
    # 旧挙動の維持: KG生成OFFで起動し enable_entity_vector 未設定なら entity-vector OFF
    if "enable_entity_vector" not in st.session_state:
        overrides["enable_entity_vector"] = False
    config = build_pipeline_config(s, **overrides)
    config["_path_rerank_log"] = log
    return config


# =====================================================================
# UIContext — assembled once per run, passed to every render_*_tab(ctx)
# =====================================================================
@dataclass
class UIContext:
    # ---- env constants (captured once from Settings) ----
    neo4j_uri: str
    neo4j_user: str
    neo4j_pw: str
    pg_conn: str
    pg_collection: str
    settings: Any                      # get_settings() object (system reads .llm_provider)
    # ---- sidebar-derived visualization settings ----
    show_graph: bool = True
    max_nodes: int = 200
    # ---- stateless factory / helper callables ----
    create_embeddings: Callable[[], Any] = create_embeddings
    normalize_pg_connection_string: Callable[[str], str] = normalize_pg_connection_string
    # ---- graph CRUD callbacks ----
    crud: dict = field(default_factory=lambda: dict(CRUD_CALLBACKS))
    # ---- dialog wrappers (CRUD already bound) ----
    edit_node_dialog: Optional[Callable] = wrapped_edit_node_dialog
    edit_edge_dialog: Optional[Callable] = wrapped_edit_edge_dialog

    # ---- helper methods ----
    def create_graph_instance(self, enhanced_schema: bool = False) -> Neo4jGraph:
        return Neo4jGraph(url=self.neo4j_uri, username=self.neo4j_user,
                          password=self.neo4j_pw, enhanced_schema=enhanced_schema)

    def build_config(self) -> dict:
        return _build_config_from_session_state()

    def make_vector_store(self) -> PGVector:
        emb = self.create_embeddings()
        conn = add_connection_timeout(self.pg_conn, timeout=30)
        return retry_on_timeout(
            lambda: PGVector(connection=conn, embeddings=emb,
                             collection_name=self.pg_collection),
            max_retries=3, delay=2.0,
        )

    # ---- live runtime objects (in session_state, never frozen on ctx) ----
    @property
    def graph(self):
        return st.session_state.get("graph")

    @property
    def chain(self):
        return st.session_state.get("chain")

    @property
    def initialized(self) -> bool:
        return bool(st.session_state.get("initialized", False))


# =====================================================================
# session-state init + context assembly
# =====================================================================
def initialize_session_state() -> None:
    """全タブが読む st.session_state キーを冪等にシードする。"""
    defaults = {
        "max_nodes": 200,
        "show_graph": True,
        "langfuse_session_id": str(uuid.uuid4()),
        "chain": None,
        "graph": None,
        "initialized": False,
        "uploaded_files": [],
        "existing_graph_loaded": False,
        "graph_data_cache": None,
        "last_qa": None,
        "qa_history": [],
        "busy": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def build_ui_context(sidebar_config: dict) -> UIContext:
    s = get_settings()
    # langchain-postgres reads PGVECTOR_CONNECTION_STRING (module-level side-effect
    # in old app.py; performed here where Settings is loaded — behavior identical)
    if s.pg_conn and not os.getenv("PGVECTOR_CONNECTION_STRING"):
        os.environ["PGVECTOR_CONNECTION_STRING"] = s.pg_conn
    # show_graph / max_nodes は旧サイドバーの返り値ではなく、設定タブが書き込む
    # st.session_state から読む（initialize_session_state がデフォルトをシード）。
    return UIContext(
        neo4j_uri=s.neo4j_uri, neo4j_user=s.neo4j_user, neo4j_pw=s.neo4j_pw,
        pg_conn=s.pg_conn, pg_collection=s.pg_collection, settings=s,
        show_graph=st.session_state.get("show_graph", True),
        max_nodes=st.session_state.get("max_nodes", 200),
        crud=dict(CRUD_CALLBACKS),
        edit_node_dialog=wrapped_edit_node_dialog,
        edit_edge_dialog=wrapped_edit_edge_dialog,
    )


def get_context(sidebar_config: dict) -> UIContext:
    """Rebuild each run so sidebar-derived show_graph/max_nodes stay fresh;
    cache on st.session_state so all tabs in this run share one instance."""
    ctx = build_ui_context(sidebar_config)
    st.session_state["_ui_ctx"] = ctx
    return ctx
