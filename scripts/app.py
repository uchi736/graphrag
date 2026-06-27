"""
Streamlit UI for Graph-RAG — thin orchestrator.
全ロジックは graphrag_core.ui.* モジュールに委譲。
"""
import os
import sys
import logging
import warnings
from pathlib import Path

# Ensure project root is importable
_root = Path(__file__).parent.parent.resolve()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st
from dotenv import load_dotenv

# ── top-matter: ノイズ抑制 ────────────────────────────────────────
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

load_dotenv()

# ── ENV_DEFAULTS (module level) ───────────────────────────────────
ENV_DEFAULTS = {
    "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
    "PG_COLLECTION": os.getenv("PG_COLLECTION", ""),
    "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "azure_openai"),
}

# ── import-error guard around cross-module imports ────────────────
try:
    from langchain_neo4j import Neo4jGraph
    from graphrag_core.llm.factory import get_llm_provider_info
    from graphrag_core.db.utils import normalize_pg_connection_string
    from graphrag_core.text.japanese import SUDACHI_AVAILABLE
    from graphrag_core.ui.css import CUSTOM_CSS, HEADER_HTML
    from graphrag_core.ui.sidebar import render_sidebar
    from graphrag_core.ui.state import initialize_session_state, get_context
    from graphrag_core.ui.qa_tab import render_qa_tab
    from graphrag_core.ui.graph_tab import render_graph_tab
    from graphrag_core.ui.documents_tab import render_documents_tab
    from graphrag_core.ui.build_tab import render_build_tab
    from graphrag_core.ui.settings_tab import render_settings_tab
except ImportError as e:
    st.set_page_config(page_title="Graph-RAG", layout="wide")
    st.error(f"モジュール読み込みエラー: {e}")
    st.stop()


def main():
    # page config first（サイドバーは expanded のまま＝設定中心UI）
    st.set_page_config(page_title="GraphRAG", page_icon="🔗", layout="wide")

    # CSS + グラデーション ヘッダー
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(HEADER_HTML, unsafe_allow_html=True)

    # session state（sidebar/context より前に）
    initialize_session_state()

    # サイドバー: 設定中心、デフォルト展開
    sidebar_config = render_sidebar(settings={
        "get_llm_provider_info": get_llm_provider_info,
        "neo4j_graph_class": Neo4jGraph,
        "sudachi_available": SUDACHI_AVAILABLE,
        "normalize_pg_connection_string": normalize_pg_connection_string,
    })

    # 全タブで共有する単一コンテキスト
    ctx = get_context(sidebar_config)

    if not ctx.initialized:
        st.info("ℹ️ グラフ未初期化です。「🛠️ 構築/取り込み」タブで既存グラフを読み込むか、ドキュメントを構築してください。")

    tabs = st.tabs(
        ["💬 質問応答", "🕸️ グラフ探索", "📄 登録ドキュメント", "🛠️ 構築/取り込み", "⚙️ 設定"]
    )
    with tabs[0]:
        render_qa_tab(ctx)
    with tabs[1]:
        render_graph_tab(ctx)
    with tabs[2]:
        render_documents_tab(ctx)
    with tabs[3]:
        render_build_tab(ctx)
    with tabs[4]:
        render_settings_tab(ctx)

    # フッター
    st.markdown("---")
    st.markdown("**GraphRAG** | Powered by LangChain, Neo4j & PGVector")


if __name__ == "__main__":
    main()
