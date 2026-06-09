"""サイドバーUI

Streamlitサイドバーの設定UIを描画し、設定値をdictとして返す。
"""

import os
import streamlit as st

from graphrag_core.config import get_settings


def render_sidebar(settings: dict = None) -> dict:
    """サイドバーUIを描画し、設定dictを返す。

    Parameters
    ----------
    settings : dict, optional
        実行時オブジェクト。以下のキーを参照:
        - max_nodes : int
        - get_llm_provider_info : callable  -> dict with 'status', 'provider', 'model'
        - neo4j_graph_class : type  (Neo4jGraph)
        - sudachi_available : bool
        - normalize_pg_connection_string : callable

    Returns
    -------
    dict
        qa_pipeline互換のconfig dict。以下のキーを含む:
        - graph_hop_count
        - retrieval_top_k, enable_japanese_search, enable_rerank
        - enable_entity_vector, entity_similarity_threshold
        - search_mode, include_kg_source_chunks
        - enable_knowledge_graph, show_graph, max_nodes
    """
    if settings is None:
        settings = {}

    s = get_settings()
    config = {}

    with st.sidebar:
        st.header("⚙️ 設定")

        # ---------- 必須環境変数チェック ----------
        # LLM/Embedding プロバイダごとに必須項目を切替（完全オンプレ運用時は
        # Azure系の環境変数は不要）
        required_envs = {"PG_CONN": s.pg_conn}

        llm_provider = s.llm_provider.lower()
        if llm_provider == "azure_openai":
            required_envs["AZURE_OPENAI_API_KEY"] = s.azure_openai_api_key
            required_envs["AZURE_OPENAI_ENDPOINT"] = s.azure_openai_endpoint
            required_envs["AZURE_OPENAI_API_VERSION"] = s.azure_openai_api_version
            required_envs["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = s.azure_openai_chat_deployment
        elif llm_provider == "vllm":
            required_envs["VLLM_ENDPOINT"] = s.vllm_endpoint

        emb_provider = s.embedding_provider.lower()
        if emb_provider == "azure_openai":
            required_envs["AZURE_OPENAI_API_KEY"] = s.azure_openai_api_key
            required_envs["AZURE_OPENAI_ENDPOINT"] = s.azure_openai_endpoint
            required_envs["AZURE_OPENAI_API_VERSION"] = s.azure_openai_api_version
            required_envs["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"] = s.azure_openai_embedding_deployment
        elif emb_provider == "vllm":
            required_envs["VLLM_EMBEDDING_ENDPOINT"] = s.vllm_embedding_endpoint

        missing_envs = [name for name, value in required_envs.items() if not value]
        if missing_envs:
            st.error(f"環境変数が不足しています: {', '.join(missing_envs)}")
            st.stop()

        # ---------- LLM Provider Status ----------
        st.markdown("---")
        st.markdown("### 🤖 LLM Provider")
        get_llm_provider_info = settings.get("get_llm_provider_info")
        if get_llm_provider_info:
            llm_info = get_llm_provider_info()
            st.info(f"{llm_info['status']}\n\nProvider: {llm_info['provider']}\nModel: {llm_info['model']}")

        # ---------- PDF前処理ステータス ----------
        st.markdown("---")
        st.markdown("### 📄 PDF解析")
        processor = (s.pdf_processor or "").lower()
        if processor == "onprem":
            backend = (s.pdf_backend or "vllm").lower()
            if backend == "paddleocr_remote" and s.paddlex_endpoint:
                st.success(f"✅ onprem + PaddleX ({s.paddlex_endpoint})")
            elif backend == "vllm" and s.vllm_vision_endpoint:
                st.success(f"✅ onprem + Vision vLLM ({s.vllm_vision_endpoint})")
            elif backend == "none":
                st.info("ℹ️ onprem (画像解析なし)")
            else:
                st.warning(f"⚠️ onprem backend={backend} (エンドポイント未設定の可能性)")
        elif processor == "azure_di":
            if s.azure_di_endpoint:
                st.success("✅ Azure Document Intelligence")
            else:
                st.warning("⚠️ PDF_PROCESSOR=azure_di だが AZURE_DI_ENDPOINT 未設定")
        else:
            st.info("ℹ️ PyMuPDF (プレーンテキスト抽出)")

        # ---------- グラフバックエンド (Neo4j) ----------
        st.markdown("---")
        st.markdown("### 🗄️ グラフバックエンド")

        Neo4jGraph = settings.get("neo4j_graph_class")
        if not all([s.neo4j_uri, s.neo4j_user, s.neo4j_pw]):
            st.error("❌ Neo4jを使用するには NEO4J_URI, NEO4J_USER, NEO4J_PW が必要です。")
            st.stop()

        # Neo4j接続テスト（session_stateでキャッシュ、初回のみ実行）
        if Neo4jGraph:
            neo4j_key = f"neo4j_conn_ok::{s.neo4j_uri}::{s.neo4j_user}"
            if st.session_state.get(neo4j_key) is None:
                try:
                    with st.spinner("Neo4j接続確認中..."):
                        test_graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw)
                        del test_graph
                    st.session_state[neo4j_key] = True
                except Exception as e:
                    st.session_state[neo4j_key] = False
                    st.session_state[neo4j_key + "_err"] = str(e)[:100]
            if st.session_state[neo4j_key]:
                st.success("✅ Neo4j接続成功")
            else:
                st.error(f"❌ Neo4j接続エラー: {st.session_state.get(neo4j_key + '_err', '')}")
                st.stop()

        # ---------- グラフ可視化設定 ----------
        st.markdown("---")
        st.markdown("### 📊 グラフ可視化設定")

        show_graph = st.checkbox("ナレッジグラフを表示", value=True)
        config["show_graph"] = show_graph

        max_nodes_default = st.session_state.get("max_nodes", settings.get("max_nodes", 200))
        if show_graph:
            max_nodes = st.slider("最大表示ノード数", 50, 100000, max_nodes_default, 50)
            if max_nodes != st.session_state.get("max_nodes", max_nodes_default):
                st.session_state.max_nodes = max_nodes
                st.session_state.graph_data_cache = None
                if 'all_node_list' in st.session_state:
                    del st.session_state.all_node_list
        else:
            max_nodes = max_nodes_default
        config["max_nodes"] = max_nodes

        # ---------- 検索設定 ----------
        st.markdown("---")
        st.markdown("### 🔍 検索設定")

        retrieval_top_k = st.slider(
            "検索結果数 (Top-K)",
            min_value=1,
            max_value=20,
            value=s.retrieval_top_k,
            step=1,
            help="RAG検索で取得するチャンク数。多いほど文脈が豊富になりますが、処理時間が増加します。"
        )
        st.session_state.retrieval_top_k = retrieval_top_k
        config["retrieval_top_k"] = retrieval_top_k

        # ---------- ナレッジグラフ機能設定 ----------
        st.markdown("---")
        st.markdown("### 🕸️ ナレッジグラフ")

        enable_knowledge_graph = st.checkbox(
            "ナレッジグラフ生成を有効化",
            value=s.enable_knowledge_graph,
            help="テキストからエンティティと関係性を抽出してグラフ構造を生成します。処理時間が増加しますが、より高度な質問応答が可能になります。"
        )
        st.session_state.enable_knowledge_graph = enable_knowledge_graph
        config["enable_knowledge_graph"] = enable_knowledge_graph

        if enable_knowledge_graph:
            st.info("🔍 ナレッジグラフ: 有効\nエンティティと関係性を抽出し、グラフベースの推論を行います")

            # グラフ探索ホップ数設定
            graph_hop_count = st.slider(
                "グラフ探索ホップ数",
                min_value=1,
                max_value=3,
                value=s.graph_hop_count,
                step=1,
                help="1hop=直接関係のみ、2hop=友達の友達まで、3hop=さらに間接的な関係まで探索"
            )
            st.session_state.graph_hop_count = graph_hop_count
            config["graph_hop_count"] = graph_hop_count

            # エンティティベクトル検索設定
            enable_entity_vector = st.checkbox(
                "エンティティベクトル検索",
                value=s.enable_entity_vector_search,
                help="エンティティの類似度検索を有効化。類義語や関連語も検索可能になります。"
            )
            st.session_state.enable_entity_vector = enable_entity_vector
            config["enable_entity_vector"] = enable_entity_vector

            if enable_entity_vector:
                entity_similarity_threshold = st.slider(
                    "エンティティ類似度閾値",
                    min_value=0.5,
                    max_value=1.0,
                    value=s.entity_similarity_threshold,
                    step=0.05,
                    help="エンティティ検索の類似度閾値。低いほど幅広く検索します。"
                )
                st.session_state.entity_similarity_threshold = entity_similarity_threshold
                config["entity_similarity_threshold"] = entity_similarity_threshold
            else:
                config["entity_similarity_threshold"] = s.entity_similarity_threshold
        else:
            st.warning("⚡ ナレッジグラフ: 無効\nベクトル検索のみ使用（高速モード）")
            config["graph_hop_count"] = s.graph_hop_count
            config["enable_entity_vector"] = False
            config["entity_similarity_threshold"] = s.entity_similarity_threshold

        # ---------- 日本語ハイブリッド検索設定 ----------
        SUDACHI_AVAILABLE = settings.get("sudachi_available", False)
        if SUDACHI_AVAILABLE:
            enable_jp_search = st.checkbox(
                "日本語ハイブリッド検索",
                value=s.enable_japanese_search,
                help="ベクトル検索とキーワード検索を組み合わせます（精度向上）"
            )

            if enable_jp_search:
                search_mode = st.radio(
                    "検索モード",
                    ["ハイブリッド (推奨)", "ベクトルのみ", "キーワードのみ"],
                    help="ハイブリッド: RRFでスコア統合 / ベクトル: 意味検索 / キーワード: 全文検索"
                )

                mode_map = {
                    "ハイブリッド (推奨)": "hybrid",
                    "ベクトルのみ": "vector",
                    "キーワードのみ": "keyword"
                }
                st.session_state.search_mode = mode_map[search_mode]
                st.session_state.enable_japanese_search = True
                config["search_mode"] = mode_map[search_mode]
                config["enable_japanese_search"] = True
            else:
                st.session_state.search_mode = "vector"
                st.session_state.enable_japanese_search = False
                config["search_mode"] = "vector"
                config["enable_japanese_search"] = False
        else:
            st.warning("⚠️ sudachipy未インストール")
            st.caption("ベクトル検索のみ使用します")
            with st.expander("インストール方法"):
                st.code("pip install sudachipy sudachidict_core")
            st.session_state.search_mode = "vector"
            st.session_state.enable_japanese_search = False
            config["search_mode"] = "vector"
            config["enable_japanese_search"] = False

        # ---------- KGソースチャンク設定 ----------
        include_kg_chunks = st.checkbox(
            "KGソースチャンクを含める",
            value=True,
            help="グラフトリプルの出典チャンクをコンテキストに含めます"
        )
        st.session_state.include_kg_source_chunks = include_kg_chunks
        config["include_kg_source_chunks"] = include_kg_chunks

        # ---------- LLMリランキング設定 ----------
        enable_rerank = st.checkbox(
            "LLMリランキング",
            value=False,
            help="LLMで検索結果を再ランキング（精度向上、速度低下）"
        )
        st.session_state.enable_rerank = enable_rerank
        config["enable_rerank"] = enable_rerank

        # ---------- データベース管理 ----------
        st.markdown("---")
        st.markdown("### 🗑️ データベース管理")

        if "confirm_delete" not in st.session_state:
            st.session_state.confirm_delete = False

        if not st.session_state.confirm_delete:
            if st.button("🗑️ データベースをクリア", width="stretch"):
                st.session_state.confirm_delete = True
                st.rerun()
        else:
            st.warning("⚠️ 本当にすべてのデータを削除しますか？")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ はい、削除", type="primary", width="stretch"):
                    with st.spinner("データベースをクリア中..."):
                        try:
                            _clear_database(
                                settings.get("neo4j_graph_class"),
                                settings.get("normalize_pg_connection_string"),
                            )
                            # セッションステートリセット
                            st.session_state.chain = None
                            st.session_state.graph = None
                            st.session_state.initialized = False
                            st.session_state.uploaded_files = []
                            st.session_state.existing_graph_loaded = False
                            st.session_state.graph_data_cache = None
                            st.session_state.confirm_delete = False

                            st.success("✅ データベースをクリアしました")
                            st.rerun()
                        except Exception as e:
                            st.error(f"クリアエラー: {e}")
                            st.session_state.confirm_delete = False
            with col2:
                if st.button("❌ キャンセル", width="stretch"):
                    st.session_state.confirm_delete = False
                    st.rerun()

        config["path_max_candidates"] = s.path_max_candidates

    return config


def _clear_database(Neo4jGraph=None, normalize_pg_connection_string=None):
    """データベースクリア処理（内部関数）"""
    s = get_settings()

    # Neo4jクリア
    if Neo4jGraph:
        temp_graph = Neo4jGraph(
            url=s.neo4j_uri,
            username=s.neo4j_user,
            password=s.neo4j_pw,
            enhanced_schema=True
        )
        temp_graph.query("MATCH (n) DETACH DELETE n")

    # PGVectorクリア（コレクション単位で削除）
    if s.pg_conn and normalize_pg_connection_string:
        try:
            import psycopg
            raw_conn = normalize_pg_connection_string(s.pg_conn)
            with psycopg.connect(raw_conn) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM langchain_pg_embedding e
                        USING langchain_pg_collection c
                        WHERE e.collection_id = c.uuid AND c.name = %s
                    """, (s.pg_collection,))
                    cur.execute("""
                        DELETE FROM langchain_pg_collection WHERE name = %s
                    """, (s.pg_collection,))
                conn.commit()
        except Exception as e:
            st.warning(f"PGVectorクリアで警告: {e}")
