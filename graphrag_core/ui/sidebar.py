"""サイドバーUI（最小構成）

アプリ全体をゲートするハードフェイルガードと、読み取り専用の接続ステータスのみ
を描画する。対話的な設定ウィジェットは「⚙️ 設定」タブ
(graphrag_core/ui/settings_tab.py の render_settings_tab) に移動した。

構成（上から）:
  ⚙️ 設定 → 必須環境変数チェック(st.stop) → Neo4j接続ガード(st.stop)
  → 🩺 接続ステータス(折りたたみ) → 現在の検索設定サマリー(1行)

guard を st.tabs の前（= main() が render_sidebar を呼ぶ位置）に置くことで、
環境/接続不備のときに全タブの描画をブロックできる。タブ内に置くと st.tabs の
内側になり、他タブを止められないため guard は必ずここに残す。
"""

import streamlit as st

from graphrag_core.config import get_settings


def render_sidebar(settings: dict = None) -> dict:
    """サイドバーUI（ガード+ステータス）を描画し、最小の設定dictを返す。

    対話的な設定は render_settings_tab に移動済み。ここでは
    アプリをゲートするハードフェイルガードと読み取り専用ステータスのみ描画する。

    Parameters
    ----------
    settings : dict, optional
        実行時オブジェクト。以下のキーを参照:
        - get_llm_provider_info : callable  -> dict with 'status', 'provider', 'model'
        - neo4j_graph_class : type  (Neo4jGraph)

    Returns
    -------
    dict
        最小の config dict。show_graph / max_nodes は settings_tab が
        st.session_state に書き込み、build_ui_context が session_state から読む。
    """
    if settings is None:
        settings = {}

    s = get_settings()
    config = {}

    with st.sidebar:
        st.header("⚙️ 設定")

        # ---------- 必須環境変数チェック（ハードフェイルは常時表示） ----------
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

        # Neo4j 認証情報チェック（ハードフェイルは常時表示）
        Neo4jGraph = settings.get("neo4j_graph_class")
        if not all([s.neo4j_uri, s.neo4j_user, s.neo4j_pw]):
            st.error("❌ Neo4jを使用するには NEO4J_URI, NEO4J_USER, NEO4J_PW が必要です。")
            st.stop()

        # Neo4j 接続テスト（session_stateでキャッシュ、初回のみ実行）
        # 接続失敗はブロッキングなので expander の外で st.stop する。
        neo4j_ok = True
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
            neo4j_ok = st.session_state[neo4j_key]
            if not neo4j_ok:
                st.error(f"❌ Neo4j接続エラー: {st.session_state.get(neo4j_key + '_err', '')}")
                st.stop()

        # ---------- 接続ステータス（読み取り専用情報は折りたたみ） ----------
        with st.expander("🩺 接続ステータス", expanded=False):
            get_llm_provider_info = settings.get("get_llm_provider_info")
            if get_llm_provider_info:
                llm_info = get_llm_provider_info()
                st.info(f"🤖 {llm_info['status']}\n\nProvider: {llm_info['provider']}\nModel: {llm_info['model']}")

            # PDF前処理ステータス
            processor = (s.pdf_processor or "").lower()
            if processor == "onprem":
                backend = (s.pdf_backend or "vllm").lower()
                if backend == "paddleocr_remote" and s.paddlex_endpoint:
                    st.success(f"📄 onprem + PaddleX ({s.paddlex_endpoint})")
                elif backend == "vllm" and s.vllm_vision_endpoint:
                    st.success(f"📄 onprem + Vision vLLM ({s.vllm_vision_endpoint})")
                elif backend == "none":
                    st.info("📄 onprem (画像解析なし)")
                else:
                    st.warning(f"📄 onprem backend={backend} (エンドポイント未設定の可能性)")
            elif processor == "azure_di":
                if s.azure_di_endpoint:
                    st.success("📄 Azure Document Intelligence")
                else:
                    st.warning("📄 PDF_PROCESSOR=azure_di だが AZURE_DI_ENDPOINT 未設定")
            else:
                st.info("📄 PyMuPDF (プレーンテキスト抽出)")

            if Neo4jGraph:
                st.success("🗄️ Neo4j接続成功")

        # ---------- 現在の検索設定サマリー（1行・読み取り専用） ----------
        # 詳細な設定は「⚙️ 設定」タブで変更する。
        st.caption("詳細な設定は「⚙️ 設定」タブで変更できます。")
        mode_labels = {"hybrid": "ハイブリッド", "vector": "ベクトル", "keyword": "キーワード"}
        cur_mode = st.session_state.get("search_mode", "vector")
        cur_top_k = st.session_state.get("retrieval_top_k", s.retrieval_top_k)
        cur_kg = "ON" if st.session_state.get("enable_knowledge_graph", s.enable_knowledge_graph) else "OFF"
        cur_rr = "ON" if st.session_state.get("enable_rerank", s.enable_rerank) else "OFF"
        st.caption(
            f"🔎 {mode_labels.get(cur_mode, 'ベクトル')}検索 · "
            f"top_k={cur_top_k} · KG {cur_kg} · リランク {cur_rr}"
        )

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
