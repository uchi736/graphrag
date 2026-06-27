"""⚙️ 設定タブ（詳細 / 管理）

回答品質の主要レバー（Top-K・リランク・日本語ハイブリッド検索・ナレッジグラフ
利用ON/OFF）は「💬 質問応答」タブの🔎検索設定へ、グラフ可視化設定（表示ON/OFF・
最大ノード数）は「🕸️ グラフ探索」タブへ移設した（制御を効果の近くに置く）。
このタブには詳細設定と管理機能のみを残す。

構成（上から）:
  🔎 詳細設定（KGソースチャンク） → 🕸️ ナレッジグラフ詳細（ホップ数・
  エンティティベクトル検索） → 🗑️ データベース管理（取り扱い注意 / 折りたたみ）

重要: ウィジェットの key / label / value= の参照元(s.xxx)、および
st.session_state への書き込みは旧サイドバーと完全に一致させる。これにより
_build_config_from_session_state() とパイプラインの挙動が変わらない。

ハードフェイルガード（環境変数チェック / Neo4j接続チェックの st.stop）は
タブに移動しない。タブは st.tabs の内側で描画されるため、他タブをブロック
できない。ガードは render_sidebar（main() が st.tabs より前に呼ぶ）に残す。
"""

import streamlit as st

from graphrag_core.config import get_settings
from graphrag_core.db.utils import normalize_pg_connection_string
from graphrag_core.ui.sidebar import _clear_database

try:
    from langchain_neo4j import Neo4jGraph
except Exception:  # pragma: no cover - import guarded at app level
    Neo4jGraph = None


def render_settings_tab(ctx) -> None:
    """設定タブを描画する。

    旧サイドバーと同じウィジェット（同一の key / label / value= / session_state 書き込み）
    を再現する。返り値は無く、すべて st.session_state 経由で
    _build_config_from_session_state() に伝播する。
    """
    s = get_settings()

    st.header("⚙️ 設定（詳細 / 管理）")
    st.caption(
        "回答品質の主要レバー（Top-K・リランク・日本語検索・ナレッジグラフ利用）は"
        "「💬 質問応答」タブの🔎検索設定へ、グラフ可視化設定は「🕸️ グラフ探索」タブへ"
        "移設しました。ここには詳細設定と管理機能のみ残しています。"
    )

    # ---------- 検索の詳細設定 ----------
    st.markdown("### 🔎 詳細設定")

    # KGソースチャンク
    include_kg_chunks = st.checkbox(
        "KGソースチャンクを含める",
        value=True,
        help="グラフトリプルの出典チャンクをコンテキストに含めます"
    )
    st.session_state.include_kg_source_chunks = include_kg_chunks

    # ---------- ナレッジグラフ詳細（回答時のグラフ探索の挙動） ----------
    st.markdown("---")
    st.markdown("### 🕸️ ナレッジグラフ詳細")

    # ナレッジグラフ利用ON/OFFは「💬 質問応答」タブの🔎検索設定へ移設。
    # ここでは session_state を READ して、ON時のみ詳細サブ設定を表示する。
    if st.session_state.get("enable_knowledge_graph", s.enable_knowledge_graph):
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

        # エンティティベクトル検索設定
        enable_entity_vector = st.checkbox(
            "エンティティベクトル検索",
            value=s.enable_entity_vector_search,
            help="エンティティの類似度検索を有効化。類義語や関連語も検索可能になります。"
        )
        st.session_state.enable_entity_vector = enable_entity_vector

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
    else:
        st.caption("ナレッジグラフは「💬 質問応答」タブの🔎検索設定でONにしてください")

    # ---------- データベース管理（取り扱い注意 / 折りたたみ） ----------
    st.markdown("---")
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = False

    # 確認待ち中は expander を開いたままにする（折りたたむと確認ボタンが隠れる）
    with st.expander("🗑️ データベース管理（取り扱い注意）", expanded=st.session_state.confirm_delete):
        st.caption("Neo4j と PGVector コレクションを完全に削除します（取り消し不可）。")
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
                                Neo4jGraph,
                                normalize_pg_connection_string,
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
