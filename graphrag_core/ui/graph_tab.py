"""
graphrag_core/ui/graph_tab.py
『🕸️ グラフ探索』タブ。render_graph_tab(ctx) + グラフ取得/Cypher ヘルパー。
4モード(全体可視化/ノード中心/データテーブル/Cypher検索)をフラットradioで切替。
"""
from typing import List

import streamlit as st

from graphrag_core.ui.visualization import visualize_graph_neo4j_viz
from graphrag_core.ui.data_tables import display_data_tables as _display_data_tables
from graphrag_core.ui.feedback import show_error
from graphrag_core.llm.factory import create_chat_llm
from graphrag_core.prompts import NL_TO_CYPHER_PROMPT
from graphrag_core.llm.langfuse_utils import observe, get_langfuse_callback


# =====================================================================
# グラフデータ取得関数
# =====================================================================
# データ取得/Cypher の実体は services/graph_explore.py へ移設（st非依存化）。
# ここは st.error 等の表示挙動を維持する薄いラッパのみ。
from graphrag_core.services.graph_explore import (  # noqa: E402,F401
    WriteQueryRejected,
    get_enhanced_graph_data,
    get_enhanced_subgraph_data,
)
from graphrag_core.services.graph_explore import (  # noqa: E402
    natural_language_to_cypher as _svc_nl_to_cypher,
    execute_readonly_cypher as _svc_execute_cypher,
)


def natural_language_to_cypher(query: str) -> str:
    """自然言語クエリをCypherクエリに変換（表示付きラッパ）"""
    try:
        return _svc_nl_to_cypher(query)
    except Exception as e:
        st.error(f"Cypherクエリ変換エラー: {e}")
        return ""


def execute_cypher_and_visualize(cypher_query: str, graph):
    """Cypherクエリ（参照のみ）を実行して結果を返す（表示付きラッパ）"""
    try:
        result = _svc_execute_cypher(graph, cypher_query)
        if result["applied_limit"]:
            st.info("結果上限として LIMIT 500 を付与しました。")
        if not result["rows"]:
            st.warning("クエリ結果が空です")
            return None
        return result["rows"]
    except WriteQueryRejected as e:
        st.error(str(e))
        return None
    except Exception as e:
        show_error("クエリ実行エラー", e)
        return None


# =====================================================================
# タブ2: グラフ探索
# =====================================================================
def render_graph_tab(ctx):
    st.header("🕸️ グラフ探索")

    if ctx.initialized:
        # 📊 グラフ可視化設定（旧⚙️設定タブから移設: 制御を効果の近くに置く）。
        # モードに依らず常に表示する（mode radio より前に描画）。
        with st.expander("📊 表示設定", expanded=False):
            show_graph = st.checkbox("ナレッジグラフを表示", value=st.session_state.get("show_graph", True))
            st.session_state.show_graph = show_graph

            max_nodes_default = st.session_state.get("max_nodes", 200)
            if show_graph:
                # 描画可能な範囲に上限を制限（旧: 100000 はブラウザ描画が破綻する）
                slider_value = min(max_nodes_default, 2000)
                max_nodes = st.slider(
                    "最大表示ノード数", 50, 2000, slider_value, 50,
                    help="可視化の最大ノード数。~1000を超えると描画が重くなることがあります。"
                )
                if max_nodes != st.session_state.get("max_nodes", max_nodes_default):
                    had_cache = st.session_state.get("graph_data_cache") is not None
                    st.session_state.max_nodes = max_nodes
                    st.session_state.graph_data_cache = None
                    if 'all_node_list' in st.session_state:
                        del st.session_state.all_node_list
                    if had_cache:
                        st.toast("表示ノード数を変更しました。グラフを再読み込みしてください", icon="🔄")
            else:
                max_nodes = max_nodes_default
            st.session_state.max_nodes = max_nodes

        # 旧: 表示モード(3) + 可視化範囲(2) のネストradio → 単一の4択にフラット化
        mode = st.radio(
            "操作",
            ["🕸️ 全体可視化", "🎯 ノード中心", "📊 データテーブル", "🔍 Cypher検索"],
            horizontal=True,
            key="graph_mode",
        )

        st.markdown("---")

        # モード1: グラフ可視化（全体 / ノード中心）
        if mode in ("🕸️ 全体可視化", "🎯 ノード中心"):
            if not ctx.show_graph:
                st.warning("上の「📊 表示設定」で「ナレッジグラフを表示」をONにしてください")
            else:
                viz_scope = "部分表示（検索）" if mode == "🎯 ノード中心" else "全体表示"

                if viz_scope == "部分表示（検索）":
                    st.markdown("### 🔍 ノード検索")

                    if 'center_nodes' not in st.session_state:
                        st.session_state.center_nodes = []

                    if 'all_node_list' not in st.session_state:
                        if st.session_state.graph_data_cache:
                            graph_data = st.session_state.graph_data_cache
                            all_nodes = list(set(
                                [item['source'] for item in graph_data] +
                                [item['target'] for item in graph_data]
                            ))
                            st.session_state.all_node_list = sorted(all_nodes)
                        else:
                            with st.spinner("ノードリスト取得中..."):
                                try:
                                    graph_data = get_enhanced_graph_data(ctx.graph, limit=ctx.max_nodes)
                                    st.session_state.graph_data_cache = graph_data
                                    all_nodes = list(set(
                                        [item['source'] for item in graph_data] +
                                        [item['target'] for item in graph_data]
                                    ))
                                    st.session_state.all_node_list = sorted(all_nodes)
                                except Exception as e:
                                    st.error(f"エラー: {e}")
                                    st.session_state.all_node_list = []

                    if st.session_state.all_node_list:
                        search_query = st.text_input(
                            "🔍 ノード検索（部分一致）",
                            placeholder="例: API",
                            help="検索したノードとその周辺を表示します"
                        )

                        if search_query:
                            matched_nodes = [n for n in st.session_state.all_node_list
                                            if search_query.lower() in n.lower()]
                            st.caption(f"🔍 検索結果: {len(matched_nodes)}件")

                            if matched_nodes:
                                selected_node = st.selectbox(
                                    "ノードを選択",
                                    options=[""] + matched_nodes,
                                    index=0,
                                    key="node_select_box",
                                    help="リストから1つ選んで追加してください"
                                )

                                def _add_center_node():
                                    node = st.session_state.node_select_box
                                    if node and node not in st.session_state.center_nodes:
                                        st.session_state.center_nodes.append(node)

                                def _reset_center_nodes():
                                    st.session_state.center_nodes = []

                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    st.button("➕ 中心ノードに追加", on_click=_add_center_node,
                                              disabled=not selected_node)
                                with col2:
                                    if st.session_state.center_nodes:
                                        st.button("🗑️ リセット", on_click=_reset_center_nodes)
                            else:
                                st.warning(f"「{search_query}」に一致するノードが見つかりませんでした")
                        else:
                            st.info("💡 ノード名を入力して検索してください")

                        if st.session_state.center_nodes:
                            st.markdown("---")
                            st.write("**中心ノード:**", ", ".join(st.session_state.center_nodes))

                            hop_distance = st.slider(
                                "周辺表示範囲（Hop数）",
                                min_value=1,
                                max_value=3,
                                value=2,
                                help="選択ノードから何Hop先まで表示するか"
                            )

                            if st.button("📊 サブグラフを表示", type="primary"):
                                with st.spinner("サブグラフ取得中..."):
                                    try:
                                        subgraph_data = get_enhanced_subgraph_data(
                                            ctx.graph,
                                            st.session_state.center_nodes,
                                            hop_distance,
                                            limit=500
                                        )

                                        if subgraph_data:
                                            unique_nodes = set()
                                            for item in subgraph_data:
                                                unique_nodes.add(item['source'])
                                                unique_nodes.add(item['target'])

                                            st.success(f"✅ サブグラフ取得完了")
                                            st.info(f"📊 表示: ノード {len(unique_nodes)}個 / エッジ {len(subgraph_data)}本")

                                            html = visualize_graph_neo4j_viz(subgraph_data)
                                            if html:
                                                st.components.v1.html(html, height=700)
                                            else:
                                                st.warning("可視化に失敗しました。")
                                        else:
                                            st.warning("選択したノードのサブグラフが見つかりませんでした")
                                    except Exception as e:
                                        show_error("サブグラフ取得エラー", e)
                        else:
                            st.info("👆 検索してノードを追加してください")
                    else:
                        st.warning("ノードリストが取得できませんでした。グラフが空か、接続に問題がある可能性があります。")

                else:
                    # 全体表示モード
                    if st.session_state.graph_data_cache is None:
                        if st.button("📊 グラフを読み込む", type="primary"):
                            with st.spinner("グラフデータ取得中..."):
                                try:
                                    graph_data = get_enhanced_graph_data(ctx.graph, limit=ctx.max_nodes)
                                    st.session_state.graph_data_cache = graph_data
                                    st.success(f"✅ {len(graph_data)}件のエッジを読み込みました")
                                except Exception as e:
                                    st.error(f"エラー: {e}")

                    if st.session_state.graph_data_cache:
                        try:
                            graph_data = st.session_state.graph_data_cache

                            if not graph_data:
                                st.warning("グラフデータがありません")
                            else:
                                unique_nodes = set()
                                for item in graph_data:
                                    unique_nodes.add(item['source'])
                                    unique_nodes.add(item['target'])

                                st.info(f"📊 表示中: ノード {len(unique_nodes)}個 / エッジ {len(graph_data)}本")

                                html = visualize_graph_neo4j_viz(graph_data)
                                if html:
                                    st.components.v1.html(html, height=700)
                                else:
                                    st.warning("可視化に失敗しました。")

                            def _reload_graph():
                                st.session_state.graph_data_cache = None
                                if 'all_node_list' in st.session_state:
                                    del st.session_state['all_node_list']

                            st.button("🔄 グラフを再読み込み", on_click=_reload_graph)

                        except Exception as e:
                            show_error("グラフ表示エラー", e)

        # モード2: データテーブル
        elif mode == "📊 データテーブル":
            # キャッシュは可視化と共有のため、現在のスコープと再取得手段を明示する
            if st.session_state.graph_data_cache is not None:
                _gd = st.session_state.graph_data_cache
                st.caption(f"読込済み: {len(_gd)}件のエッジ（上限 {ctx.max_nodes}・可視化と共有）")
                if st.button("🔄 再取得", key="reload_data_table"):
                    st.session_state.graph_data_cache = None
                    if 'all_node_list' in st.session_state:
                        del st.session_state['all_node_list']
                    st.rerun()
            else:
                if st.button("📊 データを読み込む", type="primary", key="load_data_table"):
                    with st.spinner("データ取得中..."):
                        try:
                            graph_data = get_enhanced_graph_data(ctx.graph, limit=ctx.max_nodes)
                            st.session_state.graph_data_cache = graph_data
                            st.success(f"✅ {len(graph_data)}件のエッジを読み込みました")
                        except Exception as e:
                            show_error("データ取得エラー", e)

            if st.session_state.graph_data_cache:
                try:
                    graph_data = st.session_state.graph_data_cache
                    if graph_data:
                        _display_data_tables(
                            graph_data,
                            graph=ctx.graph,
                            enable_edit=True,
                            **ctx.crud,
                            edit_node_dialog_fn=ctx.edit_node_dialog,
                            edit_edge_dialog_fn=ctx.edit_edge_dialog,
                        )
                    else:
                        st.warning("グラフデータがありません")
                except Exception as e:
                    show_error("データテーブル表示エラー", e)

        # モード3: Cypherクエリ検索
        elif mode == "🔍 Cypher検索":
            st.markdown("### 自然言語でグラフを検索")
            st.info("例: 「APIに関するグラフを見たい」「認証と関係のあるエンティティを表示」")

            with st.expander("📋 クエリテンプレート"):
                template = st.selectbox(
                    "よく使うクエリ",
                    [
                        "カスタム（自分で入力）",
                        "特定エンティティに関連するすべての関係を表示",
                        "最も接続数が多いノードTop10を表示",
                        "すべてのリレーションシップタイプを表示"
                    ]
                )

                if template == "特定エンティティに関連するすべての関係を表示":
                    entity_name = st.text_input("エンティティ名を入力:", placeholder="例: API")
                    nl_query = f"{entity_name}に関連するすべての関係を表示" if entity_name else ""
                elif template == "最も接続数が多いノードTop10を表示":
                    nl_query = "最も接続数が多いノードTop10を表示"
                elif template == "すべてのリレーションシップタイプを表示":
                    nl_query = "すべてのリレーションシップタイプとその数を表示"
                else:
                    nl_query = ""

            user_query = st.text_area(
                "自然言語クエリ:",
                value=nl_query,
                height=100,
                placeholder="例: APIに関するグラフを見たい"
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                convert_button = st.button("🔄 Cypherに変換", type="primary")

            if "generated_cypher" not in st.session_state:
                st.session_state.generated_cypher = ""

            if convert_button and user_query:
                with st.spinner("Cypherクエリを生成中..."):
                    cypher_query = natural_language_to_cypher(user_query)
                    st.session_state.generated_cypher = cypher_query

            if st.session_state.generated_cypher:
                st.markdown("### 📝 生成されたCypherクエリ")
                edited_cypher = st.text_area(
                    "Cypherクエリ（編集可能）:",
                    value=st.session_state.generated_cypher,
                    height=150,
                    key="cypher_editor"
                )

                def _clear_cypher():
                    st.session_state.generated_cypher = ""

                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    execute_button = st.button("▶️ 実行", type="primary")
                with col2:
                    st.button("🗑️ クリア", on_click=_clear_cypher)

                if execute_button and edited_cypher:
                    with st.spinner("クエリ実行中..."):
                        result = execute_cypher_and_visualize(edited_cypher, ctx.graph)

                        if result:
                            st.success(f"✅ {len(result)}件の結果を取得しました")

                            st.markdown("### 📊 クエリ結果")
                            import pandas as pd
                            df = pd.DataFrame(result)
                            for col in df.columns:
                                if df[col].apply(type).nunique() > 1:
                                    df[col] = df[col].astype(str)
                            st.dataframe(df, use_container_width=True)

                            if len(result) > 0 and 'source' in result[0] and 'target' in result[0] and 'relation' in result[0]:
                                st.markdown("### 🕸️ グラフ可視化")
                                html = visualize_graph_neo4j_viz(result)
                                if html:
                                    st.components.v1.html(html, height=700)

    else:
        st.info("「🛠️ 構築/取り込み」タブでドキュメントを構築するか、既存グラフを読み込んでください")
