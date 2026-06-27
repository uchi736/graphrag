"""データテーブル表示関数

ノードとエッジをStreamlit上でテーブル形式で表示する（編集機能付き）。
"""

import json
import streamlit as st
import pandas as pd

from graphrag_core.ui.visualization import get_node_type, get_color_for_type


def display_data_tables(graph_data, graph=None, enable_edit=False,
                        graph_add_node=None, graph_update_node=None,
                        graph_delete_node=None, graph_get_node_info=None,
                        graph_add_edge=None, graph_update_edge=None,
                        graph_delete_edge=None, graph_get_edge_info=None,
                        graph_get_data_for_cache=None,
                        edit_node_dialog_fn=None, edit_edge_dialog_fn=None):
    """ノードとエッジをテーブル形式で表示（編集機能付き）

    Parameters
    ----------
    graph_data : list[dict]
        グラフデータ（source, target, relation等のキーを持つdict のリスト）
    graph : object, optional
        グラフバックエンドインスタンス
    enable_edit : bool
        編集機能を有効にするか
    graph_add_node, graph_update_node, graph_delete_node, graph_get_node_info :
        ノードCRUD用のコールバック関数群（バックエンド共通インターフェース）
    graph_add_edge, graph_update_edge, graph_delete_edge, graph_get_edge_info :
        エッジCRUD用のコールバック関数群
    graph_get_data_for_cache :
        キャッシュ用データ取得コールバック
    edit_node_dialog_fn, edit_edge_dialog_fn :
        ダイアログ表示用コールバック関数
    """

    # ノードデータの集計
    nodes_dict = {}
    for item in graph_data:
        # ソースノード
        if item['source'] not in nodes_dict:
            source_type = get_node_type(item['source'], item.get('source_type'))
            nodes_dict[item['source']] = {
                'ノードID': item['source'],
                'タイプ': source_type,
                '接続数': item.get('source_degree', 0),
                '色': get_color_for_type(source_type)
            }

        # ターゲットノード
        if item['target'] not in nodes_dict:
            target_type = get_node_type(item['target'], item.get('target_type'))
            nodes_dict[item['target']] = {
                'ノードID': item['target'],
                'タイプ': target_type,
                '接続数': item.get('target_degree', 0),
                '色': get_color_for_type(target_type)
            }

    # エッジデータの作成
    edges_list = []
    for item in graph_data:
        edges_list.append({
            '始点': item['source'],
            'リレーション': item['relation'],
            '終点': item['target'],
            'edge_key': item.get('edge_key', 0)
        })

    # ノードテーブル
    st.subheader("📍 ノード一覧")

    # 編集機能が有効な場合は編集ボタンを追加
    if enable_edit and graph:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("➕ 新規ノード追加", key="add_node_btn"):
                st.session_state.edit_mode = "add_node"

        # 編集モードの処理
        if st.session_state.get('edit_mode') == 'add_node':
            with st.expander("➕ 新規ノード追加", expanded=True):
                if edit_node_dialog_fn:
                    edit_node_dialog_fn(graph, None)
                if st.button("閉じる"):
                    st.session_state.edit_mode = None
                    st.rerun()

    nodes_df = pd.DataFrame(list(nodes_dict.values()))
    st.dataframe(
        nodes_df.sort_values('接続数', ascending=False),
        width='stretch',
        hide_index=True
    )

    # 編集機能: ノード個別編集・削除
    if enable_edit and graph:
        st.caption("ノードを編集・削除する場合は以下から選択してください")
        selected_node = st.selectbox(
            "ノードを選択",
            options=[""] + list(nodes_dict.keys()),
            key="selected_node"
        )

        if selected_node:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✏️ 編集", key=f"edit_node_{selected_node}"):
                    st.session_state.editing_node = selected_node

            # 編集モード時は常にダイアログ表示
            if st.session_state.get('editing_node') == selected_node:
                node_info = graph_get_node_info(graph, selected_node) if graph_get_node_info else None
                if node_info:
                    with st.expander(f"✏️ ノード編集: {selected_node}", expanded=True):
                        if edit_node_dialog_fn:
                            edit_node_dialog_fn(graph, node_info)
            confirm_node_key = f'confirm_delete_node_{selected_node}'
            with col2:
                if st.button("🗑️ 削除", key=f"delete_node_{selected_node}"):
                    st.session_state[confirm_node_key] = True

            # 明示的な はい/キャンセル 確認（旧: 「もう一度押す」トグルは取り消し不可で危険）
            if st.session_state.get(confirm_node_key):
                st.warning(f"⚠️ ノード '{selected_node}' を削除します。この操作は取り消せません。")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("はい、削除", type="primary", key=f"confirm_delete_node_yes_{selected_node}", width="stretch"):
                        success = graph_delete_node(graph, selected_node) if graph_delete_node else False
                        st.session_state[confirm_node_key] = False
                        if success:
                            st.toast(f"ノード '{selected_node}' を削除しました", icon="🗑️")
                            if graph_get_data_for_cache:
                                st.session_state.graph_data_cache = graph_get_data_for_cache(graph)
                            st.rerun()
                        else:
                            st.error("削除に失敗しました")
                with c2:
                    if st.button("キャンセル", key=f"confirm_delete_node_no_{selected_node}", width="stretch"):
                        st.session_state[confirm_node_key] = False
                        st.rerun()

    # CSVダウンロード
    csv_nodes = nodes_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 ノードをCSVでダウンロード",
        data=csv_nodes,
        file_name="nodes.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # エッジテーブル
    st.subheader("🔗 エッジ一覧")

    # 編集機能が有効な場合は追加ボタンを表示
    if enable_edit and graph:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("➕ 新規エッジ追加", key="add_edge_btn"):
                st.session_state.edit_mode = "add_edge"

        # 編集モードの処理
        if st.session_state.get('edit_mode') == 'add_edge':
            all_node_ids = list(nodes_dict.keys())
            with st.expander("➕ 新規エッジ追加", expanded=True):
                if edit_edge_dialog_fn:
                    edit_edge_dialog_fn(graph, None, all_node_ids)
                if st.button("閉じる", key="close_add_edge"):
                    st.session_state.edit_mode = None
                    st.rerun()

    edges_df = pd.DataFrame(edges_list)
    st.dataframe(
        edges_df,
        width='stretch',
        hide_index=True
    )

    # 編集機能: エッジ個別編集・削除
    if enable_edit and graph:
        st.caption("エッジを編集・削除する場合は以下から選択してください")

        # エッジ選択肢を作成
        edge_options = [""] + [f"{e['始点']} → {e['終点']} ({e['リレーション']})" for e in edges_list]
        selected_edge_str = st.selectbox(
            "エッジを選択",
            options=edge_options,
            key="selected_edge"
        )

        if selected_edge_str:
            # 選択されたエッジを解析
            selected_idx = edge_options.index(selected_edge_str) - 1
            if selected_idx >= 0:
                selected_edge_data = edges_list[selected_idx]
                source = selected_edge_data['始点']
                target = selected_edge_data['終点']
                rel_type = selected_edge_data['リレーション']
                edge_key = selected_edge_data.get('edge_key', 0)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✏️ 編集", key=f"edit_edge_{selected_idx}"):
                        st.session_state.editing_edge = selected_idx

                # 編集モード時は常にダイアログ表示
                if st.session_state.get('editing_edge') == selected_idx:
                    edge_info = graph_get_edge_info(graph, source, target, edge_key) if graph_get_edge_info else None
                    if edge_info:
                        with st.expander(f"✏️ エッジ編集: {source} → {target}", expanded=True):
                            if edit_edge_dialog_fn:
                                edit_edge_dialog_fn(graph, edge_info)
                confirm_edge_key = f'confirm_delete_edge_{selected_idx}'
                with col2:
                    if st.button("🗑️ 削除", key=f"delete_edge_{selected_idx}"):
                        st.session_state[confirm_edge_key] = True

                # 明示的な はい/キャンセル 確認
                if st.session_state.get(confirm_edge_key):
                    st.warning(f"⚠️ エッジ '{source} → {target}' を削除します。この操作は取り消せません。")
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        if st.button("はい、削除", type="primary", key=f"confirm_delete_edge_yes_{selected_idx}", width="stretch"):
                            success = graph_delete_edge(graph, source, target, edge_key) if graph_delete_edge else False
                            st.session_state[confirm_edge_key] = False
                            if success:
                                st.toast(f"エッジ '{source} → {target}' を削除しました", icon="🗑️")
                                if graph_get_data_for_cache:
                                    st.session_state.graph_data_cache = graph_get_data_for_cache(graph)
                                st.rerun()
                            else:
                                st.error("削除に失敗しました")
                    with ec2:
                        if st.button("キャンセル", key=f"confirm_delete_edge_no_{selected_idx}", width="stretch"):
                            st.session_state[confirm_edge_key] = False
                            st.rerun()

    # CSVダウンロード
    csv_edges = edges_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 エッジをCSVでダウンロード",
        data=csv_edges,
        file_name="edges.csv",
        mime="text/csv"
    )

    # 統計情報
    st.markdown("---")
    st.subheader("📊 統計情報")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("総ノード数", len(nodes_dict))
    with col2:
        st.metric("総エッジ数", len(edges_list))
    with col3:
        avg_degree = sum(n['接続数'] for n in nodes_dict.values()) / len(nodes_dict) if nodes_dict else 0
        st.metric("平均接続数", f"{avg_degree:.1f}")
