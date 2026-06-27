"""グラフ編集ダイアログ関数

ノード・エッジの編集・削除用のStreamlitダイアログUI。
"""

import json
import streamlit as st


def edit_node_dialog(graph, node_info=None,
                     graph_update_node=None, graph_add_node=None,
                     graph_get_data_for_cache=None):
    """ノード編集ダイアログ

    Parameters
    ----------
    graph : object
        グラフバックエンドインスタンス
    node_info : dict or None
        編集対象のノード情報。Noneの場合は新規追加モード。
        期待するキー: 'id', 'type', 'properties'
    graph_update_node : callable, optional
        ノード更新関数 (graph, node_id, node_type, properties) -> bool
    graph_add_node : callable, optional
        ノード追加関数 (graph, node_id, node_type, properties) -> bool
    graph_get_data_for_cache : callable, optional
        キャッシュ用データ取得関数 (graph) -> list
    """
    st.subheader("✏️ ノード編集" if node_info else "➕ 新規ノード追加")

    with st.form("node_form"):
        if node_info:
            node_id = st.text_input("ノードID", value=node_info['id'], disabled=True)
            node_type = st.text_input("タイプ", value=node_info.get('type', 'Unknown'))
            properties_str = st.text_area(
                "プロパティ (JSON形式)",
                value=json.dumps(node_info.get('properties', {}), ensure_ascii=False, indent=2)
            )
        else:
            node_id = st.text_input("ノードID", placeholder="例: API")
            node_type = st.text_input("タイプ", value="Unknown", placeholder="例: Person")
            properties_str = st.text_area("プロパティ (JSON形式)", value="{}")

        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("💾 保存", type="primary")
        with col2:
            cancel = st.form_submit_button("❌ キャンセル")

        if submit:
            try:
                properties = json.loads(properties_str) if properties_str.strip() else {}

                if node_info:
                    # 更新
                    success = graph_update_node(graph, node_id, node_type, properties) if graph_update_node else False
                    if success:
                        st.toast(f"ノード '{node_id}' を更新しました", icon="✅")
                        st.session_state.editing_node = None
                        if graph_get_data_for_cache:
                            st.session_state.graph_data_cache = graph_get_data_for_cache(graph)
                        st.rerun()
                    else:
                        st.error("更新に失敗しました")
                else:
                    # 新規追加
                    if not node_id:
                        st.error("ノードIDを入力してください")
                    else:
                        success = graph_add_node(graph, node_id, node_type, properties) if graph_add_node else False
                        if success:
                            st.toast(f"ノード '{node_id}' を追加しました", icon="✅")
                            st.session_state.edit_mode = None
                            if graph_get_data_for_cache:
                                st.session_state.graph_data_cache = graph_get_data_for_cache(graph)
                            st.rerun()
                        else:
                            st.error("追加に失敗しました")
            except json.JSONDecodeError:
                st.error("プロパティのJSON形式が不正です")

        if cancel:
            # 編集状態をクリア
            st.session_state.editing_node = None
            st.session_state.edit_mode = None
            st.rerun()


def edit_edge_dialog(graph, edge_info=None, all_nodes=None,
                     graph_update_edge=None, graph_add_edge=None,
                     graph_get_data_for_cache=None):
    """エッジ編集ダイアログ

    Parameters
    ----------
    graph : object
        グラフバックエンドインスタンス
    edge_info : dict or None
        編集対象のエッジ情報。Noneの場合は新規追加モード。
        期待するキー: 'source', 'target', 'edge_key', 'type', 'properties'
    all_nodes : list or None
        ノード選択肢用の全ノードIDリスト
    graph_update_edge : callable, optional
        エッジ更新関数 (graph, source, target, edge_key, rel_type, properties) -> bool
    graph_add_edge : callable, optional
        エッジ追加関数 (graph, source, target, rel_type, properties) -> int or None
    graph_get_data_for_cache : callable, optional
        キャッシュ用データ取得関数 (graph) -> list
    """
    st.subheader("✏️ エッジ編集" if edge_info else "➕ 新規エッジ追加")

    if all_nodes is None:
        all_nodes = []

    with st.form("edge_form"):
        if edge_info:
            source = st.text_input("始点ノード", value=edge_info['source'], disabled=True)
            target = st.text_input("終点ノード", value=edge_info['target'], disabled=True)
            edge_key = edge_info.get('edge_key', 0)
            rel_type = st.text_input("リレーションタイプ", value=edge_info.get('type', 'RELATED'))
            properties_str = st.text_area(
                "プロパティ (JSON形式)",
                value=json.dumps(edge_info.get('properties', {}), ensure_ascii=False, indent=2)
            )
        else:
            if all_nodes:
                source = st.selectbox("始点ノード", options=all_nodes)
                target = st.selectbox("終点ノード", options=all_nodes)
            else:
                source = st.text_input("始点ノード", placeholder="例: API")
                target = st.text_input("終点ノード", placeholder="例: データベース")
            edge_key = 0
            rel_type = st.text_input("リレーションタイプ", value="RELATED", placeholder="例: USES")
            properties_str = st.text_area("プロパティ (JSON形式)", value="{}")

        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("💾 保存", type="primary")
        with col2:
            cancel = st.form_submit_button("❌ キャンセル")

        if submit:
            try:
                properties = json.loads(properties_str) if properties_str.strip() else {}

                if edge_info:
                    # 更新
                    success = graph_update_edge(graph, source, target, edge_key, rel_type, properties) if graph_update_edge else False
                    if success:
                        st.toast(f"エッジ '{source} -> {target}' を更新しました", icon="✅")
                        st.session_state.editing_edge = None
                        if graph_get_data_for_cache:
                            st.session_state.graph_data_cache = graph_get_data_for_cache(graph)
                        st.rerun()
                    else:
                        st.error("更新に失敗しました")
                else:
                    # 新規追加
                    if not source or not target:
                        st.error("始点と終点を指定してください")
                    else:
                        result_key = graph_add_edge(graph, source, target, rel_type, properties) if graph_add_edge else None
                        if result_key is not None:
                            st.toast(f"エッジ '{source} -> {target}' を追加しました", icon="✅")
                            st.session_state.edit_mode = None
                            if graph_get_data_for_cache:
                                st.session_state.graph_data_cache = graph_get_data_for_cache(graph)
                            st.rerun()
                        else:
                            st.error("追加に失敗しました")
            except json.JSONDecodeError:
                st.error("プロパティのJSON形式が不正です")

        if cancel:
            # 編集状態をクリア
            st.session_state.editing_edge = None
            st.session_state.edit_mode = None
            st.rerun()


def confirm_delete_dialog(item_type, item_name, callback):
    """削除確認ダイアログ

    Parameters
    ----------
    item_type : str
        削除対象の種類（例: "ノード", "エッジ"）
    item_name : str
        削除対象の名前
    callback : callable
        削除実行関数。成功時にTrueを返す。
    """
    st.warning(f"⚠️ {item_type} '{item_name}' を削除しますか？")
    st.caption("この操作は取り消せません。")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 削除する", type="primary"):
            if callback():
                st.success(f"✅ {item_type} '{item_name}' を削除しました")
                st.session_state.graph_data_cache = None  # キャッシュクリア
                st.rerun()
            else:
                st.error("削除に失敗しました")
    with col2:
        if st.button("❌ キャンセル"):
            st.rerun()
