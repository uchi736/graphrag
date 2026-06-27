"""
graphrag_core/ui/build_tab.py
『🛠️ 構築/取り込み』タブ。render_build_tab(ctx)。
既存グラフ検出/読込/クリア、ファイル・CSVアップロード、新規/再開/チャンクのみ構築。
"""
import streamlit as st
from langchain_neo4j import Neo4jGraph

from graphrag_core.ui.feedback import show_error
from graphrag_core.ui.state import check_existing_graph
from graphrag_core.graph.provenance import graph_collection_status, stamp_graph_provenance
from graphrag_core.ui.system import (
    restore_from_existing_graph,
    build_rag_system,
    update_chunks_only,
    load_documents,
    load_csv_edges,
)


# =====================================================================
# メインUI: 構築/取り込みタブ
# =====================================================================
def render_build_tab(ctx):
    """『🛠️ 構築/取り込み』タブ: アップロード・既存グラフ読込・構築。"""
    st.header("🛠️ 構築 / 取り込み")

    # 既存グラフのチェック（未初期化時のみ）
    if not st.session_state.existing_graph_loaded and not st.session_state.initialized:
        try:
            temp_graph = ctx.create_graph_instance()
            graph_info = check_existing_graph(temp_graph)

            if graph_info['exists']:
                st.info(f"📊 既存のナレッジグラフを発見しました: ノード {graph_info['node_count']}個、リレーションシップ {graph_info['rel_count']}本")

                # グラフ↔現コレクションの整合性
                gcs = graph_collection_status(temp_graph, ctx.pg_collection)
                if gcs["status"] == "match":
                    st.success(f"✅ このグラフは現在のコレクション「{ctx.pg_collection}」用です（KG有効）。")
                else:
                    _src = gcs["graph_collection"] or "不明"
                    _why = "別コレクション用" if gcs["status"] == "mismatch" else "出自未記録"
                    st.warning(
                        f"⚠️ このグラフは現コレクション「{ctx.pg_collection}」と不整合です（出自: {_src} / {_why}）。"
                        "このまま読み込むと回答時に KG はスキップされ、ベクトル+BM25のみになります。"
                    )
                    if st.button(f"🏷️ このグラフを「{ctx.pg_collection}」用として記録（出自が既知の場合のみ）"):
                        if stamp_graph_provenance(temp_graph, ctx.pg_collection):
                            st.success("✅ 出自を記録しました。読み込み後 KG が有効になります。")
                            st.rerun()
                        else:
                            st.error("記録に失敗しました")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 既存グラフを読み込む", type="primary"):
                        with st.spinner("既存グラフからシステムを復元中..."):
                            try:
                                st.session_state.chain, st.session_state.graph = restore_from_existing_graph(ctx)
                                st.session_state.initialized = True
                                st.session_state.existing_graph_loaded = True
                                st.success("✅ 既存グラフから復元完了！「💬 質問応答」タブで質問できます。")
                                st.rerun()
                            except Exception as e:
                                show_error("復元エラー", e)

                with col2:
                    if st.button("🗑️ 既存グラフをクリアして新規作成"):
                        with st.spinner("既存データをクリア中..."):
                            try:
                                temp_graph.query("MATCH (n) DETACH DELETE n")
                                st.session_state.existing_graph_loaded = True
                                st.success("✅ クリア完了。新しいドキュメントをアップロードしてください。")
                                st.rerun()
                            except Exception as e:
                                show_error("クリアエラー", e)

                st.markdown("---")
        except Exception as e:
            st.caption(f"既存グラフの確認に失敗しました（接続を確認してください）: {e}")

    uploaded_files = st.file_uploader(
        "PDF/テキスト/Markdownファイルをアップロード",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="複数ファイルをアップロード可能。Azure DI処理済みの_azure_di.mdファイルも再利用可能"
    )
    csv_edges_file = st.file_uploader(
        "edges.csv (source,target,label)",
        type=["csv"],
        accept_multiple_files=False,
        help="シンプルなノード・エッジ関係をCSVで追加する場合に指定してください"
    )
    has_docs = bool(uploaded_files)
    has_csv = bool(csv_edges_file)

    if has_docs:
        st.success(f"✅ {len(uploaded_files)} ファイルがアップロードされました")
        with st.expander("📄 アップロード済みファイル"):
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size} bytes)")

    if has_csv:
        st.info(f"🔗 edges.csv を受信: {csv_edges_file.name}")

    # ナレッジグラフ構築ボタン
    if has_docs or has_csv:
        col1, col2 = st.columns(2)
        with col1:
            new_build = st.button("🚀 新規構築", type="primary", help="処理済みデータをクリアして最初から構築")
        with col2:
            resume_build = st.button("▶️ 続きから再開", help="処理済みチャンクをスキップして続きから構築")

        st.caption("⚡ 高速オプション")
        chunks_only = st.button("📄 チャンクのみ更新", help="グラフ再構築をスキップしてPGVectorのチャンクのみ更新（高速）")

        if new_build or resume_build:
            if new_build:
                try:
                    temp_graph = Neo4jGraph(url=ctx.neo4j_uri, username=ctx.neo4j_user, password=ctx.neo4j_pw)
                    temp_graph.query("MATCH (c:ProcessedChunk) DELETE c")
                    st.info("🗑️ 処理済みデータをクリアしました")
                except Exception as e:
                    st.warning(f"クリア処理でエラー（続行します）: {e}")

            source_docs = []
            if has_docs:
                with st.spinner("ドキュメント読み込み中..."):
                    try:
                        source_docs = load_documents(uploaded_files)
                        total_chars = sum(len(doc.page_content) for doc in source_docs)
                        st.info(f"📄 {len(source_docs)} ファイル読み込み完了（総文字数: {total_chars:,} 文字）")
                    except Exception as e:
                        show_error("ファイル読み込みエラー", e)
                        return

            with st.spinner("ナレッジグラフ構築中... (数分かかる場合があります)"):
                try:
                    csv_edges = load_csv_edges(csv_edges_file) if has_csv else []
                    st.session_state.chain, st.session_state.graph = build_rag_system(ctx, source_docs, csv_edges)
                    st.session_state.uploaded_files = [f.name for f in uploaded_files] if has_docs else []
                    st.session_state.graph_data_cache = None
                    if 'all_node_list' in st.session_state:
                        st.session_state.all_node_list = None
                    # 構築の成否に応じて成功/部分成功/失敗を出し分ける（誤った成功表示を防ぐ）
                    stats = st.session_state.get("last_build_stats", {"ok": 0, "err": 0, "total": 0})
                    ok, err, total = stats["ok"], stats["err"], stats["total"]
                    if err == 0:
                        st.session_state.initialized = True
                        st.success("✅ ナレッジグラフ構築完了！「💬 質問応答」タブで質問できます。")
                        st.rerun()
                    elif ok > 0:
                        st.session_state.initialized = True
                        st.warning(f"⚠️ 部分的に完了: {ok}/{total} チャンクを処理、{err}件失敗。質問は可能ですが結果が不完全な可能性があります。")
                    else:
                        st.error(f"❌ 全 {total} チャンクの処理に失敗しました。ログ/接続を確認してください。")
                except Exception as e:
                    show_error("構築エラー", e)

        if chunks_only:
            if not has_docs:
                st.error("ドキュメントをアップロードしてください")
            else:
                with st.spinner("ドキュメント読み込み中..."):
                    try:
                        source_docs = load_documents(uploaded_files)
                        total_chars = sum(len(doc.page_content) for doc in source_docs)
                        st.info(f"📄 {len(source_docs)} ファイル読み込み完了（総文字数: {total_chars:,} 文字）")
                    except Exception as e:
                        show_error("ファイル読み込みエラー", e)
                        return

                with st.spinner("チャンクを更新中..."):
                    try:
                        vector_store = update_chunks_only(ctx, source_docs)
                        if vector_store:
                            st.session_state.uploaded_files = [f.name for f in uploaded_files]
                    except Exception as e:
                        show_error("チャンク更新エラー", e)
