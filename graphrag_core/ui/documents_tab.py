"""
graphrag_core/ui/documents_tab.py
『📄 登録ドキュメント』タブ。render_documents_tab(ctx)。
集計は services/documents.list_registered_documents に委譲（st非依存化）。
"""
import streamlit as st

from graphrag_core.services.documents import list_registered_documents
from graphrag_core.ui.feedback import show_error


# =====================================================================
# タブ3: 登録ドキュメント
# =====================================================================
def render_documents_tab(ctx):
    st.header("📄 登録ドキュメント")

    if ctx.pg_conn:
        try:
            data = list_registered_documents(ctx.pg_conn, ctx.pg_collection)
            docs = data["documents"]
            if docs:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("総チャンク数", f"{data['total_chunks']:,}")
                with col2:
                    st.metric("ドキュメント数", len(docs))

                st.markdown("### ソースファイル一覧")
                import pandas as pd
                df = pd.DataFrame(
                    [(d["source"], d["chunk_count"]) for d in docs],
                    columns=["ソースファイル", "チャンク数"],
                )
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.info("登録されたドキュメントはありません")
        except Exception as e:
            show_error("DB接続エラー", e)
    else:
        st.warning("PG_CONNが設定されていません")
