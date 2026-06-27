"""
graphrag_core/ui/documents_tab.py
『📄 登録ドキュメント』タブ。render_documents_tab(ctx)。
langchain_pg_embedding をソース別にチャンク数集計して表示する読み取り専用ビュー。
"""
import streamlit as st

from graphrag_core.ui.feedback import show_error


# =====================================================================
# タブ3: 登録ドキュメント
# =====================================================================
def render_documents_tab(ctx):
    st.header("📄 登録ドキュメント")

    if ctx.pg_conn:
        try:
            import psycopg
            raw_conn = ctx.normalize_pg_connection_string(ctx.pg_conn)
            with psycopg.connect(raw_conn) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT
                            COALESCE(e.cmetadata->>'source', '(unknown)') as source,
                            COUNT(*) as chunk_count
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = %s
                        GROUP BY e.cmetadata->>'source'
                        ORDER BY chunk_count DESC
                    """, (ctx.pg_collection,))
                    rows = cur.fetchall()

            if rows:
                total_chunks = sum(r[1] for r in rows)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("総チャンク数", f"{total_chunks:,}")
                with col2:
                    st.metric("ドキュメント数", len(rows))

                st.markdown("### ソースファイル一覧")
                import pandas as pd
                df = pd.DataFrame(rows, columns=["ソースファイル", "チャンク数"])
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.info("登録されたドキュメントはありません")
        except Exception as e:
            show_error("DB接続エラー", e)
    else:
        st.warning("PG_CONNが設定されていません")
