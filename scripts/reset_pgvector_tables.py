#!/usr/bin/env python3
"""
PostgreSQL テーブルリセットスクリプト
langchain_postgres に自動でテーブルを作成させるため、既存テーブルを削除します。

使用方法:
    python reset_pgvector_tables.py
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import psycopg
from graphrag_core.config import get_settings
from graphrag_core.db.utils import normalize_pg_connection_string


def reset_tables():
    """langchain_pg_embedding と langchain_pg_collection テーブルを削除
    （langchain_postgres が自動で正しいスキーマで再作成する）
    """

    s = get_settings()
    pg_conn = s.pg_conn
    if not pg_conn:
        print("❌ エラー: PG_CONN 環境変数が設定されていません")
        return False

    conn_string = normalize_pg_connection_string(pg_conn)

    print(f"🔗 PostgreSQL に接続中...")
    print(f"   接続先: {conn_string.split('@')[1] if '@' in conn_string else conn_string}")

    try:
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                # Step 1: pgvector 拡張を確認
                print("\n🔧 pgvector 拡張を確認中...")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                print("   ✅ pgvector 拡張確認完了")

                # Step 2: 既存テーブルを完全に削除
                print("\n🗑️  既存テーブルを削除中...")
                cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE")
                cur.execute("DROP TABLE IF EXISTS langchain_pg_collection CASCADE")
                print("   ✅ テーブル削除完了")

                # テーブルは作成しない - langchain_postgres が自動で作成する

            conn.commit()

        print("\n" + "=" * 50)
        print("✅ テーブル削除完了!")
        print("=" * 50)
        print("\n次のステップ:")
        print("  1. Streamlit アプリを再起動")
        print("  2. ドキュメントを再アップロード")
        print("\nNote: langchain_postgres がドキュメント追加時に")
        print("      正しいスキーマでテーブルを自動作成します")
        return True

    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("PostgreSQL テーブルリセットスクリプト")
    print("langchain_postgres 用にテーブルを削除します")
    print("=" * 50)

    confirm = input("\n⚠️  既存のベクトルデータは全て削除されます。続行しますか? (y/N): ")
    if confirm.lower() == 'y':
        reset_tables()
    else:
        print("キャンセルしました")
