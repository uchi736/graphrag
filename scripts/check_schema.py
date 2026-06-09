"""PostgreSQLテーブルスキーマ確認スクリプト"""
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import psycopg
from graphrag_core.config import get_settings
from graphrag_core.db.utils import normalize_pg_connection_string

s = get_settings()
PG_CONN = normalize_pg_connection_string(s.pg_conn)

with psycopg.connect(PG_CONN) as conn:
    with conn.cursor() as cur:
        print("=" * 60)
        print("langchain_pg_embedding テーブル構造")
        print("=" * 60)

        cur.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = 'langchain_pg_embedding'
            ORDER BY ordinal_position
        """)
        rows = cur.fetchall()
        if rows:
            print(f"{'カラム名':<25} {'型':<20} {'NULL可':<8} {'デフォルト'}")
            print("-" * 60)
            for row in rows:
                col, dtype, nullable, default = row
                default_str = str(default)[:20] if default else ""
                print(f"{col:<25} {dtype:<20} {nullable:<8} {default_str}")
        else:
            print("テーブルが存在しません")

        print("\n" + "=" * 60)
        print("インデックス一覧")
        print("=" * 60)

        cur.execute("""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'langchain_pg_embedding'
        """)
        for row in cur.fetchall():
            print(f"・{row[0]}")
            print(f"  {row[1][:80]}...")

        print("\n" + "=" * 60)
        print("langchain_pg_collection テーブル構造")
        print("=" * 60)

        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'langchain_pg_collection'
            ORDER BY ordinal_position
        """)
        rows = cur.fetchall()
        if rows:
            print(f"{'カラム名':<25} {'型':<20} {'NULL可'}")
            print("-" * 60)
            for row in rows:
                print(f"{row[0]:<25} {row[1]:<20} {row[2]}")
        else:
            print("テーブルが存在しません")

        print("\n" + "=" * 60)
        print("登録済みコレクション一覧")
        print("=" * 60)

        cur.execute("""
            SELECT name, uuid,
                   (SELECT COUNT(*) FROM langchain_pg_embedding e
                    WHERE e.collection_id = c.uuid) as doc_count
            FROM langchain_pg_collection c
            ORDER BY name
        """)
        rows = cur.fetchall()
        if rows:
            print(f"{'コレクション名':<30} {'UUID':<40} {'文書数'}")
            print("-" * 80)
            for row in rows:
                name, uuid, count = row
                print(f"{name:<30} {str(uuid):<40} {count}")
        else:
            print("コレクションが登録されていません")
