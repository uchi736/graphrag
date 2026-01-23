#!/usr/bin/env python
"""PostgreSQL接続テスト"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

PG_CONN = os.getenv("PG_CONN")
print(f"PG_CONN: {PG_CONN[:50]}..." if PG_CONN else "PG_CONN: 未設定")

if not PG_CONN:
    print("ERROR: PG_CONNが設定されていません")
    sys.exit(1)

# 1. psycopgで直接接続テスト
print("\n=== psycopg直接接続テスト ===")
try:
    import psycopg
    from db_utils import normalize_pg_connection_string

    raw_conn = normalize_pg_connection_string(PG_CONN)
    print(f"正規化後: {raw_conn[:50]}...")

    with psycopg.connect(raw_conn, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            print(f"✅ 接続成功!")
            print(f"   PostgreSQL: {version[:50]}...")

            # pgvector拡張確認
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
            if row:
                print(f"   pgvector: v{row[0]}")
            else:
                print("   ⚠️ pgvector拡張がインストールされていません")

except Exception as e:
    print(f"❌ 接続失敗: {e}")
    sys.exit(1)

# 2. langchain-postgres PGVectorテスト
print("\n=== PGVector接続テスト ===")
try:
    from langchain_postgres import PGVector
    from langchain_openai import AzureOpenAIEmbeddings

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    vector_store = PGVector(
        connection=PG_CONN,
        embeddings=embeddings
    )
    print("✅ PGVector初期化成功!")

except Exception as e:
    print(f"❌ PGVector初期化失敗: {e}")
    sys.exit(1)

print("\n=== すべてのテスト完了 ===")
