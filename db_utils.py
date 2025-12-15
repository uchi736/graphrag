"""
PostgreSQL 接続ユーティリティ
- SQLAlchemy 形式の接続文字列を psycopg/psycopg2 で使える形に正規化
- tokenized_content カラムと GIN インデックスを自動で保証
"""

import psycopg


def normalize_pg_connection_string(conn_string: str) -> str:
    """
    SQLAlchemy 形式の接続文字列を psycopg/psycopg2 用に正規化する。

    Args:
        conn_string: 接続文字列
            例: postgresql+psycopg://user:pass@host:port/db
                postgres+psycopg2://user:pass@host:port/db
                postgresql://user:pass@host:port/db

    Returns:
        正規化済みの接続文字列（postgresql:// か postgres://）

    Raises:
        ValueError: 接続文字列が空、または PostgreSQL 以外の場合
    """
    if not conn_string:
        raise ValueError("接続文字列が空です")

    if not (conn_string.startswith("postgresql") or conn_string.startswith("postgres")):
        raise ValueError(f"PostgreSQL の接続文字列ではありません: {conn_string[:20]}...")

    # ドライバ指定を除去
    if conn_string.startswith("postgresql+"):
        return "postgresql://" + conn_string.split("://", 1)[1]
    if conn_string.startswith("postgres+"):
        return "postgres://" + conn_string.split("://", 1)[1]

    return conn_string


def ensure_tokenized_schema(conn_string: str) -> None:
    """tokenized_content カラムと GIN インデックスを保証する（冪等）。"""
    raw_conn = normalize_pg_connection_string(conn_string)
    with psycopg.connect(raw_conn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                ALTER TABLE langchain_pg_embedding
                ADD COLUMN IF NOT EXISTS tokenized_content TEXT
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embedding_tokenized_gin
                ON langchain_pg_embedding
                USING gin (to_tsvector('simple', COALESCE(tokenized_content, '')))
                """
            )
        conn.commit()
