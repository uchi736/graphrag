"""
PostgreSQL 接続ユーティリティ
- SQLAlchemy 形式の接続文字列を psycopg/psycopg2 で使える形に正規化
- tokenized_content カラムと GIN インデックスを自動で保証
"""

import psycopg
import time
from typing import Callable, TypeVar

T = TypeVar('T')


def add_connection_timeout(conn_string: str, timeout: int = 30) -> str:
    """接続文字列にタイムアウトを追加する。

    Args:
        conn_string: 接続文字列
        timeout: 接続タイムアウト秒数（デフォルト: 30秒）

    Returns:
        タイムアウト付きの接続文字列
    """
    if not conn_string:
        return conn_string
    if "connect_timeout" in conn_string:
        return conn_string  # 既にタイムアウト設定済み
    if "?" in conn_string:
        return f"{conn_string}&connect_timeout={timeout}"
    return f"{conn_string}?connect_timeout={timeout}"


def retry_on_timeout(func: Callable[[], T], max_retries: int = 3, delay: float = 2.0) -> T:
    """タイムアウト時にリトライするラッパー。

    Args:
        func: 実行する関数
        max_retries: 最大リトライ回数
        delay: リトライ間隔（秒）

    Returns:
        関数の戻り値

    Raises:
        最後の試行で発生した例外
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            # タイムアウト関連のエラーのみリトライ
            if "timeout" in error_str or "connection" in error_str:
                if attempt < max_retries - 1:
                    print(f"接続リトライ中... ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                continue
            raise  # タイムアウト以外のエラーは即座に再送出
    raise last_exception


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


def ensure_embedding_id_unique(conn_string: str) -> None:
    """langchain_pg_embedding の id カラムにユニーク制約を保証する（冪等）。

    langchain-postgres は内部で ON CONFLICT (id) を使用するため、
    id カラムにユニーク制約がないとエラーになる。
    """
    raw_conn = normalize_pg_connection_string(conn_string)
    with psycopg.connect(raw_conn) as conn:
        with conn.cursor() as cur:
            # テーブルが存在するか確認
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'langchain_pg_embedding'
                )
                """
            )
            if not cur.fetchone()[0]:
                # テーブルがまだない場合はスキップ
                return

            # ユニークインデックスを作成（存在しなければ）
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_langchain_pg_embedding_id_unique
                ON langchain_pg_embedding (id)
                """
            )
        conn.commit()


def ensure_schema_compatibility(conn_string: str) -> None:
    """他プログラムが追加したカラムの互換性を確保する（冪等）。

    langchain-postgres は uuid 等のカラムを使用しないため、
    他プログラムがNOT NULL制約付きで追加した場合にINSERTが失敗する。
    uuid がプライマリキーの場合は DEFAULT 値を設定して自動生成させる。
    """
    raw_conn = normalize_pg_connection_string(conn_string)
    with psycopg.connect(raw_conn) as conn:
        with conn.cursor() as cur:
            # テーブルが存在するか確認
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'langchain_pg_embedding'
                )
                """
            )
            if not cur.fetchone()[0]:
                return

            # uuid カラムが存在するか確認
            cur.execute(
                """
                SELECT column_name, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = 'langchain_pg_embedding'
                AND column_name = 'uuid'
                """
            )
            row = cur.fetchone()
            if not row:
                return

            col_name, is_nullable, col_default = row

            # uuid がプライマリキーの一部かチェック
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.key_column_usage kcu
                    JOIN information_schema.table_constraints tc
                      ON kcu.constraint_name = tc.constraint_name
                    WHERE tc.table_name = 'langchain_pg_embedding'
                      AND tc.constraint_type = 'PRIMARY KEY'
                      AND kcu.column_name = 'uuid'
                )
                """
            )
            is_primary_key = cur.fetchone()[0]

            if is_primary_key:
                # プライマリキーの場合はデフォルト値を設定（NOT NULLは外せない）
                if col_default is None:
                    cur.execute(
                        """
                        ALTER TABLE langchain_pg_embedding
                        ALTER COLUMN uuid SET DEFAULT gen_random_uuid()
                        """
                    )
            else:
                # プライマリキーでない場合は NOT NULL を外す
                if is_nullable == 'NO':
                    cur.execute(
                        """
                        ALTER TABLE langchain_pg_embedding
                        ALTER COLUMN uuid DROP NOT NULL
                        """
                    )
        conn.commit()


def ensure_hnsw_index(conn_string: str) -> None:
    """コサイン距離用の HNSW インデックスを保証する（冪等）。"""
    raw_conn = normalize_pg_connection_string(conn_string)
    with psycopg.connect(raw_conn) as conn:
        with conn.cursor() as cur:
            # ベクトル次元が未指定のテーブルではHNSWを張れないので確認
            cur.execute(
                """
                SELECT atttypmod
                FROM pg_attribute
                WHERE attrelid = 'langchain_pg_embedding'::regclass
                  AND attname = 'embedding'
                  AND NOT attisdropped
                """
            )
            row = cur.fetchone()
            dim = row[0] - 4 if row else None  # vectorのtypmodから次元を計算（typmod=-1なら未指定）
            if not dim or dim <= 0:
                # 次元未指定の場合はインデックス作成をスキップ（エラー防止）
                print("Skip HNSW index: embedding column has no fixed dimension")
                return

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embedding_hnsw
                ON langchain_pg_embedding
                USING hnsw (embedding vector_cosine_ops)
                """
            )
        conn.commit()
