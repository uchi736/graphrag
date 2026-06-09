"""
日本語検索用のデータベース初期化スクリプト
既存データへの tokenized_content 追加

Usage:
    python init_japanese_search.py              # 通常初期化（NULL のみ処理）
    python init_japanese_search.py --retokenize # 全データ再トークン化
"""
import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import psycopg
from graphrag_core.config import get_settings
from graphrag_core.text.japanese import get_japanese_processor
from graphrag_core.db.utils import normalize_pg_connection_string

s = get_settings()
PG_CONN = s.pg_conn
COLLECTION_NAME = s.pg_collection

if not PG_CONN:
    print("❌ エラー: PG_CONN 環境変数が設定されていません")
    sys.exit(1)

# SQLAlchemy形式 → psycopg形式に正規化
RAW_PG_CONN = normalize_pg_connection_string(PG_CONN)


def init_db_schema():
    """スキーマ初期化"""
    print("📊 データベーススキーマを初期化しています...")
    try:
        with psycopg.connect(RAW_PG_CONN) as conn:
            with conn.cursor() as cur:
                # 列追加
                cur.execute("""
                    ALTER TABLE langchain_pg_embedding
                    ADD COLUMN IF NOT EXISTS tokenized_content TEXT
                """)

                # インデックス作成
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embedding_tokenized_gin
                    ON langchain_pg_embedding
                    USING gin (to_tsvector('simple', COALESCE(tokenized_content, '')))
                """)
            conn.commit()
        print("✅ スキーマ初期化完了")
        return True
    except Exception as e:
        print(f"❌ スキーマ初期化エラー: {e}")
        return False


def migrate_existing_data(retokenize: bool = False):
    """既存データにトークン化追加

    Args:
        retokenize: True の場合、全データの tokenized_content をリセットして再処理
    """
    processor = get_japanese_processor()
    if not processor:
        print("Sudachiが利用できません")
        print("   インストール: pip install sudachipy sudachidict_core")
        return False

    print(f"既存データを移行しています... (collection={COLLECTION_NAME})")
    try:
        with psycopg.connect(RAW_PG_CONN) as conn:
            with conn.cursor() as cur:
                if retokenize:
                    # 全データの tokenized_content をリセット
                    cur.execute("""
                        UPDATE langchain_pg_embedding
                        SET tokenized_content = NULL
                        FROM langchain_pg_collection c
                        WHERE langchain_pg_embedding.collection_id = c.uuid
                          AND c.name = %s
                    """, (COLLECTION_NAME,))
                    reset_count = cur.rowcount
                    conn.commit()
                    print(f"  {reset_count}件の tokenized_content をリセットしました")

                # tokenized_contentがNULLのレコードを取得
                cur.execute("""
                    SELECT e.id, e.document, e.cmetadata->>'id' as chunk_id
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                      AND e.tokenized_content IS NULL
                """, (COLLECTION_NAME,))

                rows = cur.fetchall()
                total = len(rows)

                if total == 0:
                    print("移行対象のデータがありません（既に移行済み）")
                    return True

                print(f"{total}件のレコードを処理します...")

                for idx, (record_id, text, chunk_id) in enumerate(rows, 1):
                    if idx % 10 == 0 or idx == total:
                        print(f"  処理中: {idx}/{total} ({idx*100//total}%)")

                    tokenized = processor.tokenize(text)
                    cur.execute("""
                        UPDATE langchain_pg_embedding
                        SET tokenized_content = %s
                        WHERE id = %s
                    """, (tokenized, record_id))

            conn.commit()
        print("既存データの移行完了")
        return True
    except Exception as e:
        print(f"データ移行エラー: {e}")
        return False


def verify_setup():
    """セットアップ確認"""
    print("\n📊 セットアップを確認しています...")
    try:
        with psycopg.connect(RAW_PG_CONN) as conn:
            with conn.cursor() as cur:
                # tokenized_content列の存在確認
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'langchain_pg_embedding'
                      AND column_name = 'tokenized_content'
                """)
                if cur.fetchone():
                    print("  ✅ tokenized_content列: 存在")
                else:
                    print("  ❌ tokenized_content列: 存在しない")
                    return False

                # インデックスの存在確認
                cur.execute("""
                    SELECT indexname
                    FROM pg_indexes
                    WHERE tablename = 'langchain_pg_embedding'
                      AND indexname = 'idx_embedding_tokenized_gin'
                """)
                if cur.fetchone():
                    print("  ✅ GINインデックス: 存在")
                else:
                    print("  ❌ GINインデックス: 存在しない")
                    return False

                # データ件数確認
                cur.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(tokenized_content) as tokenized
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                """, (COLLECTION_NAME,))
                row = cur.fetchone()
                if row:
                    total, tokenized = row
                    print(f"  📊 総レコード数: {total}")
                    print(f"  📊 トークン化済み: {tokenized} ({tokenized*100//total if total > 0 else 0}%)")

        print("\n✅ セットアップは正常です")
        return True
    except Exception as e:
        print(f"\n❌ セットアップ確認エラー: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="日本語ハイブリッド検索 初期化スクリプト")
    parser.add_argument(
        "--retokenize", action="store_true",
        help="全データの tokenized_content をリセットして再トークン化（トークン化方式変更後に使用）"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("日本語ハイブリッド検索 初期化スクリプト")
    if args.retokenize:
        print("  モード: 再トークン化（全データリセット）")
    print("=" * 60)
    print()

    # ステップ1: スキーマ初期化
    if not init_db_schema():
        print("\n初期化に失敗しました")
        sys.exit(1)

    print()

    # ステップ2: データ移行
    if args.retokenize:
        # --retokenize: 確認なしで全データ再トークン化
        if not migrate_existing_data(retokenize=True):
            print("\nデータ移行に失敗しました")
            sys.exit(1)
    else:
        migrate = input("既存データを移行しますか？ [y/N]: ").strip().lower()
        if migrate == 'y':
            print()
            if not migrate_existing_data():
                print("\nデータ移行に失敗しました")
                print("   ※ スキーマは初期化されているため、新規データは正常に処理されます")

    print()

    # ステップ3: セットアップ確認
    verify_setup()

    print()
    print("=" * 60)
    print("初期化完了")
    print("=" * 60)
