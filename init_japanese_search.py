"""
日本語検索用のデータベース初期化スクリプト
既存データへの tokenized_content 追加
"""
import os
import sys
import psycopg
from dotenv import load_dotenv
from japanese_text_processor import get_japanese_processor

load_dotenv()
PG_CONN = os.getenv("PG_CONN")

if not PG_CONN:
    print("❌ エラー: PG_CONN 環境変数が設定されていません")
    sys.exit(1)


def init_db_schema():
    """スキーマ初期化"""
    print("📊 データベーススキーマを初期化しています...")
    try:
        with psycopg.connect(PG_CONN) as conn:
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


def migrate_existing_data():
    """既存データにトークン化追加"""
    processor = get_japanese_processor()
    if not processor:
        print("❌ Sudachiが利用できません")
        print("   インストール: pip install sudachipy sudachidict_core")
        return False

    print("📊 既存データを移行しています...")
    try:
        with psycopg.connect(PG_CONN) as conn:
            with conn.cursor() as cur:
                # tokenized_contentがNULLのレコードを取得
                cur.execute("""
                    SELECT id, document
                    FROM langchain_pg_embedding
                    WHERE tokenized_content IS NULL
                """)

                rows = cur.fetchall()
                total = len(rows)

                if total == 0:
                    print("✅ 移行対象のデータがありません（既に移行済み）")
                    return True

                print(f"📊 {total}件のレコードを処理します...")

                for idx, (record_id, text) in enumerate(rows, 1):
                    if idx % 10 == 0 or idx == total:
                        print(f"  処理中: {idx}/{total} ({idx*100//total}%)")

                    tokenized = processor.tokenize(text)
                    cur.execute("""
                        UPDATE langchain_pg_embedding
                        SET tokenized_content = %s
                        WHERE id = %s
                    """, (tokenized, record_id))

            conn.commit()
        print("✅ 既存データの移行完了")
        return True
    except Exception as e:
        print(f"❌ データ移行エラー: {e}")
        return False


def verify_setup():
    """セットアップ確認"""
    print("\n📊 セットアップを確認しています...")
    try:
        with psycopg.connect(PG_CONN) as conn:
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
                    FROM langchain_pg_embedding
                """)
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
    print("=" * 60)
    print("日本語ハイブリッド検索 初期化スクリプト")
    print("=" * 60)
    print()

    # ステップ1: スキーマ初期化
    if not init_db_schema():
        print("\n❌ 初期化に失敗しました")
        sys.exit(1)

    print()

    # ステップ2: データ移行（オプション）
    migrate = input("既存データを移行しますか？ [y/N]: ").strip().lower()
    if migrate == 'y':
        print()
        if not migrate_existing_data():
            print("\n⚠️ データ移行に失敗しました")
            print("   ※ スキーマは初期化されているため、新規データは正常に処理されます")

    print()

    # ステップ3: セットアップ確認
    verify_setup()

    print()
    print("=" * 60)
    print("初期化完了")
    print("=" * 60)
    print()
    print("次のステップ:")
    print("  1. Sudachiをインストール: pip install sudachipy sudachidict_core")
    print("  2. .envに ENABLE_JAPANESE_SEARCH=true を追加")
    print("  3. アプリを起動: streamlit run app.py")
    print()
