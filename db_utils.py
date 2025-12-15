"""
データベース接続ユーティリティ
SQLAlchemy形式の接続文字列をpsycopg/psycopg2形式に変換
"""


def normalize_pg_connection_string(conn_string: str) -> str:
    """
    SQLAlchemy形式の接続文字列をpsycopg/psycopg2形式に正規化

    SQLAlchemyは postgresql+psycopg:// や postgres+psycopg2:// のような
    スキームを使用しますが、psycopg/psycopg2は postgresql:// や postgres://
    のみをサポートします。この関数は両者の互換性を保ちます。

    Args:
        conn_string: 接続文字列
            例: postgresql+psycopg://user:pass@host:port/db
            例: postgres+psycopg2://user:pass@host:port/db
            例: postgresql://user:pass@host:port/db (変換不要)

    Returns:
        正規化された接続文字列
            例: postgresql://user:pass@host:port/db
            例: postgres://user:pass@host:port/db

    Raises:
        ValueError: 接続文字列が空、またはPostgreSQL形式でない場合

    Examples:
        >>> normalize_pg_connection_string("postgresql+psycopg://localhost/mydb")
        'postgresql://localhost/mydb'

        >>> normalize_pg_connection_string("postgres+psycopg2://localhost/mydb")
        'postgres://localhost/mydb'

        >>> normalize_pg_connection_string("postgresql://localhost/mydb")
        'postgresql://localhost/mydb'
    """
    if not conn_string:
        raise ValueError("接続文字列が空です")

    # スキーム検証
    if not (conn_string.startswith("postgresql") or conn_string.startswith("postgres")):
        raise ValueError(f"PostgreSQL接続文字列ではありません: {conn_string[:20]}...")

    # 正規化処理: postgresql+ または postgres+ のプレフィックスを除去
    if conn_string.startswith("postgresql+"):
        return "postgresql://" + conn_string.split("://", 1)[1]
    if conn_string.startswith("postgres+"):
        return "postgres://" + conn_string.split("://", 1)[1]

    # 既に正規化済み
    return conn_string


if __name__ == "__main__":
    # テストケース
    print("PostgreSQL接続文字列正規化ヘルパー - テスト")
    print("=" * 60)

    test_cases = [
        ("postgresql+psycopg://localhost/db", "postgresql://localhost/db"),
        ("postgres+psycopg2://user:pass@host:5432/db", "postgres://user:pass@host:5432/db"),
        ("postgresql://localhost/db", "postgresql://localhost/db"),
        ("postgres://localhost/db", "postgres://localhost/db"),
        ("postgresql+psycopg://user:p@ss@host/db", "postgresql://user:p@ss@host/db"),  # パスワードに特殊文字
    ]

    all_passed = True
    for input_str, expected in test_cases:
        try:
            result = normalize_pg_connection_string(input_str)
            if result == expected:
                print(f"✅ PASS: {input_str[:50]}")
            else:
                print(f"❌ FAIL: {input_str[:50]}")
                print(f"   期待: {expected}")
                print(f"   結果: {result}")
                all_passed = False
        except Exception as e:
            print(f"❌ ERROR: {input_str[:50]}")
            print(f"   エラー: {e}")
            all_passed = False

    # エラーケースのテスト
    print("\nエラーケースのテスト:")
    error_cases = [
        ("", ValueError),
        ("mysql://localhost/db", ValueError),
    ]

    for input_str, expected_error in error_cases:
        try:
            result = normalize_pg_connection_string(input_str)
            print(f"❌ FAIL: '{input_str}' should raise {expected_error.__name__}")
            all_passed = False
        except expected_error:
            print(f"✅ PASS: '{input_str}' correctly raised {expected_error.__name__}")
        except Exception as e:
            print(f"❌ FAIL: '{input_str}' raised {type(e).__name__} instead of {expected_error.__name__}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
