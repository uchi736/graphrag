# Claude Code プロジェクト指示

## PostgreSQL スキーマ変更時の注意

**他プログラムと共有しているDBを操作する際は、必ず先にスキーマを確認すること：**

1. カラムの制約を変更する前に、以下を確認：
   - プライマリキーの一部かどうか
   - ユニーク制約があるかどうか
   - 外部キー参照があるかどうか

2. `DROP NOT NULL` は以下の場合に失敗する：
   - カラムがプライマリキーの一部
   - → 代わりに `SET DEFAULT` を使用

3. スキーマ確認用クエリ：
```sql
-- プライマリキー確認
SELECT kcu.column_name
FROM information_schema.key_column_usage kcu
JOIN information_schema.table_constraints tc
  ON kcu.constraint_name = tc.constraint_name
WHERE tc.table_name = 'テーブル名'
  AND tc.constraint_type = 'PRIMARY KEY';

-- カラム詳細確認
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_name = 'テーブル名';
```

4. `check_schema.py` を使用してスキーマを確認してから修正を行う

## 共有DB環境

- `langchain_pg_embedding` テーブルは他プログラムと共有
- 他プログラムは `uuid` をプライマリキーとして使用
- langchain-postgres は `id` (TEXT) を識別子として使用
- コレクション (`collection_id`) でデータを分離

## コード変更時のルール

- **変更後は必ず動作チェックを行うこと**（スキップ不可）
- Pythonコードの変更後は最低限 `python -c "..."` で該当機能の動作確認を実行する
- グラフ関連の変更は `graph.pkl` を使った実データでの検証を含めること
