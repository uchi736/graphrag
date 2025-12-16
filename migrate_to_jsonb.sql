-- PostgreSQL Schema Migration Script for langchain_postgres
-- ============================================================
-- langchain_postgres (新しいパッケージ) は langchain_community.vectorstores.pgvector とは
-- 異なるテーブルスキーマを使用します。このスクリプトで既存テーブルを再作成します。

-- ============================================================
-- STEP 1: 既存テーブルを削除（データも削除されます！バックアップ推奨）
-- ============================================================
DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;
DROP TABLE IF EXISTS langchain_pg_collection CASCADE;

-- ============================================================
-- STEP 2: langchain_postgres 用の新しいテーブルを作成
-- ============================================================

-- コレクション管理テーブル
CREATE TABLE IF NOT EXISTS langchain_pg_collection (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR NOT NULL UNIQUE,
    cmetadata JSONB
);

-- 埋め込みベクトルテーブル（langchain_postgres v0.0.6+ 形式）
-- 注: id カラムが VARCHAR で必要（langchain_community 版は uuid だった）
CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    id VARCHAR PRIMARY KEY,
    collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
    embedding vector,  -- 次元は動的に決定される
    document VARCHAR,
    cmetadata JSONB
);

-- ============================================================
-- STEP 3: インデックス作成
-- ============================================================

-- コレクション名インデックス
CREATE INDEX IF NOT EXISTS ix_langchain_pg_collection_name
    ON langchain_pg_collection (name);

-- メタデータ用 GIN インデックス（JSONB用）
CREATE INDEX IF NOT EXISTS ix_cmetadata_gin
    ON langchain_pg_embedding
    USING gin (cmetadata jsonb_path_ops);

-- 日本語検索用カラム（オプション）
ALTER TABLE langchain_pg_embedding
    ADD COLUMN IF NOT EXISTS tokenized_content TEXT;

CREATE INDEX IF NOT EXISTS idx_embedding_tokenized_gin
    ON langchain_pg_embedding
    USING gin (to_tsvector('simple', COALESCE(tokenized_content, '')));

-- ============================================================
-- STEP 4: 確認
-- ============================================================
\echo '=== Table Structure ==='
\d langchain_pg_embedding

\echo ''
\echo '=== Indexes ==='
\di+ *langchain*

\echo ''
\echo 'Migration completed successfully!'
\echo ''
\echo 'Note: HNSW index will be created automatically when embedding dimensions are known.'
\echo 'Note: Use $ prefix operators for metadata filters ($eq, $ne, $gt, etc.)'
