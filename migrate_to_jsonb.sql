-- =============================================================================
-- langchain_pg_embedding π≠¸ﬁ˚LπØÍ◊»
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. PRIMARY KEY í uuid í id k	Ù
--    1: langchain-postgres o INSERT Bk uuid íöWjD_Å
--          uuid L PK `h NOT NULL 6UÕkjã
--          id (TEXT) o langchain-postgres L≈Z-öYã_Å PK ki
-- ---------------------------------------------------------------------------
-- Step 1: ‚Xn PK íJd
ALTER TABLE langchain_pg_embedding DROP CONSTRAINT langchain_pg_embedding_pkey;

-- Step 2: id í∞WD PK k-ö
ALTER TABLE langchain_pg_embedding ADD PRIMARY KEY (id);

-- Step 3: uuid n NOT NULL 6íYPK gjOjc_ngÔ˝	
ALTER TABLE langchain_pg_embedding ALTER COLUMN uuid DROP NOT NULL;

-- ---------------------------------------------------------------------------
-- 2. tokenized_content ´È‡˝†Â,ûBM25"(	
--    ensure_tokenized_schema() gÍ’˝†UåãLK’üL(k	
-- ---------------------------------------------------------------------------
-- ALTER TABLE langchain_pg_embedding ADD COLUMN IF NOT EXISTS tokenized_content TEXT;

-- ---------------------------------------------------------------------------
-- ∫çØ®Í
-- ---------------------------------------------------------------------------
-- PK ∫ç
-- SELECT kcu.column_name
-- FROM information_schema.key_column_usage kcu
-- JOIN information_schema.table_constraints tc
--   ON kcu.constraint_name = tc.constraint_name
-- WHERE tc.table_name = 'langchain_pg_embedding'
--   AND tc.constraint_type = 'PRIMARY KEY';

-- ´È‡s0
-- SELECT column_name, data_type, is_nullable, column_default
-- FROM information_schema.columns
-- WHERE table_name = 'langchain_pg_embedding'
-- ORDER BY ordinal_position;
