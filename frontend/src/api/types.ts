/**
 * API契約の手書き型定義。
 * OpenAPIに乗らない SSE ペイロードと、生成移行（P5）までの暫定REST型。
 * バックエンド（api/routers/*）と対で更新すること。
 */

// ---- GET /api/health ----
export interface HealthReport {
  ok: boolean
  status: "ok" | "degraded"
  checks: {
    env: { ok: boolean; missing: string[] }
    neo4j: { ok: boolean; error: string | null }
    llm: { provider?: string; model?: string; status?: string }
    embedding: { provider: string }
    pdf: { processor: string; backend: string | null; endpoint: string | null; ok: boolean; note: string }
    sudachi_available: boolean
  }
  graph: { exists: boolean; node_count: number; rel_count: number; error?: string | null }
  provenance: { status: "match" | "mismatch" | "unknown"; graph_collection: string | null }
  collection: string
  startup_error?: string | null
}

// ---- GET /api/settings ----
export interface QaDefaults {
  retrieval_top_k: number
  enable_rerank: boolean
  enable_japanese_search: boolean
  search_mode: "hybrid" | "vector" | "keyword" | "none"
  enable_knowledge_graph: boolean
  include_kg_source_chunks: boolean
  graph_hop_count: number
  enable_entity_vector: boolean
  entity_similarity_threshold: number
}

export interface SettingsInfo {
  collection: string
  llm: { provider: string; endpoint: string; model: string; status: string }
  embedding_provider: string
  reranker_enabled: boolean
  sudachi_available: boolean
  qa_defaults: QaDefaults
}

// ---- POST /api/qa, /api/qa/stream ----
export interface SourceChunk {
  id: string | null
  source: string | null
  page: number | null
  text: string
  /** "figure" = 図チャンク（キャプション本文＋画像）。それ以外/undefinedは通常チャンク */
  type?: string | null
  /** 図画像のファイル名（/figures/<image_path> で配信） */
  image_path?: string | null
}

export interface Triple {
  start: string
  type: string
  end: string
  /** 実ノードラベル（QA参照グラフの色分け用。グラフ未使用時は null） */
  start_type?: string | null
  end_type?: string | null
}

// ---- GET /api/graph/node ----
export interface NodeInfo {
  id: string
  type: string
  degree: number
  properties: Record<string, unknown>
}

export interface GraphPath {
  path_text: string
  [key: string]: unknown
}

export interface QaEvidence {
  vector_sources: SourceChunk[]
  kg_source_chunks: SourceChunk[]
  graph_sources: Triple[]
  graph_paths: GraphPath[]
  extracted_entities: Record<string, unknown>
  kg_used: boolean
  kg_skip_reason: string | null
}

export interface QaResponse extends QaEvidence {
  answer: string
}

// SSE イベント（バックエンド services/qa.answer_question_events と対）
export type QaSseEvent =
  | { type: "meta"; data: { question: string; effective_config: Partial<QaDefaults> } }
  | { type: "retrieval"; data: QaEvidence }
  | { type: "token"; data: { delta: string } }
  | { type: "done"; data: { answer: string; timing_ms: { retrieval: number; generation: number } } }
  | { type: "error"; data: { stage: "retrieval" | "generation"; message: string } }

// ---- GET /api/graph/* ----
export interface EdgeRecord {
  source: string
  source_type: string
  relation: string
  target: string
  target_type: string
  source_degree: number
  target_degree: number
  source_docs: (string | null)[]
  target_docs: (string | null)[]
  edge_key?: number
}

export interface GraphStatus {
  graph: { exists: boolean; node_count: number; rel_count: number; error?: string | null }
  provenance: { status: "match" | "mismatch" | "unknown"; graph_collection: string | null }
  collection: string
}

// ---- GET /api/graph/schema ----
export interface SchemaInfo {
  domain: string
  version: string
  source: string | null
  node_types: string[]
  relations: string[]
}

export interface SchemaReport {
  /** グラフ構築時に刻印されたスキーマ（SchemaMeta）。未刻印なら null */
  active: (SchemaInfo & { stamped_at: string; kind: string }) | null
  /** SHARED_SCHEMA_PATH が指す、次回ビルドで使われるスキーマ */
  configured: SchemaInfo
  /** active.source と configured.source の一致。active 無しなら null */
  match: boolean | null
  /** builtin=同梱EDCを子プロセス実行（サーバ不要） / http=EDC_ENDPOINT指定時 */
  edc_mode: "builtin" | "http"
  edc_endpoint: string | null
}

// ---- GET/PUT /api/graph/schema/file ----
export interface SchemaFileResponse {
  path: string | null
  exists: boolean
  data: {
    domain?: string
    version?: string
    node_types: string[]
    node_type_definitions?: Record<string, string>
    relations: { name: string; description?: string }[]
  }
}

// ---- /api/dictionary（専門用語辞書＝名寄せ） ----
export interface DictEntry {
  canonical: string
  aliases: string[]
  category?: string
  definition?: string
}

export interface DictReportEntry extends DictEntry {
  matched_ids: string[]
  status: "merge_candidate" | "matched" | "unmatched"
}

export interface DictionaryReport {
  path: string
  exists: boolean
  entries: DictReportEntry[]
  counts: { merge_candidate: number; matched: number; unmatched: number }
}

// ---- /api/admin/collections（コレクション切替） ----
export interface CollectionsResponse {
  current: string
  graph_collection: string | null
  collections: { name: string; chunks: number }[]
}

// ---- GET /api/documents ----
export interface DocumentsSummary {
  collection: string
  total_chunks: number
  documents: { source: string; chunk_count: number; original_available?: boolean }[]
}

// ---- GET /api/documents/chunks ----
export interface DocumentChunksResponse {
  source: string
  total: number
  offset: number
  chunks: {
    id: string
    page: string | null
    text: string
    type?: string | null
    image_path?: string | null
  }[]
}
