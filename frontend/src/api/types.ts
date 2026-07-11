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
  search_mode: "hybrid" | "vector" | "keyword"
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
}

export interface Triple {
  start: string
  type: string
  end: string
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

// ---- GET /api/documents ----
export interface DocumentsSummary {
  collection: string
  total_chunks: number
  documents: { source: string; chunk_count: number }[]
}
