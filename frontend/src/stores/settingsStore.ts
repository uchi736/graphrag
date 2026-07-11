import { create } from "zustand"
import { persist } from "zustand/middleware"
import type { QaDefaults } from "@/api/types"

/**
 * 検索設定9キー + 表示設定（旧 st.session_state 相当、localStorage 永続）。
 * サーバ既定（GET /api/settings の qa_defaults）は「未変更キーの初期値」として
 * applyDefaults で一度だけマージする。
 */

export interface SettingsState extends QaDefaults {
  show_graph: boolean
  max_nodes: number
  _defaultsApplied: boolean
  set: <K extends keyof QaDefaults>(key: K, value: QaDefaults[K]) => void
  setView: (key: "show_graph" | "max_nodes", value: boolean | number) => void
  applyDefaults: (d: QaDefaults) => void
  resetToDefaults: (d: QaDefaults) => void
}

const initial: QaDefaults = {
  retrieval_top_k: 5,
  enable_rerank: true,
  enable_japanese_search: true,
  search_mode: "hybrid",
  enable_knowledge_graph: true,
  include_kg_source_chunks: true,
  graph_hop_count: 2,
  enable_entity_vector: true,
  entity_similarity_threshold: 0.85,
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      ...initial,
      show_graph: true,
      max_nodes: 200,
      _defaultsApplied: false,
      set: (key, value) => set({ [key]: value } as Partial<SettingsState>),
      setView: (key, value) => set({ [key]: value } as Partial<SettingsState>),
      applyDefaults: (d) =>
        set((state) => (state._defaultsApplied ? {} : { ...d, _defaultsApplied: true })),
      resetToDefaults: (d) => set({ ...d, _defaultsApplied: true }),
    }),
    {
      name: "graphrag.settings.v1",
      version: 1,
    },
  ),
)

/** QAリクエストボディの config 部を組み立てる（9キーのみ抽出） */
export function selectQaConfig(s: SettingsState): QaDefaults {
  return {
    retrieval_top_k: s.retrieval_top_k,
    enable_rerank: s.enable_rerank,
    enable_japanese_search: s.enable_japanese_search,
    search_mode: s.search_mode,
    enable_knowledge_graph: s.enable_knowledge_graph,
    include_kg_source_chunks: s.include_kg_source_chunks,
    graph_hop_count: s.graph_hop_count,
    enable_entity_vector: s.enable_entity_vector,
    entity_similarity_threshold: s.entity_similarity_threshold,
  }
}
