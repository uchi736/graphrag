import { create } from "zustand"
import { persist } from "zustand/middleware"
import type { QaDefaults } from "@/api/types"

/** QA履歴（question + answer + 実行時設定の要約に縮約して localStorage 永続、20件cap） */
export interface QaHistoryItem {
  question: string
  answer: string
  at: number
  config_summary: string
}

interface QaStore {
  history: QaHistoryItem[]
  push: (item: QaHistoryItem) => void
  clear: () => void
}

export const useQaStore = create<QaStore>()(
  persist(
    (set) => ({
      history: [],
      push: (item) =>
        set((s) => ({ history: [item, ...s.history].slice(0, 20) })),
      clear: () => set({ history: [] }),
    }),
    { name: "graphrag.qa-history.v1", version: 1 },
  ),
)

export function summarizeConfig(c: QaDefaults): string {
  const mode = { hybrid: "ハイブリッド", vector: "ベクトル", keyword: "キーワード", none: "グラフのみ" }[c.search_mode]
  return `${mode} · top_k=${c.retrieval_top_k} · KG ${c.enable_knowledge_graph ? "ON" : "OFF"} · リランク ${c.enable_rerank ? "ON" : "OFF"}`
}
