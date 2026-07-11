import { useQuery } from "@tanstack/react-query"
import { apiGet } from "@/api/client"
import type { DocumentsSummary, EdgeRecord, GraphStatus, SchemaReport } from "@/api/types"

/** 全体エッジ。staleTime: Infinity + 手動 refetch（現行の graph_data_cache と同じモデル） */
export function useGraphOverview(limit: number, enabled: boolean) {
  return useQuery({
    queryKey: ["graph-overview", limit],
    queryFn: () => apiGet<EdgeRecord[]>("/api/graph/overview", { limit }),
    staleTime: Infinity,
    enabled,
  })
}

export function useSubgraph(centers: string[], hop: number, enabled: boolean) {
  return useQuery({
    queryKey: ["graph-subgraph", centers.join("|"), hop],
    queryFn: async () => {
      const params = new URLSearchParams()
      for (const c of centers) params.append("center", c)
      params.set("hop", String(hop))
      params.set("limit", "500")
      const res = await fetch(`/api/graph/subgraph?${params.toString()}`)
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      return (await res.json()) as EdgeRecord[]
    },
    staleTime: Infinity,
    enabled: enabled && centers.length > 0,
  })
}

export function useNodeSearch(q: string) {
  return useQuery({
    queryKey: ["node-search", q],
    queryFn: () => apiGet<{ ids: string[] }>("/api/graph/node-ids", { q, limit: 50 }),
    staleTime: 60_000,
    enabled: q.length > 0,
  })
}

export function useGraphStatus() {
  return useQuery({
    queryKey: ["graph-status"],
    queryFn: () => apiGet<GraphStatus>("/api/graph/status"),
    staleTime: 30_000,
  })
}

export function useSchemaReport() {
  return useQuery({
    queryKey: ["graph-schema"],
    queryFn: () => apiGet<SchemaReport>("/api/graph/schema"),
    staleTime: 60_000,
  })
}

export function useDocuments() {
  return useQuery({
    queryKey: ["documents"],
    queryFn: () => apiGet<DocumentsSummary>("/api/documents"),
    staleTime: 60_000,
  })
}

/**
 * 文書チャンク一覧。offset=null かつ focus 指定時はサーバが該当チャンクの
 * ページに offset を自動調整して返す（QA根拠→「文書内で見る」用）。
 */
export function useDocumentChunks(source: string | null, offset: number | null, focus?: string | null) {
  return useQuery({
    queryKey: ["document-chunks", source, offset, focus ?? ""],
    queryFn: () =>
      apiGet<import("@/api/types").DocumentChunksResponse>("/api/documents/chunks", {
        source: source ?? "",
        limit: 50,
        offset: offset ?? undefined,
        focus: offset === null && focus ? focus : undefined,
      }),
    enabled: source !== null,
    staleTime: 60_000,
  })
}
