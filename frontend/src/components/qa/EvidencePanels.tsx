import { lazy, Suspense, type ReactNode } from "react"
import type { EdgeRecord, QaEvidence, SourceChunk } from "@/api/types"

// force-graph はサイズが大きいので遅延ロード（QA初期バンドルから隔離）
const GraphCanvas = lazy(() =>
  import("@/components/graph/GraphCanvas").then((m) => ({ default: m.GraphCanvas })),
)

/** QAで実際に使われたトリプルを EdgeRecord に変換して可視化用に渡す */
function triplesToEdges(evidence: QaEvidence): EdgeRecord[] {
  return evidence.graph_sources
    .filter((t) => t.start && t.end)
    .map((t) => ({
      source: t.start,
      source_type: "Unknown",
      relation: t.type,
      target: t.end,
      target_type: "Unknown",
      source_degree: 0,
      target_degree: 0,
      source_docs: [],
      target_docs: [],
    }))
}

function Panel({
  title,
  count,
  defaultOpen,
  children,
}: {
  title: string
  count: number
  defaultOpen?: boolean
  children: ReactNode
}) {
  if (count === 0) return null
  return (
    <details className="rounded-lg border bg-card shadow-sm" open={defaultOpen}>
      <summary className="cursor-pointer px-4 py-2.5 text-sm font-medium">
        {title} <span className="ml-1 text-xs text-muted-foreground">({count})</span>
      </summary>
      <div className="space-y-2 border-t px-4 py-3">{children}</div>
    </details>
  )
}

function ChunkCard({ chunk }: { chunk: SourceChunk }) {
  return (
    <div className="rounded-md border bg-background p-3">
      <span className="mb-1 inline-block rounded bg-[var(--color-brand-from)]/10 px-2 py-0.5 font-mono text-xs text-primary">
        {chunk.source ?? "不明"}
        {chunk.page != null && ` · p.${chunk.page}`}
      </span>
      <p className="max-h-40 overflow-y-auto whitespace-pre-wrap text-xs leading-relaxed text-foreground/85">
        {chunk.text}
      </p>
    </div>
  )
}

export function EvidencePanels({ evidence }: { evidence: QaEvidence | null }) {
  if (!evidence) return null
  const entities = evidence.extracted_entities ?? {}
  const merged = (entities as { merged_entities?: string[] }).merged_entities ?? []

  return (
    <div className="space-y-3">
      <Panel title="📚 参照ドキュメント" count={evidence.vector_sources.length} defaultOpen>
        {evidence.vector_sources.map((c, i) => (
          <ChunkCard key={c.id ?? i} chunk={c} />
        ))}
      </Panel>

      <Panel title="🕸️ 参照グラフ（回答に使われた関係）" count={evidence.graph_sources.length} defaultOpen={evidence.kg_used}>
        <Suspense
          fallback={
            <div className="flex h-40 items-center justify-center text-xs text-muted-foreground">
              グラフを読み込み中…
            </div>
          }
        >
          <GraphCanvas edges={triplesToEdges(evidence)} height={380} />
        </Suspense>
        <p className="text-right text-xs text-muted-foreground">
          {evidence.graph_sources.length} 関係 — エッジにカーソルを乗せると関係名を表示
        </p>
      </Panel>

      <Panel title="🧭 グラフ推論パス" count={evidence.graph_paths.length}>
        <ul className="space-y-1 font-mono text-xs text-foreground/85">
          {evidence.graph_paths.map((p, i) => (
            <li key={i} className="rounded bg-muted px-2 py-1">
              {p.path_text}
            </li>
          ))}
        </ul>
      </Panel>

      <Panel title="🔗 KGソースチャンク" count={evidence.kg_source_chunks.length}>
        {evidence.kg_source_chunks.map((c, i) => (
          <ChunkCard key={c.id ?? i} chunk={c} />
        ))}
      </Panel>

      <Panel title="🧩 抽出エンティティ（デバッグ）" count={merged.length}>
        <div className="flex flex-wrap gap-1.5">
          {merged.map((e) => (
            <span key={e} className="rounded-full bg-muted px-2.5 py-0.5 text-xs">
              {e}
            </span>
          ))}
        </div>
      </Panel>
    </div>
  )
}
