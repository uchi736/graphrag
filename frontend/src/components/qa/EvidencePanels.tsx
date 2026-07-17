import { lazy, memo, Suspense, useMemo, useState, type ReactNode } from "react"
import { BookOpen } from "lucide-react"
import type { EdgeRecord, QaEvidence, SourceChunk } from "@/api/types"
import { ChunkBrowserModal } from "@/components/documents/ChunkBrowserModal"
import { GraphLegend } from "@/components/graph/GraphLegend"
import { toGraphData } from "@/lib/graphTransform"

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
      source_type: t.start_type ?? "Unknown",
      relation: t.type,
      target: t.end,
      target_type: t.end_type ?? "Unknown",
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
  emptyNote,
  children,
}: {
  title: string
  count: number
  defaultOpen?: boolean
  /** count=0 のとき、パネルを消す代わりにこの説明文を表示する（機能の存在を可視化） */
  emptyNote?: string
  children: ReactNode
}) {
  if (count === 0 && !emptyNote) return null
  return (
    <details className="rounded-lg border bg-card shadow-sm" open={defaultOpen}>
      <summary className="cursor-pointer px-4 py-2.5 text-sm font-medium">
        {title} <span className="ml-1 text-xs text-muted-foreground">({count})</span>
      </summary>
      <div className="space-y-2 border-t px-4 py-3">
        {count === 0 ? (
          <p className="py-2 text-xs text-muted-foreground">{emptyNote}</p>
        ) : (
          children
        )}
      </div>
    </details>
  )
}

function ChunkCard({
  chunk,
  onOpenDoc,
}: {
  chunk: SourceChunk
  onOpenDoc?: (source: string, chunkId: string | null) => void
}) {
  const isFigure = chunk.type === "figure" && chunk.image_path
  // 出典クリック→原本を開く（/originals 配信。pageがあればPDFビューアのページアンカー付き）
  const originalUrl = chunk.source
    ? `/originals/${encodeURIComponent(chunk.source)}` +
      (chunk.page != null ? `#page=${Number(chunk.page) + 1}` : "")
    : null
  return (
    <div className="rounded-md border bg-background p-3">
      <div className="mb-1 flex items-center gap-2">
        {originalUrl ? (
          <a
            href={originalUrl}
            target="_blank"
            rel="noreferrer"
            title="原本ドキュメントを開く"
            className="inline-block rounded bg-[var(--color-brand-from)]/10 px-2 py-0.5 font-mono text-xs text-primary underline-offset-2 hover:underline"
          >
            {chunk.source}
            {chunk.page != null && ` · p.${chunk.page}`} ↗
          </a>
        ) : (
          <span className="inline-block rounded bg-[var(--color-brand-from)]/10 px-2 py-0.5 font-mono text-xs text-primary">
            不明
          </span>
        )}
        {isFigure && (
          <span className="rounded bg-amber-50 px-1.5 py-0.5 text-[10px] font-medium text-amber-700">
            🖼 図
          </span>
        )}
        {chunk.source && onOpenDoc && (
          <button
            onClick={() => onOpenDoc(chunk.source!, chunk.id)}
            className="ml-auto inline-flex items-center gap-1 rounded border px-2 py-0.5 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
            title="この文書の全チャンクを開き、該当チャンクにジャンプ"
          >
            <BookOpen className="h-3 w-3" />
            文書内で見る
          </button>
        )}
      </div>
      {isFigure && (
        <a href={`/figures/${chunk.image_path}`} target="_blank" rel="noreferrer" title="クリックで原寸表示">
          <img
            src={`/figures/${chunk.image_path}`}
            alt={chunk.text}
            loading="lazy"
            className="mb-2 max-h-64 rounded border bg-white object-contain"
          />
        </a>
      )}
      <p className="max-h-40 overflow-y-auto whitespace-pre-wrap text-xs leading-relaxed text-foreground/85">
        {chunk.text}
      </p>
    </div>
  )
}

function EvidencePanelsInner({ evidence }: { evidence: QaEvidence | null }) {
  const [openDoc, setOpenDoc] = useState<{ source: string; chunkId: string | null } | null>(null)
  // 参照が変わらない限り同一配列を渡す。毎レンダーで新配列を作ると
  // force-graph が graphData 変更とみなし物理シミュレーションを再加熱してしまう
  // （トークンストリーミング中ずっとグラフが揺れ続ける原因）。
  const graphEdges = useMemo(() => (evidence ? triplesToEdges(evidence) : []), [evidence])
  const legendNodes = useMemo(() => toGraphData(graphEdges).nodes, [graphEdges])
  if (!evidence) return null
  const entities = evidence.extracted_entities ?? {}
  const merged = (entities as { merged_entities?: string[] }).merged_entities ?? []
  const themes = (entities as { theme_keywords?: string[] }).theme_keywords ?? []
  const fallbackSeeds =
    (entities as { fallback_seed_entities?: string[] }).fallback_seed_entities ?? []
  const openChunk = (source: string, chunkId: string | null) => setOpenDoc({ source, chunkId })

  const graphEmptyNote = evidence.kg_used
    ? "この質問ではグラフから関係（トリプル）が取得されませんでした。質問に対応するエンティティがグラフに無いか、パスが見つからなかったため、ベクトル検索のみで回答しています。"
    : `ナレッジグラフは使用されませんでした${evidence.kg_skip_reason ? `（理由: ${evidence.kg_skip_reason}）` : "（設定でOFF、またはグラフ未構築）"}。`

  return (
    <div className="space-y-3">
      <Panel title="📚 参照ドキュメント" count={evidence.vector_sources.length} defaultOpen>
        {evidence.vector_sources.map((c, i) => (
          <ChunkCard key={c.id ?? i} chunk={c} onOpenDoc={openChunk} />
        ))}
      </Panel>

      <Panel
        title="🕸️ 参照グラフ（回答に使われた関係）"
        count={evidence.graph_sources.length}
        defaultOpen={evidence.graph_sources.length > 0}
        emptyNote={graphEmptyNote}
      >
        <GraphLegend nodes={legendNodes} />
        <Suspense
          fallback={
            <div className="flex h-40 items-center justify-center text-xs text-muted-foreground">
              グラフを読み込み中…
            </div>
          }
        >
          <GraphCanvas edges={graphEdges} height={380} showEdgeLabels />
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
          <ChunkCard key={c.id ?? i} chunk={c} onOpenDoc={openChunk} />
        ))}
      </Panel>

      <Panel title="🧩 抽出キーワード（デバッグ）" count={merged.length + themes.length}>
        <div className="flex flex-wrap gap-1.5">
          {merged.map((e) => (
            <span key={e} className="rounded-full bg-muted px-2.5 py-0.5 text-xs" title="低レベル（固有名・search_keys照合）">
              {e}
            </span>
          ))}
          {themes.map((t) => (
            <span
              key={t}
              className="rounded-full bg-[var(--color-brand-from)]/10 px-2.5 py-0.5 text-xs text-primary"
              title="高レベル（テーマ語・関係キーワード索引照合）"
            >
              🏷 {t}
            </span>
          ))}
          {fallbackSeeds.map((s) => (
            <span
              key={s}
              className="rounded-full border border-amber-300 bg-amber-50 px-2.5 py-0.5 text-xs text-amber-700"
              title="フォールバック（質問全文→エンティティ埋め込みで採用した起点）"
            >
              ⚓ {s}
            </span>
          ))}
        </div>
      </Panel>

      {openDoc && (
        <ChunkBrowserModal
          source={openDoc.source}
          focusId={openDoc.chunkId}
          onClose={() => setOpenDoc(null)}
        />
      )}
    </div>
  )
}

// トークンイベント毎の QaPage 再レンダリングから根拠パネル一式を遮断する
// （evidence の参照はストリーミング中不変なので memo が効く）。
export const EvidencePanels = memo(EvidencePanelsInner)
