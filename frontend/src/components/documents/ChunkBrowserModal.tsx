import { useState, useEffect, useRef } from "react"
import { ChevronLeft, ChevronRight, RefreshCw, X } from "lucide-react"
import { useDocumentChunks } from "@/hooks/useGraphData"

/**
 * 文書のチャンク本文ブラウザ（50件ページング）。
 * focusId を渡すと、そのチャンクを含むページを開いてハイライト＋自動スクロール
 * （QAタブの参照ドキュメント→「文書内で見る」から使う）。
 */
export function ChunkBrowserModal({
  source,
  focusId,
  onClose,
}: {
  source: string | null
  focusId?: string | null
  onClose: () => void
}) {
  // null = 「focusId のページをサーバに決めさせる」初期状態
  const [offset, setOffset] = useState<number | null>(focusId ? null : 0)
  const { data, isLoading } = useDocumentChunks(source, offset, focusId)
  const focusRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => setOffset(focusId ? null : 0), [source, focusId])
  useEffect(() => {
    if (data && focusRef.current) focusRef.current.scrollIntoView({ block: "center" })
  }, [data])

  if (!source) return null
  const total = data?.total ?? 0
  const effOffset = data?.offset ?? 0
  const pageEnd = Math.min(effOffset + 50, total)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
      <div
        className="flex max-h-[85vh] w-full max-w-3xl flex-col rounded-lg border bg-card shadow-xl"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <div className="flex items-center justify-between border-b px-5 py-3">
          <div>
            <h3 className="font-mono text-sm font-bold">{source}</h3>
            <p className="text-xs text-muted-foreground">
              {total > 0 ? `チャンク ${effOffset + 1}–${pageEnd} / ${total}` : "チャンク"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setOffset(Math.max(0, effOffset - 50))}
              disabled={effOffset === 0}
              className="rounded-md border p-1.5 hover:bg-muted disabled:opacity-40"
              title="前の50件"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button
              onClick={() => setOffset(effOffset + 50 < total ? effOffset + 50 : effOffset)}
              disabled={pageEnd >= total}
              className="rounded-md border p-1.5 hover:bg-muted disabled:opacity-40"
              title="次の50件"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
            <button onClick={onClose} className="ml-2 text-muted-foreground hover:text-foreground">
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        <div className="flex-1 space-y-3 overflow-y-auto p-5">
          {isLoading ? (
            <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              読み込み中…
            </div>
          ) : (
            data?.chunks.map((c, i) => {
              const isFocus = focusId != null && c.id === focusId
              return (
                <div
                  key={c.id}
                  ref={isFocus ? focusRef : undefined}
                  className={
                    "rounded-md border bg-background p-3" +
                    (isFocus ? " border-primary ring-2 ring-primary/30" : "")
                  }
                >
                  <div className="mb-1.5 flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="rounded bg-muted px-1.5 py-0.5 font-mono">#{effOffset + i + 1}</span>
                    {c.page != null && c.page !== "" && <span>p.{c.page}</span>}
                    {isFocus && (
                      <span className="rounded bg-primary/10 px-1.5 py-0.5 font-medium text-primary">
                        回答が参照したチャンク
                      </span>
                    )}
                    <span className="ml-auto font-mono opacity-60">{c.id.slice(0, 12)}…</span>
                  </div>
                  <p className="whitespace-pre-wrap text-xs leading-relaxed text-foreground/90">{c.text}</p>
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}
