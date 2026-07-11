import { Trash2 } from "lucide-react"
import { useQaStore } from "@/stores/qaStore"

export function QaHistoryList() {
  const history = useQaStore((s) => s.history)
  const clear = useQaStore((s) => s.clear)
  if (history.length === 0) return null

  return (
    <section className="mt-8">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-bold text-muted-foreground">🕘 質問履歴（{history.length}件）</h3>
        <button
          onClick={clear}
          className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-destructive"
        >
          <Trash2 className="h-3.5 w-3.5" />
          履歴をクリア
        </button>
      </div>
      <div className="space-y-2">
        {history.map((h) => (
          <details key={h.at} className="rounded-lg border bg-card shadow-sm">
            <summary className="cursor-pointer px-4 py-2.5 text-sm">
              {h.question.length > 60 ? h.question.slice(0, 60) + "…" : h.question}
              <span className="ml-2 text-xs text-muted-foreground">
                {new Date(h.at).toLocaleString("ja-JP")} · {h.config_summary}
              </span>
            </summary>
            <p className="whitespace-pre-wrap border-t px-4 py-3 text-sm leading-relaxed text-foreground/85">
              {h.answer}
            </p>
          </details>
        ))}
      </div>
    </section>
  )
}
