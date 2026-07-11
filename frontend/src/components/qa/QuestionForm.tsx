import { useState } from "react"
import { Send, Square } from "lucide-react"

export function QuestionForm({
  busy,
  onSubmit,
  onCancel,
}: {
  busy: boolean
  onSubmit: (question: string) => void
  onCancel: () => void
}) {
  const [question, setQuestion] = useState("")

  const submit = () => {
    const q = question.trim()
    if (q && !busy) onSubmit(q)
  }

  return (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault()
            submit()
          }
        }}
        rows={3}
        placeholder="質問を入力してください（Ctrl+Enter で送信）"
        className="w-full resize-y rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
      />
      <div className="mt-3 flex items-center gap-2">
        <button
          onClick={submit}
          disabled={busy || !question.trim()}
          className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
        >
          <Send className="h-4 w-4" />
          質問する
        </button>
        {busy && (
          <button
            onClick={onCancel}
            className="inline-flex items-center gap-2 rounded-md border px-4 py-2 text-sm font-medium text-muted-foreground hover:bg-muted"
          >
            <Square className="h-4 w-4" />
            キャンセル
          </button>
        )}
      </div>
    </div>
  )
}
