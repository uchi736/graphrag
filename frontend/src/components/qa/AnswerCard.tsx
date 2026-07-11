import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Loader2 } from "lucide-react"
import type { QaStreamState } from "@/hooks/useQaStream"

export function AnswerCard({ state }: { state: QaStreamState }) {
  if (state.status === "idle") return null

  return (
    <div className="rounded-lg border bg-card p-5 shadow-sm">
      <div className="mb-2 flex items-center gap-2 text-sm font-bold text-primary">
        💡 回答
        {state.status === "retrieving" && (
          <span className="inline-flex items-center gap-1 text-xs font-normal text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" /> 検索中…
          </span>
        )}
        {state.status === "streaming" && state.answer === "" && (
          <span className="inline-flex items-center gap-1 text-xs font-normal text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" /> 回答生成中…
          </span>
        )}
      </div>

      {state.error ? (
        <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-destructive">{state.error}</p>
      ) : (
        <div className="prose prose-sm max-w-none text-[0.925rem] leading-relaxed [&_li]:my-0.5 [&_p]:my-1.5">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{state.answer}</ReactMarkdown>
          {state.status === "streaming" && state.answer !== "" && (
            <span className="ml-0.5 inline-block h-4 w-2 animate-pulse bg-primary/60 align-text-bottom" />
          )}
        </div>
      )}

      {state.timing && (
        <p className="mt-3 border-t pt-2 text-xs text-muted-foreground">
          検索 {(state.timing.retrieval / 1000).toFixed(1)}s · 生成 {(state.timing.generation / 1000).toFixed(1)}s
          {state.evidence &&
            ` · 参照 ${state.evidence.vector_sources.length}件 · KGチャンク ${state.evidence.kg_source_chunks.length}件` +
              (state.evidence.kg_used ? "" : " · KG未使用")}
        </p>
      )}
    </div>
  )
}
