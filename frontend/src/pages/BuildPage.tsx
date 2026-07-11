import { useRef, useState } from "react"
import { toast } from "sonner"
import {
  FileUp,
  Hammer,
  Loader2,
  Play,
  RotateCcw,
  Square,
  Stamp,
  Trash2,
} from "lucide-react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { apiSend } from "@/api/client"
import { useGraphStatus } from "@/hooks/useGraphData"
import { useJobProgress } from "@/hooks/useJobProgress"
import { SchemaCard } from "@/components/schema/SchemaCard"
import { cn } from "@/lib/utils"

export default function BuildPage() {
  const [docs, setDocs] = useState<File[]>([])
  const [csv, setCsv] = useState<File | null>(null)
  const docInput = useRef<HTMLInputElement>(null)
  const csvInput = useRef<HTMLInputElement>(null)
  const { job, attach, cancel, busy } = useJobProgress()
  const { data: status } = useGraphStatus()
  const qc = useQueryClient()

  const provenance = useMutation({
    mutationFn: () => apiSend("POST", "/api/graph/provenance"),
    onSuccess: () => {
      toast.success("グラフの出自を記録しました")
      qc.invalidateQueries({ queryKey: ["health"] })
      qc.invalidateQueries({ queryKey: ["graph-status"] })
    },
    onError: (e) => toast.error(String(e)),
  })
  const clearGraph = useMutation({
    mutationFn: () => apiSend("POST", "/api/admin/clear-graph"),
    onSuccess: () => {
      toast.success("グラフをクリアしました")
      qc.invalidateQueries()
    },
    onError: (e) => toast.error(String(e)),
  })

  const submitBuild = async (mode: "new" | "resume" | "chunks_only") => {
    const fd = new FormData()
    for (const f of docs) fd.append("files", f)
    let path = "/api/build"
    if (mode === "chunks_only") {
      path = "/api/build/chunks-only"
    } else {
      if (csv) fd.append("csv_file", csv)
      fd.append("mode", mode)
    }
    const res = await fetch(path, { method: "POST", body: fd })
    if (!res.ok) {
      const err = await res.json().catch(() => null)
      const detail = err?.detail
      toast.error(typeof detail === "string" ? detail : (detail?.message ?? `${res.status}`))
      return
    }
    const { job_id } = await res.json()
    attach(job_id)
    toast.info("構築ジョブを開始しました")
  }

  const prov = status?.provenance
  const pct = job.progress?.percent

  return (
    <div className="space-y-4">
      {/* 既存グラフステータス */}
      {status && (
        <div className="flex flex-wrap items-center gap-3 rounded-lg border bg-card p-4 text-sm shadow-sm">
          <span>
            🕸️ 既存グラフ: <b className="font-mono">{status.graph.node_count.toLocaleString()}</b> ノード /{" "}
            <b className="font-mono">{status.graph.rel_count.toLocaleString()}</b> エッジ
          </span>
          {prov && prov.status !== "match" && status.graph.exists && (
            <span className="rounded bg-amber-50 px-2 py-0.5 text-xs text-amber-700">
              ⚠️ {prov.status === "mismatch" ? `別コレクション用（${prov.graph_collection}）` : "出自未記録"}
            </span>
          )}
          <div className="ml-auto flex gap-2">
            {prov && prov.status !== "match" && status.graph.exists && (
              <button
                onClick={() => provenance.mutate()}
                disabled={provenance.isPending}
                className="inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs hover:bg-muted"
              >
                <Stamp className="h-3.5 w-3.5" /> このグラフを現コレクション用として記録
              </button>
            )}
            {status.graph.exists && (
              <button
                onClick={() => {
                  if (window.confirm("Neo4j のグラフを全削除します。よろしいですか？")) clearGraph.mutate()
                }}
                disabled={clearGraph.isPending || busy}
                className="inline-flex items-center gap-1 rounded-md border border-destructive/40 px-2.5 py-1 text-xs text-destructive hover:bg-red-50"
              >
                <Trash2 className="h-3.5 w-3.5" /> グラフをクリア
              </button>
            )}
          </div>
        </div>
      )}

      {/* KGスキーマ（アクティブ vs 次回ビルド設定 + EDC同期） */}
      <SchemaCard attach={attach} busy={busy} />

      {/* アップロード */}
      <div className="grid gap-4 sm:grid-cols-2">
        <button
          onClick={() => docInput.current?.click()}
          className="rounded-lg border-2 border-dashed bg-card p-6 text-center text-sm text-muted-foreground hover:border-primary hover:text-primary"
        >
          <FileUp className="mx-auto mb-2 h-6 w-6" />
          {docs.length > 0
            ? `${docs.length} ファイル選択中（${docs.map((f) => f.name).slice(0, 3).join(", ")}${docs.length > 3 ? " …" : ""}）`
            : "ドキュメントを選択（PDF / txt / md、複数可）"}
          <input
            ref={docInput}
            type="file"
            multiple
            accept=".pdf,.txt,.md"
            className="hidden"
            onChange={(e) => setDocs([...(e.target.files ?? [])])}
          />
        </button>
        <button
          onClick={() => csvInput.current?.click()}
          className="rounded-lg border-2 border-dashed bg-card p-6 text-center text-sm text-muted-foreground hover:border-primary hover:text-primary"
        >
          <FileUp className="mx-auto mb-2 h-6 w-6" />
          {csv ? `edges.csv: ${csv.name}` : "エッジCSVを選択（source,target,label・任意）"}
          <input
            ref={csvInput}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => setCsv(e.target.files?.[0] ?? null)}
          />
        </button>
      </div>

      {/* 実行ボタン */}
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => submitBuild("new")}
          disabled={busy || (docs.length === 0 && !csv)}
          className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
        >
          <Hammer className="h-4 w-4" /> 新規構築
        </button>
        <button
          onClick={() => submitBuild("resume")}
          disabled={busy || (docs.length === 0 && !csv)}
          className="inline-flex items-center gap-2 rounded-md border px-4 py-2 text-sm font-medium hover:bg-muted disabled:opacity-50"
        >
          <Play className="h-4 w-4" /> 続きから再開
        </button>
        <button
          onClick={() => submitBuild("chunks_only")}
          disabled={busy || docs.length === 0}
          className="inline-flex items-center gap-2 rounded-md border px-4 py-2 text-sm font-medium hover:bg-muted disabled:opacity-50"
        >
          <RotateCcw className="h-4 w-4" /> チャンクのみ更新（高速）
        </button>
        {busy && (
          <button
            onClick={cancel}
            className="inline-flex items-center gap-2 rounded-md border border-destructive/40 px-4 py-2 text-sm font-medium text-destructive hover:bg-red-50"
          >
            <Square className="h-4 w-4" /> キャンセル
          </button>
        )}
      </div>

      {/* ジョブ進捗 */}
      {job.state !== "idle" && (
        <div className="rounded-lg border bg-card p-4 shadow-sm">
          <div className="flex items-center gap-2 text-sm font-medium">
            {job.state === "running" && <Loader2 className="h-4 w-4 animate-spin text-primary" />}
            {job.state === "running"
              ? (job.progress?.message ?? "実行中…")
              : job.state === "succeeded"
                ? "✅ 完了"
                : job.state === "cancelled"
                  ? "⏹️ キャンセル済み"
                  : `❌ 失敗: ${job.error ?? ""}`}
            {job.progress?.total != null && (
              <span className="ml-auto font-mono text-xs text-muted-foreground">
                {job.progress.current}/{job.progress.total}
                {job.progress.err > 0 && ` · エラー ${job.progress.err}`}
              </span>
            )}
          </div>
          <div className="mt-2 h-2 overflow-hidden rounded-full bg-muted">
            <div
              className={cn(
                "h-full rounded-full transition-all",
                job.state === "failed" ? "bg-destructive" : "brand-gradient",
                job.state === "running" && pct == null && "w-full animate-pulse",
              )}
              style={pct != null ? { width: `${pct}%` } : undefined}
            />
          </div>
          {job.warnings.length > 0 && (
            <details className="mt-2 text-xs text-amber-700">
              <summary className="cursor-pointer">警告 {job.warnings.length} 件</summary>
              <ul className="mt-1 list-disc space-y-0.5 pl-5">
                {job.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </details>
          )}
          {job.state === "succeeded" && job.result && (
            <p className="mt-2 text-xs text-muted-foreground">結果: {JSON.stringify(job.result)}</p>
          )}
        </div>
      )}
    </div>
  )
}
