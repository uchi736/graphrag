import { useRef, useState } from "react"
import { Loader2, RefreshCw, Square, Trash2, Upload } from "lucide-react"
import { toast } from "sonner"
import { useQueryClient } from "@tanstack/react-query"
import { apiSend } from "@/api/client"
import { useDocuments } from "@/hooks/useGraphData"
import { useJobProgress } from "@/hooks/useJobProgress"
import { ChunkBrowserModal } from "@/components/documents/ChunkBrowserModal"

export default function DocumentsPage() {
  const { data, isLoading, isError, error, refetch, isFetching } = useDocuments()
  const [selected, setSelected] = useState<string | null>(null)
  const { job, attach, cancel, busy } = useJobProgress()
  const qc = useQueryClient()

  // 増分更新: 行の「更新」→ ファイル選択 → POST /api/documents/{id}/update ジョブ
  const fileInput = useRef<HTMLInputElement>(null)
  const pendingDoc = useRef<string | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)

  const pickUpdateFile = (source: string) => {
    pendingDoc.current = source
    fileInput.current?.click()
  }

  const submitUpdate = async (file: File) => {
    const doc = pendingDoc.current
    if (!doc) return
    const fd = new FormData()
    fd.append("file", file)
    const res = await fetch(`/api/documents/${encodeURIComponent(doc)}/update`, {
      method: "POST",
      body: fd,
    })
    if (!res.ok) {
      const err = await res.json().catch(() => null)
      const detail = err?.detail
      toast.error(typeof detail === "string" ? detail : (detail?.message ?? `${res.status}`))
      return
    }
    const { job_id } = await res.json()
    attach(job_id)
    toast.info(`増分更新を開始: ${doc}（差分チャンクのみ再抽出）`)
  }

  const deleteDoc = async (source: string) => {
    if (!window.confirm(`「${source}」をグラフ・ベクトルDBから完全削除します。よろしいですか？`)) return
    setDeleting(source)
    try {
      const r = await apiSend<{ deleted_chunks?: number }>(
        "DELETE", `/api/documents/${encodeURIComponent(source)}`)
      toast.success(`削除しました: ${source}${r.deleted_chunks != null ? `（${r.deleted_chunks}チャンク）` : ""}`)
      qc.invalidateQueries({ queryKey: ["documents"] })
      qc.invalidateQueries({ queryKey: ["graph-status"] })
      qc.invalidateQueries({ queryKey: ["graph-overview"] })
    } catch (e) {
      toast.error(`削除失敗: ${e instanceof Error ? e.message : e}`)
    } finally {
      setDeleting(null)
    }
  }

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
        読み込み中…
      </div>
    )
  }
  if (isError) {
    return (
      <p className="rounded-md bg-red-50 px-4 py-3 text-sm text-destructive">
        取得エラー: {error instanceof Error ? error.message : String(error)}
      </p>
    )
  }
  if (!data || data.documents.length === 0) {
    return (
      <div className="rounded-lg border bg-card p-10 text-center text-sm text-muted-foreground">
        登録されたドキュメントはありません（「🛠️ 構築 / 取り込み」から追加してください）
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <div className="rounded-lg border bg-card px-5 py-3 shadow-sm">
          <p className="text-xs text-muted-foreground">総チャンク数</p>
          <p className="font-mono text-2xl font-bold">{data.total_chunks.toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-card px-5 py-3 shadow-sm">
          <p className="text-xs text-muted-foreground">ドキュメント数</p>
          <p className="font-mono text-2xl font-bold">{data.documents.length}</p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="ml-auto inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs hover:bg-muted disabled:opacity-50"
        >
          <RefreshCw className={isFetching ? "h-3.5 w-3.5 animate-spin" : "h-3.5 w-3.5"} />
          再取得
        </button>
      </div>

      {/* 増分更新ジョブの進捗 */}
      {job.state !== "idle" && (
        <div className="rounded-lg border bg-card p-3 text-sm shadow-sm">
          <div className="flex items-center gap-2">
            {job.state === "running" && <Loader2 className="h-4 w-4 animate-spin text-primary" />}
            {job.state === "running"
              ? (job.progress?.message ?? "増分更新 実行中…")
              : job.state === "succeeded"
                ? "✅ 増分更新 完了"
                : job.state === "cancelled"
                  ? "⏹️ キャンセル済み"
                  : `❌ 失敗: ${job.error ?? ""}`}
            {job.progress?.current != null && (
              <span className="ml-auto font-mono text-xs text-muted-foreground">
                {job.progress.current}
                {job.progress.total != null && `/${job.progress.total}`}
              </span>
            )}
            {busy && (
              <button
                onClick={cancel}
                className="ml-2 inline-flex items-center gap-1 rounded-md border border-destructive/40 px-2 py-0.5 text-xs text-destructive hover:bg-red-50"
              >
                <Square className="h-3 w-3" /> 中止
              </button>
            )}
          </div>
          {job.state === "succeeded" && job.result && (
            <p className="mt-1 text-xs text-muted-foreground">結果: {JSON.stringify(job.result)}</p>
          )}
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border bg-card shadow-sm">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/60 text-left text-xs text-muted-foreground">
              <th className="px-4 py-2.5 font-medium">ソースファイル（クリックでチャンクを表示）</th>
              <th className="px-4 py-2.5 text-right font-medium">チャンク数</th>
              <th className="px-4 py-2.5 text-right font-medium">操作</th>
            </tr>
          </thead>
          <tbody>
            {data.documents.map((d) => (
              <tr
                key={d.source}
                onClick={() => setSelected(d.source)}
                className="cursor-pointer border-b last:border-0 hover:bg-muted/40"
              >
                <td className="px-4 py-2 font-mono text-xs text-primary underline-offset-2 hover:underline">
                  {d.source}
                </td>
                <td className="px-4 py-2 text-right font-mono">{d.chunk_count.toLocaleString()}</td>
                <td className="px-4 py-2 text-right">
                  <div className="inline-flex gap-1.5" onClick={(e) => e.stopPropagation()}>
                    <button
                      onClick={() => pickUpdateFile(d.source)}
                      disabled={busy}
                      className="inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs hover:bg-muted disabled:opacity-40"
                      title="改訂版ファイルを選んで増分更新（内容が変わったチャンクだけ再抽出）"
                    >
                      <Upload className="h-3 w-3" /> 更新
                    </button>
                    <button
                      onClick={() => deleteDoc(d.source)}
                      disabled={busy || deleting === d.source}
                      className="inline-flex items-center gap-1 rounded-md border border-destructive/40 px-2 py-0.5 text-xs text-destructive hover:bg-red-50 disabled:opacity-40"
                      title="この文書をグラフ・ベクトルDBから完全削除"
                    >
                      <Trash2 className="h-3 w-3" /> 削除
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <input
        ref={fileInput}
        type="file"
        accept=".pdf,.txt,.md"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) submitUpdate(f)
          e.target.value = ""
        }}
      />

      <ChunkBrowserModal source={selected} onClose={() => setSelected(null)} />
    </div>
  )
}
