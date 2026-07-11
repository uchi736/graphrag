import { useState } from "react"
import { toast } from "sonner"
import { AlertTriangle, FileJson, Pencil, RefreshCw, Sparkles } from "lucide-react"
import { useQueryClient } from "@tanstack/react-query"
import { useSchemaReport } from "@/hooks/useGraphData"
import { SchemaEditorModal } from "@/components/schema/SchemaEditorModal"
import type { SchemaInfo } from "@/api/types"

function basename(p: string | null): string {
  if (!p) return "デフォルト（組み込み）"
  return p.split(/[\\/]/).pop() ?? p
}

function SchemaSummary({ label, info, stampedAt }: { label: string; info: SchemaInfo; stampedAt?: string }) {
  return (
    <div className="min-w-0 flex-1 rounded-md border bg-background p-3">
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      <p className="mt-1 truncate font-mono text-sm" title={info.source ?? undefined}>
        {basename(info.source)}
      </p>
      <p className="mt-0.5 text-xs text-muted-foreground">
        {info.domain} / {info.version} — タイプ {info.node_types.length}・関係 {info.relations.length}
        {stampedAt && ` · 刻印 ${stampedAt.slice(0, 10)}`}
      </p>
      <details className="mt-1.5">
        <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
          タイプ / 関係を表示
        </summary>
        <div className="mt-1.5 flex flex-wrap gap-1">
          {info.node_types.map((t) => (
            <span key={t} className="rounded-full bg-[var(--color-brand-from)]/10 px-2 py-0.5 text-xs text-primary">
              {t}
            </span>
          ))}
        </div>
        <div className="mt-1.5 flex flex-wrap gap-1">
          {info.relations.map((r) => (
            <span key={r} className="rounded-full bg-muted px-2 py-0.5 font-mono text-[10px]">
              {r}
            </span>
          ))}
        </div>
      </details>
    </div>
  )
}

/**
 * KGスキーマカード（構築タブ）。
 * - アクティブ（現グラフ構築時に刻印されたSchemaMeta）と次回ビルド設定
 *   （SHARED_SCHEMA_PATH）を並べ、不一致なら警告
 * - EDCスキーマ同期: 現コレクションの文書サンプル→EDC→スキーマJSON生成ジョブ。
 *   既定は同梱EDCの子プロセス実行（サーバ不要）、EDC_ENDPOINT指定時のみHTTP
 * - 編集: 発見スキーマの人手キュレーション（不要タイプ/関係の削除・追加）
 */
export function SchemaCard({
  attach,
  busy,
}: {
  attach: (jobId: string) => void
  busy: boolean
}) {
  const { data, isLoading, refetch } = useSchemaReport()
  const [submitting, setSubmitting] = useState(false)
  const [editorOpen, setEditorOpen] = useState(false)
  const qc = useQueryClient()

  const startEdcSync = async () => {
    setSubmitting(true)
    try {
      const res = await fetch("/api/build/edc-sync", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => null)
        const detail = err?.detail
        toast.error(typeof detail === "string" ? detail : (detail?.message ?? `${res.status}`))
        return
      }
      const { job_id } = await res.json()
      attach(job_id)
      toast.info("EDCスキーマ同期を開始しました（完了後「編集」でキュレーション→SHARED_SCHEMA_PATHに設定して再構築）")
      qc.invalidateQueries({ queryKey: ["graph-schema"] })
    } finally {
      setSubmitting(false)
    }
  }

  if (isLoading || !data) return null

  const edcModeLabel =
    data.edc_mode === "builtin"
      ? "内蔵EDCで実行（サーバ不要・vendor/EDC）"
      : `外部EDC: ${data.edc_endpoint}`

  return (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <FileJson className="h-4 w-4 text-primary" />
        <span className="text-sm font-medium">KGスキーマ</span>
        {data.match === false && (
          <span className="inline-flex items-center gap-1 rounded bg-amber-50 px-2 py-0.5 text-xs text-amber-700">
            <AlertTriangle className="h-3 w-3" />
            現グラフと次回ビルドのスキーマが異なります — このまま再構築するとスキーマが変わります
          </span>
        )}
        <div className="ml-auto flex gap-2">
          <button
            onClick={() => refetch()}
            className="inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs hover:bg-muted"
            title="再読み込み"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={() => setEditorOpen(true)}
            className="inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs hover:bg-muted"
            title="次回ビルド用スキーマを人手キュレーション（不要なタイプ/関係の削除・追加）"
          >
            <Pencil className="h-3.5 w-3.5" /> 編集
          </button>
          <button
            onClick={startEdcSync}
            disabled={busy || submitting}
            className="inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs hover:bg-muted disabled:opacity-50"
            title={`現コレクションの文書サンプルからスキーマを自動発見 — ${edcModeLabel}`}
          >
            <Sparkles className="h-3.5 w-3.5" /> EDCスキーマ同期
          </button>
        </div>
      </div>
      <div className="flex flex-col gap-3 sm:flex-row">
        {data.active ? (
          <SchemaSummary label="アクティブ（現グラフ構築時）" info={data.active} stampedAt={data.active.stamped_at} />
        ) : (
          <div className="min-w-0 flex-1 rounded-md border bg-background p-3 text-xs text-muted-foreground">
            アクティブスキーマ未刻印（このグラフはスキーマ刻印前に構築されたか、グラフが空です）
          </div>
        )}
        <SchemaSummary label="次回ビルド設定（SHARED_SCHEMA_PATH）" info={data.configured} />
      </div>
      <p className="mt-2 text-right text-[11px] text-muted-foreground">同期モード: {edcModeLabel}</p>

      <SchemaEditorModal open={editorOpen} onClose={() => setEditorOpen(false)} />
    </div>
  )
}
