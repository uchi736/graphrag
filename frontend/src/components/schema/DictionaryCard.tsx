import { useEffect, useState } from "react"
import { toast } from "sonner"
import { BookMarked, Merge, Pencil, Plus, RefreshCw, Save, Trash2, X } from "lucide-react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { apiGet, apiSend } from "@/api/client"
import type { DictEntry, DictionaryReport, DictReportEntry } from "@/api/types"

function useDictionary() {
  return useQuery({
    queryKey: ["dictionary"],
    queryFn: () => apiGet<DictionaryReport>("/api/dictionary"),
    staleTime: 30_000,
  })
}

const STATUS_BADGE: Record<DictReportEntry["status"], { label: string; cls: string }> = {
  merge_candidate: { label: "統合候補", cls: "bg-amber-50 text-amber-700" },
  matched: { label: "一致", cls: "bg-green-50 text-green-700" },
  unmatched: { label: "未マッチ", cls: "bg-muted text-muted-foreground" },
}

interface Row extends DictEntry {
  aliasesText: string
  status?: DictReportEntry["status"]
  matched_ids?: string[]
}

function EditorModal({ report, onClose }: { report: DictionaryReport; onClose: () => void }) {
  const qc = useQueryClient()
  const [rows, setRows] = useState<Row[]>([])
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    setRows(report.entries.map((e) => ({ ...e, aliasesText: e.aliases.join(" | ") })))
  }, [report])

  const addRow = () => setRows((rs) => [...rs, { canonical: "", aliases: [], aliasesText: "" }])
  const update = (i: number, patch: Partial<Row>) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)))
  const remove = (i: number) => setRows((rs) => rs.filter((_, j) => j !== i))

  const save = async () => {
    const entries = rows
      .map((r) => ({
        canonical: r.canonical.trim(),
        aliases: r.aliasesText.split("|").map((a) => a.trim()).filter(Boolean),
        category: r.category ?? "",
        definition: r.definition ?? "",
      }))
      .filter((e) => e.canonical)
    if (entries.length === 0) {
      toast.error("canonical（正式名）を1件以上入力してください")
      return
    }
    setSaving(true)
    try {
      const res = await apiSend<{ path: string; n_entries: number }>("PUT", "/api/dictionary", { entries })
      toast.success(`辞書を保存しました（${res.n_entries}エントリ）`)
      qc.invalidateQueries({ queryKey: ["dictionary"] })
      onClose()
    } catch (e) {
      toast.error(`保存失敗: ${e instanceof Error ? e.message : e}`)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
      <div
        className="flex max-h-[85vh] w-full max-w-4xl flex-col rounded-lg border bg-card shadow-xl"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <div className="flex items-center justify-between border-b px-5 py-3">
          <div>
            <h3 className="text-sm font-bold">専門用語辞書の編集（名寄せ）</h3>
            <p className="text-xs text-muted-foreground">
              正式名（canonical）と別名（| 区切り）を登録。「適用」で別名ノードを正式名ノードへ統合します。
              保存先: {report.path}
            </p>
          </div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-left text-muted-foreground">
                <th className="pb-1.5 pr-2 font-medium">正式名（canonical）</th>
                <th className="pb-1.5 pr-2 font-medium">別名（| 区切り）</th>
                <th className="pb-1.5 pr-2 font-medium">定義（任意）</th>
                <th className="pb-1.5 pr-2 font-medium">状況</th>
                <th className="pb-1.5" />
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className="border-b last:border-0">
                  <td className="py-1.5 pr-2">
                    <input
                      value={r.canonical}
                      onChange={(e) => update(i, { canonical: e.target.value })}
                      placeholder="労働基準法"
                      className="w-full rounded-md border bg-background px-2 py-1 font-mono"
                    />
                  </td>
                  <td className="py-1.5 pr-2">
                    <input
                      value={r.aliasesText}
                      onChange={(e) => update(i, { aliasesText: e.target.value })}
                      placeholder="労基法 | 労働基準法(昭和22年)"
                      className="w-full rounded-md border bg-background px-2 py-1 font-mono"
                    />
                  </td>
                  <td className="py-1.5 pr-2">
                    <input
                      value={r.definition ?? ""}
                      onChange={(e) => update(i, { definition: e.target.value })}
                      className="w-full rounded-md border bg-background px-2 py-1"
                    />
                  </td>
                  <td className="py-1.5 pr-2">
                    {r.status && (
                      <span
                        className={`whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] ${STATUS_BADGE[r.status].cls}`}
                        title={r.matched_ids?.join(", ")}
                      >
                        {STATUS_BADGE[r.status].label}
                        {r.matched_ids && r.matched_ids.length > 0 && ` (${r.matched_ids.length})`}
                      </span>
                    )}
                  </td>
                  <td className="py-1.5 text-right">
                    <button onClick={() => remove(i)} className="text-muted-foreground hover:text-destructive">
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <button
            onClick={addRow}
            className="mt-3 inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs hover:bg-muted"
          >
            <Plus className="h-3.5 w-3.5" /> 行を追加
          </button>
        </div>

        <div className="flex items-center justify-end gap-2 border-t px-5 py-3">
          <button onClick={onClose} className="rounded-md border px-3 py-1.5 text-sm hover:bg-muted">
            キャンセル
          </button>
          <button
            onClick={save}
            disabled={saving}
            className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
          >
            <Save className="h-4 w-4" /> 保存
          </button>
        </div>
      </div>
    </div>
  )
}

/**
 * 専門用語辞書カード（構築タブ）— 名寄せ用途。
 * 辞書を編集 → 「名寄せ実行」で別名ノードを正式名ノードへ統合するジョブ
 * （エッジ付け替え → プロパティ付与 → search_keys 再計算。LLM不要）。
 */
export function DictionaryCard({
  attach,
  busy,
}: {
  attach: (jobId: string) => void
  busy: boolean
}) {
  const { data, isLoading, refetch } = useDictionary()
  const [editorOpen, setEditorOpen] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const qc = useQueryClient()

  const apply = async () => {
    const n = data?.counts.merge_candidate ?? 0
    if (
      !window.confirm(
        `辞書を適用します。統合候補 ${n} 件の別名ノードが正式名ノードへマージされます（破壊的・エッジ付け替え）。よろしいですか？`,
      )
    )
      return
    setSubmitting(true)
    try {
      const res = await apiSend<{ job_id: string }>("POST", "/api/dictionary/apply", { merge: true })
      attach(res.job_id)
      toast.info("辞書適用（名寄せ）を開始しました")
      qc.invalidateQueries({ queryKey: ["dictionary"] })
    } catch (e) {
      toast.error(String(e))
    } finally {
      setSubmitting(false)
    }
  }

  if (isLoading || !data) return null
  const c = data.counts

  return (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      <div className="flex flex-wrap items-center gap-2">
        <BookMarked className="h-4 w-4 text-primary" />
        <span className="text-sm font-medium">専門用語辞書（名寄せ）</span>
        <span className="text-xs text-muted-foreground">
          {data.entries.length === 0 ? (
            "未登録 — 「編集」から正式名と別名を登録すると、分裂ノードを統合できます"
          ) : (
            <>
              {data.entries.length} エントリ · 統合候補{" "}
              <b className={c.merge_candidate > 0 ? "text-amber-600" : ""}>{c.merge_candidate}</b> / 一致 {c.matched} /
              未マッチ {c.unmatched}
            </>
          )}
        </span>
        <div className="ml-auto flex gap-2">
          <button
            onClick={() => refetch()}
            className="inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs hover:bg-muted"
            title="マッチ状況を再取得"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={() => setEditorOpen(true)}
            className="inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs hover:bg-muted"
          >
            <Pencil className="h-3.5 w-3.5" /> 編集
          </button>
          <button
            onClick={apply}
            disabled={busy || submitting || data.entries.length === 0}
            className="inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs hover:bg-muted disabled:opacity-50"
            title="別名ノードを正式名ノードへ統合し、search_keys を再計算（LLM不要）"
          >
            <Merge className="h-3.5 w-3.5" /> 名寄せ実行
          </button>
        </div>
      </div>

      {editorOpen && <EditorModal report={data} onClose={() => setEditorOpen(false)} />}
    </div>
  )
}
