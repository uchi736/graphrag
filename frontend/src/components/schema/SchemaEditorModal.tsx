import { useEffect, useState } from "react"
import { toast } from "sonner"
import { Plus, RefreshCw, Save, X } from "lucide-react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { apiGet, apiSend } from "@/api/client"
import type { SchemaFileResponse } from "@/api/types"

interface Item {
  name: string
  description?: string
  keep: boolean
  added?: boolean
}

/**
 * スキーマの人手キュレーションモーダル。
 * EDCが発見したスキーマ（SHARED_SCHEMA_PATH のJSON）を読み込み、
 * 不要なタイプ/関係のチェックを外す・新規追加する → 保存（.bakバックアップ付き上書き）。
 * 保存したスキーマは次回ビルドから有効（既存グラフは変わらない）。
 */
export function SchemaEditorModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const qc = useQueryClient()
  const { data, isLoading } = useQuery({
    queryKey: ["graph-schema-file"],
    queryFn: () => apiGet<SchemaFileResponse>("/api/graph/schema/file"),
    enabled: open,
    staleTime: 0,
  })

  const [types, setTypes] = useState<Item[]>([])
  const [rels, setRels] = useState<Item[]>([])
  const [newType, setNewType] = useState("")
  const [newRel, setNewRel] = useState("")
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!data) return
    const defs = data.data.node_type_definitions ?? {}
    setTypes(data.data.node_types.map((t) => ({ name: t, description: defs[t], keep: true })))
    setRels(data.data.relations.map((r) => ({ name: r.name, description: r.description, keep: true })))
  }, [data])

  if (!open) return null

  const addType = () => {
    const v = newType.trim()
    if (!v || types.some((t) => t.name === v)) return
    setTypes((ts) => [...ts, { name: v, keep: true, added: true }])
    setNewType("")
  }
  const addRel = () => {
    const v = newRel.trim()
    if (!v || rels.some((r) => r.name === v)) return
    setRels((rs) => [...rs, { name: v, keep: true, added: true }])
    setNewRel("")
  }

  const save = async () => {
    const keptTypes = types.filter((t) => t.keep).map((t) => t.name)
    const keptRels = rels.filter((r) => r.keep).map((r) => ({ name: r.name, description: r.description ?? "" }))
    if (keptTypes.length === 0 || keptRels.length === 0) {
      toast.error("タイプと関係は最低1件ずつ残してください")
      return
    }
    setSaving(true)
    try {
      const res = await apiSend<{ path: string; hint: string | null }>(
        "PUT", "/api/graph/schema/file", { node_types: keptTypes, relations: keptRels })
      toast.success(`スキーマを保存しました: ${res.path.split(/[\\/]/).pop()}（次回ビルドから有効）`)
      if (res.hint) toast.info(res.hint)
      qc.invalidateQueries({ queryKey: ["graph-schema"] })
      qc.invalidateQueries({ queryKey: ["graph-schema-file"] })
      onClose()
    } catch (e) {
      toast.error(`保存失敗: ${e instanceof Error ? e.message : e}`)
    } finally {
      setSaving(false)
    }
  }

  const Section = ({
    title,
    items,
    setItems,
    newValue,
    setNewValue,
    onAdd,
    placeholder,
  }: {
    title: string
    items: Item[]
    setItems: (fn: (prev: Item[]) => Item[]) => void
    newValue: string
    setNewValue: (v: string) => void
    onAdd: () => void
    placeholder: string
  }) => (
    <div className="min-w-0 flex-1">
      <p className="mb-2 text-xs font-medium text-muted-foreground">
        {title}（{items.filter((i) => i.keep).length}/{items.length} 残す）
      </p>
      <div className="max-h-72 space-y-1 overflow-y-auto rounded-md border bg-background p-2">
        {items.map((it) => (
          <label
            key={it.name}
            className={
              "flex cursor-pointer items-start gap-2 rounded px-2 py-1 text-xs hover:bg-muted " +
              (it.keep ? "" : "opacity-40 line-through")
            }
            title={it.description || undefined}
          >
            <input
              type="checkbox"
              checked={it.keep}
              onChange={() =>
                setItems((prev) => prev.map((p) => (p.name === it.name ? { ...p, keep: !p.keep } : p)))
              }
              className="mt-0.5"
            />
            <span className="font-mono">{it.name}</span>
            {it.added && <span className="rounded bg-green-50 px-1 text-[10px] text-green-700">新規</span>}
            {it.description && (
              <span className="truncate text-muted-foreground" title={it.description}>
                — {it.description}
              </span>
            )}
          </label>
        ))}
      </div>
      <div className="mt-2 flex gap-1.5">
        <input
          value={newValue}
          onChange={(e) => setNewValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onAdd()}
          placeholder={placeholder}
          className="min-w-0 flex-1 rounded-md border bg-background px-2 py-1 text-xs"
        />
        <button onClick={onAdd} className="rounded-md border px-2 py-1 text-xs hover:bg-muted">
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )

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
            <h3 className="text-sm font-bold">スキーマのキュレーション</h3>
            <p className="text-xs text-muted-foreground">
              {data?.path ? `編集対象: ${data.path}` : "SHARED_SCHEMA_PATH 未設定 → schemas/custom_*.json に新規保存"}
              　チェックを外す＝削除。保存で .bak を残して上書き、次回ビルドから有効。
            </p>
          </div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          {isLoading ? (
            <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" /> 読み込み中…
            </div>
          ) : (
            <div className="flex flex-col gap-4 sm:flex-row">
              <Section
                title="ノードタイプ"
                items={types}
                setItems={setTypes}
                newValue={newType}
                setNewValue={setNewType}
                onAdd={addType}
                placeholder="タイプを追加（例: Equipment）"
              />
              <Section
                title="関係"
                items={rels}
                setItems={setRels}
                newValue={newRel}
                setNewValue={setNewRel}
                onAdd={addRel}
                placeholder="関係を追加（例: RECORDED_IN）"
              />
            </div>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 border-t px-5 py-3">
          <button onClick={onClose} className="rounded-md border px-3 py-1.5 text-sm hover:bg-muted">
            キャンセル
          </button>
          <button
            onClick={save}
            disabled={saving || isLoading}
            className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
          >
            <Save className="h-4 w-4" /> 保存（次回ビルドから有効）
          </button>
        </div>
      </div>
    </div>
  )
}
