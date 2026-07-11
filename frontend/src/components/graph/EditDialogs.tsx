import { useEffect, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"
import { apiSend } from "@/api/client"
import { Modal } from "@/components/common/Modal"

function useGraphMutation(onDone: () => void) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ method, path, body }: { method: "POST" | "PUT" | "DELETE"; path: string; body?: unknown }) =>
      apiSend(method, path, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["graph-overview"] })
      qc.invalidateQueries({ queryKey: ["graph-status"] })
      onDone()
    },
    onError: (e) => toast.error(e instanceof Error ? e.message : String(e)),
  })
}

const inputCls =
  "w-full rounded-md border bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"

// ── ノード追加/編集 ─────────────────────────────────────────────
export function NodeEditDialog({
  open,
  initial,
  onClose,
}: {
  open: boolean
  initial: { id: string; type: string } | null // null = 新規
  onClose: () => void
}) {
  const [id, setId] = useState("")
  const [type, setType] = useState("Term")
  const [propsJson, setPropsJson] = useState("{}")
  const isEdit = initial !== null

  useEffect(() => {
    setId(initial?.id ?? "")
    setType(initial?.type ?? "Term")
    setPropsJson("{}")
  }, [initial, open])

  const mut = useGraphMutation(() => {
    toast.success(isEdit ? "ノードを更新しました" : "ノードを追加しました")
    onClose()
  })

  const save = () => {
    let properties: Record<string, unknown>
    try {
      properties = JSON.parse(propsJson || "{}")
    } catch {
      toast.error("プロパティが正しいJSONではありません")
      return
    }
    if (isEdit) {
      mut.mutate({ method: "PUT", path: "/api/graph/node", body: { id, type, properties } })
    } else {
      mut.mutate({ method: "POST", path: "/api/graph/node", body: { id, type, properties } })
    }
  }

  return (
    <Modal open={open} title={isEdit ? "✏️ ノードを編集" : "➕ ノードを追加"} onClose={onClose}>
      <div className="space-y-3 text-sm">
        <label className="block">
          <span className="mb-1 block text-muted-foreground">ノードID</span>
          <input value={id} onChange={(e) => setId(e.target.value)} disabled={isEdit} className={inputCls} />
        </label>
        <label className="block">
          <span className="mb-1 block text-muted-foreground">タイプ（ラベル）</span>
          <input value={type} onChange={(e) => setType(e.target.value)} className={inputCls} />
        </label>
        <label className="block">
          <span className="mb-1 block text-muted-foreground">追加プロパティ（JSON）</span>
          <textarea
            value={propsJson}
            onChange={(e) => setPropsJson(e.target.value)}
            rows={3}
            className={inputCls + " font-mono text-xs"}
          />
        </label>
        <div className="flex justify-end gap-2 pt-1">
          <button onClick={onClose} className="rounded-md border px-4 py-1.5 hover:bg-muted">
            キャンセル
          </button>
          <button
            onClick={save}
            disabled={!id.trim() || mut.isPending}
            className="rounded-md bg-primary px-4 py-1.5 font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
          >
            保存
          </button>
        </div>
      </div>
    </Modal>
  )
}

// ── エッジ追加 ──────────────────────────────────────────────────
export function EdgeAddDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [source, setSource] = useState("")
  const [target, setTarget] = useState("")
  const [relType, setRelType] = useState("RELATED_TO")

  useEffect(() => {
    if (open) {
      setSource("")
      setTarget("")
      setRelType("RELATED_TO")
    }
  }, [open])

  const mut = useGraphMutation(() => {
    toast.success("エッジを追加しました")
    onClose()
  })

  return (
    <Modal open={open} title="➕ エッジを追加" onClose={onClose}>
      <div className="space-y-3 text-sm">
        <label className="block">
          <span className="mb-1 block text-muted-foreground">ソースノードID</span>
          <input value={source} onChange={(e) => setSource(e.target.value)} className={inputCls} />
        </label>
        <label className="block">
          <span className="mb-1 block text-muted-foreground">ターゲットノードID</span>
          <input value={target} onChange={(e) => setTarget(e.target.value)} className={inputCls} />
        </label>
        <label className="block">
          <span className="mb-1 block text-muted-foreground">リレーションタイプ</span>
          <input value={relType} onChange={(e) => setRelType(e.target.value)} className={inputCls} />
        </label>
        <div className="flex justify-end gap-2 pt-1">
          <button onClick={onClose} className="rounded-md border px-4 py-1.5 hover:bg-muted">
            キャンセル
          </button>
          <button
            onClick={() =>
              mut.mutate({
                method: "POST",
                path: "/api/graph/edge",
                body: { source, target, rel_type: relType },
              })
            }
            disabled={!source.trim() || !target.trim() || mut.isPending}
            className="rounded-md bg-primary px-4 py-1.5 font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
          >
            追加
          </button>
        </div>
      </div>
    </Modal>
  )
}

// ── 削除確認 ────────────────────────────────────────────────────
export function DeleteConfirmDialog({
  open,
  target,
  onClose,
}: {
  open: boolean
  target: { kind: "node"; id: string } | { kind: "edge"; source: string; target: string; relation: string } | null
  onClose: () => void
}) {
  const mut = useGraphMutation(() => {
    toast.success("削除しました")
    onClose()
  })
  if (!target) return null

  const label =
    target.kind === "node"
      ? `ノード「${target.id}」（接続エッジも削除されます）`
      : `エッジ「${target.source} -[${target.relation}]→ ${target.target}」`

  const doDelete = () => {
    if (target.kind === "node") {
      mut.mutate({ method: "DELETE", path: `/api/graph/node?id=${encodeURIComponent(target.id)}` })
    } else {
      mut.mutate({
        method: "DELETE",
        path: `/api/graph/edge?source=${encodeURIComponent(target.source)}&target=${encodeURIComponent(target.target)}`,
      })
    }
  }

  return (
    <Modal open={open} title="🗑️ 削除の確認" onClose={onClose}>
      <p className="text-sm">{label} を削除しますか？この操作は取り消せません。</p>
      <div className="mt-4 flex justify-end gap-2 text-sm">
        <button onClick={onClose} className="rounded-md border px-4 py-1.5 hover:bg-muted">
          キャンセル
        </button>
        <button
          onClick={doDelete}
          disabled={mut.isPending}
          className="rounded-md bg-destructive px-4 py-1.5 font-medium text-white hover:opacity-90 disabled:opacity-50"
        >
          削除する
        </button>
      </div>
    </Modal>
  )
}
