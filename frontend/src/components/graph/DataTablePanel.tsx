import { useMemo, useState } from "react"
import { Download, Pencil, Plus, Trash2 } from "lucide-react"
import type { EdgeRecord } from "@/api/types"
import { toGraphData } from "@/lib/graphTransform"
import { downloadCsv } from "@/lib/csv"
import { NodeEditDialog, EdgeAddDialog, DeleteConfirmDialog } from "@/components/graph/EditDialogs"
import { cn } from "@/lib/utils"

type DeleteTarget =
  | { kind: "node"; id: string }
  | { kind: "edge"; source: string; target: string; relation: string }
  | null

export function DataTablePanel({ edges }: { edges: EdgeRecord[] }) {
  const [view, setView] = useState<"nodes" | "edges">("nodes")
  const [filter, setFilter] = useState("")
  const [nodeDialog, setNodeDialog] = useState<{ open: boolean; initial: { id: string; type: string } | null }>({
    open: false,
    initial: null,
  })
  const [edgeDialogOpen, setEdgeDialogOpen] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<DeleteTarget>(null)

  const { nodes } = useMemo(() => toGraphData(edges), [edges])
  const f = filter.toLowerCase()
  const filteredNodes = nodes.filter((n) => !f || n.id.toLowerCase().includes(f) || n.type.toLowerCase().includes(f))
  const filteredEdges = edges.filter(
    (e) => !f || e.source.toLowerCase().includes(f) || e.target.toLowerCase().includes(f) || e.relation.toLowerCase().includes(f),
  )

  return (
    <div className="rounded-lg border bg-card shadow-sm">
      <div className="flex flex-wrap items-center gap-2 border-b p-3">
        <div className="flex rounded-md border p-0.5 text-xs">
          {(["nodes", "edges"] as const).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={cn(
                "rounded px-3 py-1",
                view === v ? "bg-primary text-primary-foreground" : "text-muted-foreground",
              )}
            >
              {v === "nodes" ? `ノード (${nodes.length})` : `エッジ (${edges.length})`}
            </button>
          ))}
        </div>
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="フィルタ…"
          className="w-44 rounded-md border bg-background px-2.5 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
        />
        <div className="ml-auto flex gap-2">
          {view === "nodes" ? (
            <button
              onClick={() => setNodeDialog({ open: true, initial: null })}
              className="inline-flex items-center gap-1 rounded-md bg-primary px-2.5 py-1 text-xs text-primary-foreground hover:opacity-90"
            >
              <Plus className="h-3.5 w-3.5" /> ノード追加
            </button>
          ) : (
            <button
              onClick={() => setEdgeDialogOpen(true)}
              className="inline-flex items-center gap-1 rounded-md bg-primary px-2.5 py-1 text-xs text-primary-foreground hover:opacity-90"
            >
              <Plus className="h-3.5 w-3.5" /> エッジ追加
            </button>
          )}
          <button
            onClick={() =>
              view === "nodes"
                ? downloadCsv(
                    "nodes.csv",
                    ["id", "type", "degree", "docs"],
                    filteredNodes.map((n) => [n.id, n.type, n.degree, n.docs.join(";")]),
                  )
                : downloadCsv(
                    "edges.csv",
                    ["source", "relation", "target", "source_type", "target_type"],
                    filteredEdges.map((e) => [e.source, e.relation, e.target, e.source_type, e.target_type]),
                  )
            }
            className="inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs hover:bg-muted"
          >
            <Download className="h-3.5 w-3.5" /> CSV
          </button>
        </div>
      </div>

      <div className="max-h-[560px] overflow-auto">
        {view === "nodes" ? (
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-muted/90 text-left text-xs text-muted-foreground">
              <tr>
                <th className="px-3 py-2 font-medium">ID</th>
                <th className="px-3 py-2 font-medium">タイプ</th>
                <th className="px-3 py-2 text-right font-medium">次数</th>
                <th className="px-3 py-2 font-medium">操作</th>
              </tr>
            </thead>
            <tbody>
              {filteredNodes.map((n) => (
                <tr key={n.id} className="border-t hover:bg-muted/40">
                  <td className="max-w-72 break-all px-3 py-1.5">{n.id}</td>
                  <td className="px-3 py-1.5">
                    <span
                      className="mr-1.5 inline-block h-2 w-2 rounded-full"
                      style={{ backgroundColor: n.color }}
                    />
                    {n.type}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-xs">{n.degree}</td>
                  <td className="px-3 py-1.5">
                    <div className="flex gap-1.5">
                      <button
                        title="編集"
                        onClick={() => setNodeDialog({ open: true, initial: { id: n.id, type: n.type } })}
                        className="text-muted-foreground hover:text-primary"
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </button>
                      <button
                        title="削除"
                        onClick={() => setDeleteTarget({ kind: "node", id: n.id })}
                        className="text-muted-foreground hover:text-destructive"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-muted/90 text-left text-xs text-muted-foreground">
              <tr>
                <th className="px-3 py-2 font-medium">ソース</th>
                <th className="px-3 py-2 font-medium">リレーション</th>
                <th className="px-3 py-2 font-medium">ターゲット</th>
                <th className="px-3 py-2 font-medium">操作</th>
              </tr>
            </thead>
            <tbody>
              {filteredEdges.map((e, i) => (
                <tr key={`${e.source}-${e.relation}-${e.target}-${i}`} className="border-t hover:bg-muted/40">
                  <td className="max-w-56 break-all px-3 py-1.5">{e.source}</td>
                  <td className="px-3 py-1.5 font-mono text-xs text-primary">{e.relation}</td>
                  <td className="max-w-56 break-all px-3 py-1.5">{e.target}</td>
                  <td className="px-3 py-1.5">
                    <button
                      title="削除"
                      onClick={() =>
                        setDeleteTarget({ kind: "edge", source: e.source, target: e.target, relation: e.relation })
                      }
                      className="text-muted-foreground hover:text-destructive"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <NodeEditDialog
        open={nodeDialog.open}
        initial={nodeDialog.initial}
        onClose={() => setNodeDialog({ open: false, initial: null })}
      />
      <EdgeAddDialog open={edgeDialogOpen} onClose={() => setEdgeDialogOpen(false)} />
      <DeleteConfirmDialog open={deleteTarget !== null} target={deleteTarget} onClose={() => setDeleteTarget(null)} />
    </div>
  )
}
