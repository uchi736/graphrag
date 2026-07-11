import { Crosshair, X } from "lucide-react"
import type { GraphNode } from "@/lib/graphTransform"

export function NodeDetailPanel({
  node,
  onClose,
  onCenterOn,
}: {
  node: GraphNode | null
  onClose: () => void
  onCenterOn: (id: string) => void
}) {
  if (!node) return null
  return (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="break-all text-sm font-bold">{node.id}</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            <span
              className="mr-1 inline-block h-2.5 w-2.5 rounded-full align-middle"
              style={{ backgroundColor: node.color }}
            />
            {node.type} · degree {node.degree}
          </p>
        </div>
        <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
          <X className="h-4 w-4" />
        </button>
      </div>
      {node.docs.length > 0 && (
        <div className="mt-3">
          <p className="mb-1 text-xs font-medium text-muted-foreground">言及ドキュメント</p>
          <div className="flex flex-wrap gap-1">
            {node.docs.map((d) => (
              <span key={d} className="rounded bg-muted px-2 py-0.5 font-mono text-xs">
                {d}
              </span>
            ))}
          </div>
        </div>
      )}
      <button
        onClick={() => onCenterOn(node.id)}
        className="mt-4 inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:opacity-90"
      >
        <Crosshair className="h-3.5 w-3.5" />
        このノードを中心に表示
      </button>
    </div>
  )
}
