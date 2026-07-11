import { useMemo } from "react"
import type { GraphNode } from "@/lib/graphTransform"

/**
 * ノードタイプの凡例（件数付きチップ）。
 * onToggle を渡すとチップがクリック可能になり、選択タイプ以外を淡色化する
 * フィルタとして機能する（selectedTypes が空 = フィルタなし）。
 */
export function GraphLegend({
  nodes,
  selectedTypes,
  onToggle,
}: {
  nodes: GraphNode[]
  selectedTypes?: Set<string>
  onToggle?: (type: string) => void
}) {
  const items = useMemo(() => {
    const m = new Map<string, { color: string; count: number }>()
    for (const n of nodes) {
      const cur = m.get(n.type)
      if (cur) cur.count += 1
      else m.set(n.type, { color: n.color, count: 1 })
    }
    return [...m.entries()]
      .map(([type, v]) => ({ type, ...v }))
      .sort((a, b) => b.count - a.count)
  }, [nodes])

  if (items.length === 0) return null
  const filtering = (selectedTypes?.size ?? 0) > 0

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {items.map((it) => {
        const active = !filtering || selectedTypes!.has(it.type)
        return (
          <button
            key={it.type}
            onClick={onToggle ? () => onToggle(it.type) : undefined}
            disabled={!onToggle}
            className={
              "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs transition-opacity " +
              (onToggle ? "cursor-pointer hover:bg-muted " : "cursor-default ") +
              (active ? "" : "opacity-35")
            }
            title={onToggle ? "クリックでこのタイプを強調（再クリックで解除）" : undefined}
          >
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: it.color }}
            />
            {it.type}
            <span className="text-muted-foreground">({it.count})</span>
          </button>
        )
      })}
      {filtering && onToggle && (
        <button
          onClick={() => selectedTypes!.forEach((t) => onToggle(t))}
          className="text-xs text-muted-foreground underline-offset-2 hover:underline"
        >
          解除
        </button>
      )}
    </div>
  )
}
