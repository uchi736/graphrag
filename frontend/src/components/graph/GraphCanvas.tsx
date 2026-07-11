import { useEffect, useMemo, useRef, useState } from "react"
import ForceGraph2D from "react-force-graph-2d"
import type { EdgeRecord } from "@/api/types"
import { toGraphData, type GraphNode } from "@/lib/graphTransform"

/**
 * force-graph によるグラフ描画（旧 neo4j-viz iframe の置換）。
 * - タイプ別色 / 次数でサイズ / 日本語ラベル（ズームに応じて表示）
 * - ノードクリック → onNodeClick（詳細パネル連携）
 */
export function GraphCanvas({
  edges,
  height = 620,
  onNodeClick,
  showEdgeLabels = false,
  dimTypes = null,
}: {
  edges: EdgeRecord[]
  height?: number
  onNodeClick?: (node: GraphNode) => void
  /** 関係名をエッジ上に常時描画（QA参照グラフ等、エッジが主役の少数グラフ用） */
  showEdgeLabels?: boolean
  /** 凡例フィルタ: 指定タイプ以外のノード/エッジを淡色化（null/空=全表示） */
  dimTypes?: Set<string> | null
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(800)
  const data = useMemo(() => toGraphData(edges), [edges])
  const labelThreshold = data.nodes.length > 1000 ? 1.8 : 0.8
  const dimming = dimTypes != null && dimTypes.size > 0
  const isDimmed = (type: string) => dimming && !dimTypes!.has(type)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    setWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  return (
    <div ref={containerRef} className="overflow-hidden rounded-lg border bg-white">
      <ForceGraph2D
        width={width}
        height={height}
        graphData={data}
        cooldownTicks={200}
        nodeId="id"
        nodeVal={(n) => (n as GraphNode).size}
        nodeLabel={(n) => {
          const g = n as GraphNode
          return `${g.id}\n[${g.type}] degree=${g.degree}`
        }}
        nodeColor={(n) => {
          const g = n as GraphNode
          return isDimmed(g.type) ? "rgba(203,208,218,0.25)" : g.color
        }}
        nodeCanvasObjectMode={() => "after"}
        nodeCanvasObject={(node, ctx, globalScale) => {
          if (globalScale < labelThreshold) return
          const g = node as GraphNode & { x?: number; y?: number }
          if (isDimmed(g.type)) return
          const label = g.id.length > 16 ? g.id.slice(0, 16) + "…" : g.id
          const fontSize = Math.max(10 / globalScale, 2.2)
          ctx.font = `${fontSize}px "Yu Gothic UI", "Meiryo", sans-serif`
          ctx.textAlign = "center"
          ctx.textBaseline = "top"
          ctx.fillStyle = "rgba(29,35,51,0.85)"
          ctx.fillText(label, g.x ?? 0, (g.y ?? 0) + g.size + 1.5)
        }}
        linkLabel={(l) => (l as { relation: string }).relation}
        linkColor={(l) => {
          const link = l as { source_type?: string; target_type?: string }
          if (dimming && (isDimmed(link.source_type ?? "") || isDimmed(link.target_type ?? "")))
            return "rgba(150,155,170,0.06)"
          return showEdgeLabels ? "rgba(100,110,140,0.75)" : "rgba(120,130,150,0.35)"
        }}
        linkWidth={showEdgeLabels ? 1.4 : 1}
        linkDirectionalArrowLength={showEdgeLabels ? 4.5 : 3}
        linkDirectionalArrowRelPos={1}
        linkCanvasObjectMode={() => (showEdgeLabels ? "after" : undefined)}
        linkCanvasObject={
          showEdgeLabels
            ? (link, ctx, globalScale) => {
                const l = link as {
                  relation: string
                  source: { x?: number; y?: number } | string
                  target: { x?: number; y?: number } | string
                }
                if (typeof l.source !== "object" || typeof l.target !== "object") return
                const sx = l.source.x ?? 0
                const sy = l.source.y ?? 0
                const tx = l.target.x ?? 0
                const ty = l.target.y ?? 0
                const fontSize = Math.max(8 / globalScale, 2)
                ctx.font = `${fontSize}px "Yu Gothic UI", "Meiryo", sans-serif`
                // エッジ方向に沿わせ、上下逆さにならないよう反転
                let angle = Math.atan2(ty - sy, tx - sx)
                if (angle > Math.PI / 2 || angle < -Math.PI / 2) angle += Math.PI
                ctx.save()
                ctx.translate((sx + tx) / 2, (sy + ty) / 2)
                ctx.rotate(angle)
                const w = ctx.measureText(l.relation).width
                ctx.fillStyle = "rgba(255,255,255,0.85)"
                ctx.fillRect(-w / 2 - 1.5, -fontSize / 2 - 1, w + 3, fontSize + 2)
                ctx.fillStyle = "rgba(102,110,234,0.95)"
                ctx.textAlign = "center"
                ctx.textBaseline = "middle"
                ctx.fillText(l.relation, 0, 0)
                ctx.restore()
              }
            : undefined
        }
        onNodeClick={(n) => onNodeClick?.(n as GraphNode)}
      />
    </div>
  )
}
