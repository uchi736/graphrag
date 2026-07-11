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
}: {
  edges: EdgeRecord[]
  height?: number
  onNodeClick?: (node: GraphNode) => void
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(800)
  const data = useMemo(() => toGraphData(edges), [edges])
  const labelThreshold = data.nodes.length > 1000 ? 1.8 : 0.8

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
        nodeColor={(n) => (n as GraphNode).color}
        nodeCanvasObjectMode={() => "after"}
        nodeCanvasObject={(node, ctx, globalScale) => {
          if (globalScale < labelThreshold) return
          const g = node as GraphNode & { x?: number; y?: number }
          const label = g.id.length > 16 ? g.id.slice(0, 16) + "…" : g.id
          const fontSize = Math.max(10 / globalScale, 2.2)
          ctx.font = `${fontSize}px "Yu Gothic UI", "Meiryo", sans-serif`
          ctx.textAlign = "center"
          ctx.textBaseline = "top"
          ctx.fillStyle = "rgba(29,35,51,0.85)"
          ctx.fillText(label, g.x ?? 0, (g.y ?? 0) + g.size + 1.5)
        }}
        linkLabel={(l) => (l as { relation: string }).relation}
        linkColor={() => "rgba(120,130,150,0.35)"}
        linkDirectionalArrowLength={3}
        linkDirectionalArrowRelPos={1}
        onNodeClick={(n) => onNodeClick?.(n as GraphNode)}
      />
    </div>
  )
}
