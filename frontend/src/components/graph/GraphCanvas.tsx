import { useEffect, useMemo, useRef, useState } from "react"
import ForceGraph2D from "react-force-graph-2d"
import type { EdgeRecord } from "@/api/types"
import { toGraphData, type GraphNode } from "@/lib/graphTransform"

// force-graph の nodeRelSize 既定値。実描画半径 = sqrt(nodeVal) * NODE_REL_SIZE
const NODE_REL_SIZE = 4

/** 少数グラフ（サブグラフ・QA参照グラフ）とみなすエッジ数。関係名の常時描画等を自動有効化 */
const SMALL_GRAPH_LINKS = 80

/** 大きいグラフでも、このズーム倍率以上に拡大したら関係名をエッジ上に描画する */
const EDGE_LABEL_ZOOM = 1.5

/**
 * force-graph によるグラフ描画（旧 neo4j-viz iframe の置換）。
 * - タイプ別色 / 次数でサイズ / 日本語ラベル（ノード半径に連動した可読サイズ＋白フチ）
 * - 少数グラフでは自動でエッジ濃色化＋関係名を常時描画
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
  /** 関係名をエッジ上に常時描画（少数グラフでは自動有効） */
  showEdgeLabels?: boolean
  /** 凡例フィルタ: 指定タイプ以外のノード/エッジを淡色化（null/空=全表示） */
  dimTypes?: Set<string> | null
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null)
  const [width, setWidth] = useState(800)
  const data = useMemo(() => toGraphData(edges), [edges])
  const small = data.links.length <= SMALL_GRAPH_LINKS
  const edgeLabels = showEdgeLabels || small
  const labelThreshold = small ? 0 : data.nodes.length > 1000 ? 1.8 : 0.8
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

  // 少数グラフはノード同士が接触しやすいので斥力とリンク距離を広げる
  useEffect(() => {
    const fg = fgRef.current
    if (!fg) return
    try {
      fg.d3Force("charge")?.strength(small ? -180 : -40)
      fg.d3Force("link")?.distance(small ? 70 : 30)
      fg.d3ReheatSimulation()
    } catch {
      /* force未初期化時は無視 */
    }
  }, [data, small])

  return (
    <div ref={containerRef} className="overflow-hidden rounded-lg border bg-white">
      <ForceGraph2D
        ref={fgRef}
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
          const label = small
            ? g.id
            : g.id.length > 16 ? g.id.slice(0, 16) + "…" : g.id
          // 実描画半径（sqrt(nodeVal)*relSize）。旧実装は g.size を半径扱いして
          // ラベルが円に食い込んでいた
          const r = Math.sqrt(Math.max(g.size, 0.1)) * NODE_REL_SIZE
          // フォントはズーム反比例と半径連動の大きい方 → ノードが大きく見える時は
          // ラベルも比例して読めるサイズになる
          const fontSize = Math.max(10 / globalScale, r * 0.42)
          ctx.font = `${fontSize}px "Yu Gothic UI", "Meiryo", sans-serif`
          ctx.textAlign = "center"
          ctx.textBaseline = "top"
          const x = g.x ?? 0
          const y = (g.y ?? 0) + r + fontSize * 0.25
          // 白フチで背景・エッジと分離
          ctx.lineWidth = Math.max(fontSize / 4.5, 1 / globalScale)
          ctx.strokeStyle = "rgba(255,255,255,0.9)"
          ctx.strokeText(label, x, y)
          ctx.fillStyle = "rgba(24,30,46,0.95)"
          ctx.fillText(label, x, y)
        }}
        linkLabel={(l) => (l as { relation: string }).relation}
        linkColor={(l) => {
          const link = l as { source_type?: string; target_type?: string }
          if (dimming && (isDimmed(link.source_type ?? "") || isDimmed(link.target_type ?? "")))
            return "rgba(150,155,170,0.06)"
          return edgeLabels ? "rgba(100,110,140,0.75)" : "rgba(120,130,150,0.35)"
        }}
        linkWidth={edgeLabels ? 1.4 : 1}
        linkDirectionalArrowLength={edgeLabels ? 4.5 : 3}
        linkDirectionalArrowRelPos={1}
        linkCanvasObjectMode={() => "after"}
        linkCanvasObject={
          (link, ctx, globalScale) => {
                // 少数グラフ/明示ONは常時、大きいグラフはズームインした時だけ関係名を描画
                if (!edgeLabels && globalScale < EDGE_LABEL_ZOOM) return
                const l = link as {
                  relation: string
                  source_type?: string
                  target_type?: string
                  pair_index?: number
                  source: { x?: number; y?: number } | string
                  target: { x?: number; y?: number } | string
                }
                if (typeof l.source !== "object" || typeof l.target !== "object") return
                if (dimming && (isDimmed(l.source_type ?? "") || isDimmed(l.target_type ?? ""))) return
                const sx = l.source.x ?? 0
                const sy = l.source.y ?? 0
                const tx = l.target.x ?? 0
                const ty = l.target.y ?? 0
                // ツールチップと同じ「濃色の角丸バッジ＋白文字」を水平描画。
                // 同一ペア間の複数エッジはエッジの法線方向に段積みして重なり回避
                const fontSize = Math.max(8 / globalScale, 2)
                const padX = fontSize * 0.5
                const padY = fontSize * 0.28
                let mx = (sx + tx) / 2
                let my = (sy + ty) / 2
                const k = l.pair_index ?? 0
                if (k > 0) {
                  const len = Math.hypot(tx - sx, ty - sy) || 1
                  const nx = -(ty - sy) / len
                  const ny = (tx - sx) / len
                  const off = Math.ceil(k / 2) * (fontSize + padY * 2 + 1) * (k % 2 === 1 ? 1 : -1)
                  mx += nx * off
                  my += ny * off
                }
                ctx.font = `${fontSize}px "Yu Gothic UI", "Meiryo", sans-serif`
                const w = ctx.measureText(l.relation).width
                const bx = mx - w / 2 - padX
                const by = my - fontSize / 2 - padY
                const bw = w + padX * 2
                const bh = fontSize + padY * 2
                const r = Math.min(2 / globalScale + 1, bh / 2)
                ctx.fillStyle = "rgba(71,77,94,0.88)"
                ctx.beginPath()
                if (typeof ctx.roundRect === "function") ctx.roundRect(bx, by, bw, bh, r)
                else ctx.rect(bx, by, bw, bh)
                ctx.fill()
                ctx.fillStyle = "rgba(255,255,255,0.96)"
                ctx.textAlign = "center"
                ctx.textBaseline = "middle"
                ctx.fillText(l.relation, mx, my)
              }
        }
        onNodeClick={(n) => onNodeClick?.(n as GraphNode)}
      />
    </div>
  )
}
