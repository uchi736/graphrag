import type { EdgeRecord } from "@/api/types"
import { getColorForType, getNodeType } from "@/lib/nodeColors"

export interface GraphNode {
  id: string
  type: string
  color: string
  degree: number
  size: number
  docs: string[]
}

export interface GraphLink {
  source: string
  target: string
  relation: string
  /** 両端ノードのタイプ（凡例フィルタでのエッジ淡色化用） */
  source_type: string
  target_type: string
}

export interface GraphData {
  nodes: GraphNode[]
  links: GraphLink[]
}

/**
 * EdgeRecord[] → force-graph 用 {nodes, links}。
 * ノードサイズは 12 + min(degree, 18)（visualization.py:81 踏襲。canvas半径なので /2）。
 */
export function toGraphData(edges: EdgeRecord[]): GraphData {
  const nodeMap = new Map<string, GraphNode>()

  const upsert = (id: string, label: string, degree: number, docs: (string | null)[]) => {
    const existing = nodeMap.get(id)
    const cleanDocs = (docs ?? []).filter((d): d is string => !!d)
    if (existing) {
      existing.degree = Math.max(existing.degree, degree)
      for (const d of cleanDocs) if (!existing.docs.includes(d)) existing.docs.push(d)
      return
    }
    const type = getNodeType(id, label)
    nodeMap.set(id, {
      id,
      type,
      color: getColorForType(type),
      degree,
      size: (12 + Math.min(degree, 18)) / 3.5,
      docs: cleanDocs,
    })
  }

  const links: GraphLink[] = []
  for (const e of edges) {
    if (!e.source || !e.target) continue
    upsert(e.source, e.source_type, e.source_degree ?? 0, e.source_docs)
    upsert(e.target, e.target_type, e.target_degree ?? 0, e.target_docs)
    links.push({
      source: e.source,
      target: e.target,
      relation: e.relation,
      source_type: nodeMap.get(e.source)!.type,
      target_type: nodeMap.get(e.target)!.type,
    })
  }

  return { nodes: [...nodeMap.values()], links }
}
