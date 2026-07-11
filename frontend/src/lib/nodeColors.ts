/**
 * ノードタイプ→色。
 * タイプ名のハッシュで安定パレットに割り当てる（EDC/fujitsu等の任意スキーマ対応）。
 * Other/Unknown（ラベル無し）だけ灰色固定。
 * ※旧・昔話キーワード推論（太郎/山/川→Person/Place）は桃太郎デモの遺産のため廃止。
 */

const GRAY_TYPES: Record<string, string> = {
  Other: "#95A5A6",
  Unknown: "#7F8C8D",
}

const PALETTE = [
  "#667eea", "#f78fb3", "#63cdda", "#f5cd79", "#78e08f",
  "#e77f67", "#786fa6", "#4bcffa", "#ffb8b8", "#3dc1d3",
  "#e15f41", "#546de5", "#c44569", "#574b90", "#05c46b",
]

export function getNodeType(_nodeName: string, nodeLabel?: string | null): string {
  if (nodeLabel && nodeLabel !== "Unknown") return nodeLabel
  return "Other"
}

function hashString(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0
  return Math.abs(h)
}

export function getColorForType(nodeType: string): string {
  if (GRAY_TYPES[nodeType]) return GRAY_TYPES[nodeType]
  return PALETTE[hashString(nodeType) % PALETTE.length]
}
