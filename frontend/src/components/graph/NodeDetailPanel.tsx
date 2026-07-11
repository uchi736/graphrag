import { Crosshair, RefreshCw, X } from "lucide-react"
import { useQuery } from "@tanstack/react-query"
import { apiGet } from "@/api/client"
import type { NodeInfo } from "@/api/types"
import type { GraphNode } from "@/lib/graphTransform"

/** プロパティの表示順（グラフに実在する主要キー。それ以外は汎用表示に回す） */
const KNOWN_ORDER = [
  "mention_count", "pagerank", "clause_no", "title", "heading_path",
  "version", "level", "name", "value", "type", "doc_type",
  "canonical_form", "aliases", "category", "definition", "notes", "surface_forms",
] as const

const LABELS: Record<string, string> = {
  mention_count: "言及数",
  pagerank: "PageRank",
  clause_no: "条番号",
  title: "タイトル",
  heading_path: "見出しパス",
  version: "版",
  level: "階層",
  name: "名称",
  value: "値",
  type: "値タイプ",
  doc_type: "文書タイプ",
  canonical_form: "正式名",
  aliases: "別名",
  category: "分類",
  definition: "定義",
  notes: "備考",
  surface_forms: "表記ゆれ",
}

/** search_keys / norm_id は検索用の内部キーなので折りたたみに隔離 */
const INTERNAL_KEYS = new Set(["search_keys", "norm_id"])

function formatValue(key: string, v: unknown): string {
  if (key === "pagerank" && typeof v === "number") return v.toPrecision(3)
  if (typeof v === "number") return String(v)
  return String(v)
}

function PropRow({ propKey, label, value }: { propKey: string; label: string; value: unknown }) {
  return (
    <div className="flex items-start gap-2 py-0.5">
      <span className="w-20 shrink-0 text-muted-foreground">{label}</span>
      {Array.isArray(value) ? (
        <span className="flex flex-wrap gap-1">
          {value.map((x, i) => (
            <span key={i} className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px]">
              {String(x)}
            </span>
          ))}
        </span>
      ) : (
        <span className="break-all font-mono">{formatValue(propKey, value)}</span>
      )}
    </div>
  )
}

export function NodeDetailPanel({
  node,
  onClose,
  onCenterOn,
}: {
  node: GraphNode | null
  onClose: () => void
  onCenterOn: (id: string) => void
}) {
  const { data: info, isLoading, isError } = useQuery({
    queryKey: ["node-info", node?.id],
    queryFn: () => apiGet<NodeInfo>("/api/graph/node", { id: node!.id }),
    enabled: node !== null,
    staleTime: 30_000,
  })

  if (!node) return null

  const props = info?.properties ?? {}
  const knownEntries = KNOWN_ORDER
    .filter((k) => props[k] != null && props[k] !== "")
    .map((k) => [k, props[k]] as const)
  const otherEntries = Object.entries(props).filter(
    ([k, v]) =>
      v != null && v !== "" &&
      !KNOWN_ORDER.includes(k as (typeof KNOWN_ORDER)[number]) &&
      !INTERNAL_KEYS.has(k),
  )
  const internalEntries = Object.entries(props).filter(([k]) => INTERNAL_KEYS.has(k))

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
            {info?.type ?? node.type} · degree {info?.degree ?? node.degree}
          </p>
        </div>
        <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* プロパティ */}
      <div className="mt-3 text-xs">
        {isLoading ? (
          <p className="flex items-center gap-1.5 text-muted-foreground">
            <RefreshCw className="h-3 w-3 animate-spin" /> プロパティ取得中…
          </p>
        ) : isError ? (
          <p className="text-muted-foreground">
            グラフにこのノードが見つかりません（表示が古い可能性。再読み込みしてください）
          </p>
        ) : knownEntries.length + otherEntries.length === 0 ? (
          <p className="text-muted-foreground">追加プロパティはありません</p>
        ) : (
          <div className="rounded-md border bg-background px-2.5 py-1.5">
            {knownEntries.map(([k, v]) => (
              <PropRow key={k} propKey={k} label={LABELS[k] ?? k} value={v} />
            ))}
            {otherEntries.map(([k, v]) => (
              <PropRow key={k} propKey={k} label={k} value={v} />
            ))}
          </div>
        )}
        {internalEntries.length > 0 && (
          <details className="mt-1.5">
            <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
              内部キー（検索用）
            </summary>
            <div className="mt-1 rounded-md border bg-background px-2.5 py-1.5">
              {internalEntries.map(([k, v]) => (
                <PropRow key={k} propKey={k} label={k} value={v} />
              ))}
            </div>
          </details>
        )}
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
