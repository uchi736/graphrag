import { useState } from "react"
import { Download, RefreshCw } from "lucide-react"
import { toast } from "sonner"
import { useSettingsStore } from "@/stores/settingsStore"
import { useGraphOverview, useSubgraph, useGraphStatus } from "@/hooks/useGraphData"
import { GraphCanvas } from "@/components/graph/GraphCanvas"
import { NodeDetailPanel } from "@/components/graph/NodeDetailPanel"
import { SubgraphControls } from "@/components/graph/SubgraphControls"
import { DataTablePanel } from "@/components/graph/DataTablePanel"
import { CypherPanel } from "@/components/graph/CypherPanel"
import type { GraphNode } from "@/lib/graphTransform"
import { cn } from "@/lib/utils"

type Mode = "overview" | "subgraph" | "table" | "cypher"

export default function GraphPage() {
  const [mode, setMode] = useState<Mode>("overview")
  const [selected, setSelected] = useState<GraphNode | null>(null)
  const [centers, setCenters] = useState<string[]>([])
  const [hop, setHop] = useState(1)
  const maxNodes = useSettingsStore((s) => s.max_nodes)
  const setView = useSettingsStore((s) => s.setView)

  const { data: status } = useGraphStatus()
  const overview = useGraphOverview(maxNodes, mode === "overview" || mode === "table")
  const subgraph = useSubgraph(centers, hop, mode === "subgraph")

  const centerOn = (id: string) => {
    setCenters((cs) => (cs.includes(id) ? cs : [...cs, id]))
    setMode("subgraph")
  }

  const active = mode === "overview" ? overview : subgraph
  const edges = active.data ?? []

  const [exporting, setExporting] = useState(false)
  const exportGraphJson = async () => {
    setExporting(true)
    try {
      const res = await fetch("/api/graph/export")
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = "graph.json"
      a.click()
      URL.revokeObjectURL(url)
      toast.success("graph.json をダウンロードしました")
    } catch (e) {
      toast.error(`エクスポート失敗: ${e instanceof Error ? e.message : e}`)
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex rounded-md border bg-card p-0.5 text-sm">
          {(
            [
              ["overview", "全体可視化"],
              ["subgraph", "ノード中心"],
              ["table", "データテーブル"],
              ["cypher", "Cypher検索"],
            ] as [Mode, string][]
          ).map(([m, label]) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={cn(
                "rounded px-3 py-1.5",
                mode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              {label}
            </button>
          ))}
        </div>

        {status && (
          <span className="text-xs text-muted-foreground">
            グラフ全体: {status.graph.node_count.toLocaleString()} ノード /{" "}
            {status.graph.rel_count.toLocaleString()} エッジ
          </span>
        )}

        <button
          onClick={exportGraphJson}
          disabled={exporting}
          className="inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs hover:bg-muted disabled:opacity-50"
          title="全ノード・エッジを graph.json（node_link_data互換）でダウンロード"
        >
          <Download className="h-3.5 w-3.5" /> graph.json
        </button>

        {mode === "overview" && (
          <>
            <label className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
              最大表示ノード
              <input
                type="number"
                min={50}
                max={2000}
                step={50}
                value={maxNodes}
                onChange={(e) => setView("max_nodes", Number(e.target.value))}
                className="w-20 rounded-md border bg-background px-2 py-1 text-sm"
              />
            </label>
            <button
              onClick={() => overview.refetch()}
              disabled={overview.isFetching}
              className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs hover:bg-muted disabled:opacity-50"
            >
              <RefreshCw className={cn("h-3.5 w-3.5", overview.isFetching && "animate-spin")} />
              再読み込み
            </button>
          </>
        )}
      </div>

      {mode === "subgraph" && (
        <SubgraphControls
          centers={centers}
          hop={hop}
          onAddCenter={(id) => setCenters((cs) => [...cs, id])}
          onRemoveCenter={(id) => setCenters((cs) => cs.filter((c) => c !== id))}
          onClearCenters={() => setCenters([])}
          onHopChange={setHop}
        />
      )}

      {mode === "table" &&
        (overview.isLoading ? (
          <div className="flex h-64 items-center justify-center rounded-lg border bg-card text-sm text-muted-foreground">
            <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
            読み込み中…
          </div>
        ) : (
          <DataTablePanel edges={overview.data ?? []} />
        ))}

      {mode === "cypher" && <CypherPanel />}

      {(mode === "overview" || mode === "subgraph") && (
      <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
        <div>
          {active.isLoading ? (
            <div className="flex h-96 items-center justify-center rounded-lg border bg-card text-sm text-muted-foreground">
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              グラフを読み込んでいます…
            </div>
          ) : edges.length === 0 ? (
            <div className="flex h-96 items-center justify-center rounded-lg border bg-card text-sm text-muted-foreground">
              {mode === "subgraph" && centers.length === 0
                ? "中心ノードを検索して追加してください"
                : "表示できるエッジがありません"}
            </div>
          ) : (
            <GraphCanvas edges={edges} onNodeClick={setSelected} />
          )}
          {edges.length > 0 && (
            <p className="mt-1 text-right text-xs text-muted-foreground">{edges.length} エッジ表示中</p>
          )}
        </div>
        <div>
          {selected ? (
            <NodeDetailPanel node={selected} onClose={() => setSelected(null)} onCenterOn={centerOn} />
          ) : (
            <div className="rounded-lg border border-dashed bg-card/50 p-4 text-center text-xs text-muted-foreground">
              ノードをクリックすると詳細が表示されます
            </div>
          )}
        </div>
      </div>
      )}
    </div>
  )
}
