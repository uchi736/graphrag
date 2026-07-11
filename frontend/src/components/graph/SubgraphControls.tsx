import { useState } from "react"
import { Plus, Trash2 } from "lucide-react"
import { useNodeSearch } from "@/hooks/useGraphData"

export function SubgraphControls({
  centers,
  hop,
  onAddCenter,
  onRemoveCenter,
  onClearCenters,
  onHopChange,
}: {
  centers: string[]
  hop: number
  onAddCenter: (id: string) => void
  onRemoveCenter: (id: string) => void
  onClearCenters: () => void
  onHopChange: (hop: number) => void
}) {
  const [q, setQ] = useState("")
  const { data } = useNodeSearch(q)
  const candidates = (data?.ids ?? []).filter((id) => !centers.includes(id))

  return (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      <div className="flex flex-wrap items-end gap-4">
        <label className="block min-w-56 flex-1 text-sm">
          <span className="mb-1 block text-muted-foreground">中心ノードを検索</span>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="ノード名の一部を入力（例: ボイラー）"
            className="w-full rounded-md border bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
          />
          {q && candidates.length > 0 && (
            <ul className="mt-1 max-h-44 overflow-y-auto rounded-md border bg-card text-sm shadow">
              {candidates.slice(0, 12).map((id) => (
                <li key={id}>
                  <button
                    onClick={() => {
                      onAddCenter(id)
                      setQ("")
                    }}
                    className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left hover:bg-muted"
                  >
                    <Plus className="h-3.5 w-3.5 text-primary" />
                    {id}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </label>

        <label className="block text-sm">
          <span className="mb-1 flex justify-between gap-3">
            <span className="text-muted-foreground">表示範囲（Hop）</span>
            <span className="font-mono">{hop}</span>
          </span>
          <input
            type="range"
            min={1}
            max={3}
            value={hop}
            onChange={(e) => onHopChange(Number(e.target.value))}
            className="w-36 accent-[var(--color-primary)]"
          />
        </label>
      </div>

      {centers.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          {centers.map((c) => (
            <span key={c} className="inline-flex items-center gap-1 rounded-full bg-[var(--color-brand-from)]/10 px-2.5 py-0.5 text-xs text-primary">
              {c}
              <button onClick={() => onRemoveCenter(c)} className="hover:text-destructive">
                ×
              </button>
            </span>
          ))}
          <button
            onClick={onClearCenters}
            className="ml-1 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-destructive"
          >
            <Trash2 className="h-3 w-3" />
            リセット
          </button>
        </div>
      )}
    </div>
  )
}
