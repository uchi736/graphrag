import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { Play, Wand2 } from "lucide-react"
import { toast } from "sonner"
import { apiSend } from "@/api/client"

const TEMPLATES = [
  { label: "カスタム（自分で入力）", nl: "" },
  { label: "特定エンティティに関連するすべての関係を表示", nl: "◯◯に関連するすべての関係を表示" },
  { label: "最も接続数が多いノードTop10を表示", nl: "最も接続数が多いノードTop10を表示" },
  { label: "すべてのリレーションシップタイプを表示", nl: "すべてのリレーションシップタイプとその数を表示" },
]

interface ExecuteResult {
  columns: string[]
  rows: Record<string, unknown>[]
  applied_limit: boolean
}

export function CypherPanel() {
  const [nl, setNl] = useState("")
  const [cypher, setCypher] = useState("")
  const [result, setResult] = useState<ExecuteResult | null>(null)

  const gen = useMutation({
    mutationFn: () => apiSend<{ cypher: string }>("POST", "/api/graph/cypher/generate", { query: nl }),
    onSuccess: (d) => setCypher(d.cypher),
    onError: (e) => toast.error(e instanceof Error ? e.message : String(e)),
  })

  const exec = useMutation({
    mutationFn: () => apiSend<ExecuteResult>("POST", "/api/graph/cypher/execute", { cypher }),
    onSuccess: (d) => {
      setResult(d)
      if (d.applied_limit) toast.info("結果上限として LIMIT 500 を付与しました")
      if (d.rows.length === 0) toast.warning("クエリ結果が空です")
    },
    onError: (e) => toast.error(e instanceof Error ? e.message : String(e)),
  })

  return (
    <div className="space-y-3">
      <div className="rounded-lg border bg-card p-4 shadow-sm">
        <div className="grid gap-3 sm:grid-cols-[220px_1fr]">
          <label className="block text-sm">
            <span className="mb-1 block text-muted-foreground">テンプレート</span>
            <select
              onChange={(e) => setNl(TEMPLATES[Number(e.target.value)].nl)}
              className="w-full rounded-md border bg-background px-2 py-1.5 text-sm"
            >
              {TEMPLATES.map((t, i) => (
                <option key={t.label} value={i}>
                  {t.label}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-sm">
            <span className="mb-1 block text-muted-foreground">自然言語クエリ</span>
            <div className="flex gap-2">
              <input
                value={nl}
                onChange={(e) => setNl(e.target.value)}
                placeholder="例: ボイラーに関するグラフを見たい"
                className="flex-1 rounded-md border bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
              />
              <button
                onClick={() => gen.mutate()}
                disabled={!nl.trim() || gen.isPending}
                className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
              >
                <Wand2 className={gen.isPending ? "h-3.5 w-3.5 animate-pulse" : "h-3.5 w-3.5"} />
                Cypherに変換
              </button>
            </div>
          </label>
        </div>

        <label className="mt-3 block text-sm">
          <span className="mb-1 block text-muted-foreground">Cypher（参照クエリのみ・編集可）</span>
          <textarea
            value={cypher}
            onChange={(e) => setCypher(e.target.value)}
            rows={4}
            placeholder="MATCH (n)-[r]->(m) RETURN n.id, type(r), m.id LIMIT 50"
            className="w-full rounded-md border bg-background px-3 py-2 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
          />
        </label>
        <button
          onClick={() => exec.mutate()}
          disabled={!cypher.trim() || exec.isPending}
          className="mt-2 inline-flex items-center gap-1.5 rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
        >
          <Play className={exec.isPending ? "h-4 w-4 animate-pulse" : "h-4 w-4"} />
          実行
        </button>
      </div>

      {result && result.rows.length > 0 && (
        <div className="overflow-auto rounded-lg border bg-card shadow-sm">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-muted/90 text-left text-xs text-muted-foreground">
              <tr>
                {result.columns.map((c) => (
                  <th key={c} className="px-3 py-2 font-medium">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.rows.map((r, i) => (
                <tr key={i} className="border-t hover:bg-muted/40">
                  {result.columns.map((c) => (
                    <td key={c} className="max-w-72 break-all px-3 py-1.5 text-xs">
                      {typeof r[c] === "object" ? JSON.stringify(r[c]) : String(r[c] ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="border-t px-3 py-1.5 text-right text-xs text-muted-foreground">{result.rows.length} 行</p>
        </div>
      )}
    </div>
  )
}
