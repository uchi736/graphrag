import { useEffect } from "react"
import { Search } from "lucide-react"
import { useSettingsStore } from "@/stores/settingsStore"
import { useHealth, useSettingsInfo } from "@/hooks/useHealth"

const MODES = [
  { value: "hybrid", label: "ハイブリッド（BM25+ベクトル）" },
  { value: "vector", label: "ベクトル" },
  { value: "keyword", label: "キーワード（BM25）" },
] as const

export function SearchSettingsPanel() {
  const s = useSettingsStore()
  const { data: info } = useSettingsInfo()
  const { data: health } = useHealth()

  // サーバ既定を初回のみ反映（localStorage保存済みの値を優先）
  useEffect(() => {
    if (info) s.applyDefaults(info.qa_defaults)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [info])

  const sudachiOk = info?.sudachi_available ?? true
  const provMismatch = health?.provenance.status !== "match"

  return (
    <details className="rounded-lg border bg-card shadow-sm">
      <summary className="flex cursor-pointer items-center gap-2 px-4 py-3 text-sm font-medium">
        <Search className="h-4 w-4 text-primary" />
        検索設定
        <span className="ml-auto text-xs font-normal text-muted-foreground">
          {MODES.find((m) => m.value === s.search_mode)?.label.split("（")[0]} · top_k={s.retrieval_top_k} · KG{" "}
          {s.enable_knowledge_graph ? "ON" : "OFF"} · リランク {s.enable_rerank ? "ON" : "OFF"}
        </span>
      </summary>
      <div className="grid gap-4 border-t px-4 py-4 sm:grid-cols-2">
        <label className="block text-sm">
          <span className="mb-1 flex justify-between">
            <span>検索結果数（Top-K）</span>
            <span className="font-mono text-muted-foreground">{s.retrieval_top_k}</span>
          </span>
          <input
            type="range"
            min={1}
            max={20}
            value={s.retrieval_top_k}
            onChange={(e) => s.set("retrieval_top_k", Number(e.target.value))}
            className="w-full accent-[var(--color-primary)]"
          />
        </label>

        <fieldset className="text-sm">
          <legend className="mb-1">検索モード</legend>
          <div className="flex flex-col gap-1">
            {MODES.map((m) => (
              <label key={m.value} className="flex items-center gap-2">
                <input
                  type="radio"
                  name="search_mode"
                  checked={s.search_mode === m.value}
                  onChange={() => s.set("search_mode", m.value)}
                  className="accent-[var(--color-primary)]"
                />
                {m.label}
              </label>
            ))}
          </div>
        </fieldset>

        <div className="flex flex-col gap-2 text-sm">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={s.enable_rerank}
              onChange={(e) => s.set("enable_rerank", e.target.checked)}
              className="accent-[var(--color-primary)]"
            />
            リランキング（cross-encoder）
          </label>
          <label className={sudachiOk ? "flex items-center gap-2" : "flex items-center gap-2 opacity-50"}>
            <input
              type="checkbox"
              checked={s.enable_japanese_search && sudachiOk}
              disabled={!sudachiOk}
              onChange={(e) => s.set("enable_japanese_search", e.target.checked)}
              className="accent-[var(--color-primary)]"
            />
            日本語ハイブリッド検索{!sudachiOk && "（Sudachi未導入）"}
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={s.enable_knowledge_graph}
              onChange={(e) => s.set("enable_knowledge_graph", e.target.checked)}
              className="accent-[var(--color-primary)]"
            />
            回答にナレッジグラフを使用
          </label>
        </div>

        {provMismatch && s.enable_knowledge_graph && (
          <p className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-700 sm:col-span-2">
            ⚠️ グラフの出自コレクション（{health?.provenance.graph_collection ?? "不明"}）が現在のコレクション（
            {health?.collection}）と一致しないため、KGはスキップされます。
          </p>
        )}
      </div>
    </details>
  )
}
