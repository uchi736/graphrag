import { useEffect } from "react"
import { Search } from "lucide-react"
import { useSettingsStore } from "@/stores/settingsStore"
import { useHealth, useSettingsInfo } from "@/hooks/useHealth"
import { cn } from "@/lib/utils"

const MODES = [
  { value: "hybrid", label: "ハイブリッド（BM25+ベクトル）" },
  { value: "vector", label: "ベクトル" },
  { value: "keyword", label: "キーワード（BM25）" },
  { value: "none", label: "文書検索なし（グラフのみ用）" },
] as const

/**
 * 検索モードのプリセット（llm-graph-builder のチャットモード切替に相当）。
 * 既存の設定キーの組み合わせに名前を付けたもの — 選ぶと該当キーを一括設定する。
 * ここに無いキー（top_k・リランク等）は現在値を維持。
 */
const PRESETS = [
  {
    key: "graph_hybrid",
    label: "🕸️+🔀 グラフ＋ハイブリッド",
    hint: "既定。KGとBM25+ベクトルの全部入り",
    cfg: { search_mode: "hybrid", enable_knowledge_graph: true },
  },
  {
    key: "graph",
    label: "🕸️ グラフのみ",
    hint: "文書検索なし。KGの関係とKGソースチャンクだけで回答",
    cfg: { search_mode: "none", enable_knowledge_graph: true, include_kg_source_chunks: true },
  },
  {
    key: "hybrid",
    label: "🔀 ハイブリッド",
    hint: "BM25+ベクトル（KGなし）",
    cfg: { search_mode: "hybrid", enable_knowledge_graph: false },
  },
  {
    key: "vector",
    label: "🎯 ベクトル",
    hint: "意味類似のみ（KGなし）",
    cfg: { search_mode: "vector", enable_knowledge_graph: false },
  },
  {
    key: "keyword",
    label: "🔤 キーワード",
    hint: "BM25のみ（KGなし）",
    cfg: { search_mode: "keyword", enable_knowledge_graph: false },
  },
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

  // 現在の設定に一致するプリセットを判定（どれとも違えばカスタム）
  const activePreset = PRESETS.find((p) =>
    Object.entries(p.cfg).every(([k, v]) => (s as unknown as Record<string, unknown>)[k] === v),
  )
  const applyPreset = (p: (typeof PRESETS)[number]) => {
    for (const [k, v] of Object.entries(p.cfg)) {
      s.set(k as Parameters<typeof s.set>[0], v as never)
    }
  }

  return (
    <div className="space-y-2">
      {/* モード切替バー（常時表示） */}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-xs text-muted-foreground">検索モード:</span>
        {PRESETS.map((p) => (
          <button
            key={p.key}
            onClick={() => applyPreset(p)}
            title={p.hint}
            className={cn(
              "rounded-full border px-3 py-1 text-xs transition-colors",
              activePreset?.key === p.key
                ? "border-transparent bg-primary font-medium text-primary-foreground"
                : "bg-card text-muted-foreground hover:bg-muted hover:text-foreground",
            )}
          >
            {p.label}
          </button>
        ))}
        {!activePreset && (
          <span className="rounded-full border border-dashed px-3 py-1 text-xs text-muted-foreground">
            ⚙ カスタム
          </span>
        )}
      </div>

      <details className="rounded-lg border bg-card shadow-sm">
        <summary className="flex cursor-pointer items-center gap-2 px-4 py-3 text-sm font-medium">
          <Search className="h-4 w-4 text-primary" />
          詳細設定
          <span className="ml-auto text-xs font-normal text-muted-foreground">
            {activePreset ? activePreset.label.replace(/^\S+ /, "") : "カスタム"} · top_k={s.retrieval_top_k} · KG{" "}
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
            <legend className="mb-1">文書検索方式</legend>
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

          {s.search_mode === "none" && !s.enable_knowledge_graph && (
            <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-destructive sm:col-span-2">
              ⚠️ 文書検索なし＋KG OFF ではコンテキストが空になります。KGを有効にするか検索方式を変更してください。
            </p>
          )}

          {provMismatch && s.enable_knowledge_graph && (
            <p className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-700 sm:col-span-2">
              ⚠️ グラフの出自コレクション（{health?.provenance.graph_collection ?? "不明"}）が現在のコレクション（
              {health?.collection}）と一致しないため、KGはスキップされます。
            </p>
          )}
        </div>
      </details>
    </div>
  )
}
