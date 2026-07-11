import { Link } from "react-router-dom"
import { useHealth } from "@/hooks/useHealth"
import { cn } from "@/lib/utils"

export function AppHeader() {
  const { data: health } = useHealth()
  const dot = health?.ok ? "bg-emerald-400" : "bg-amber-400"
  const mismatch =
    health?.graph.exists && health.provenance.status === "mismatch"
  return (
    <header>
      <div className="brand-gradient text-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-xl font-bold tracking-wide">GraphRAG</h1>
            <p className="text-xs text-white/80">
              ハイブリッド検索 + ナレッジグラフ · 完全オンプレ
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className={cn("inline-block h-2.5 w-2.5 rounded-full", dot)} />
            <span className="text-white/90">
              {health
                ? `${health.checks.llm.model ?? "LLM"} · ${health.collection}`
                : "接続確認中…"}
            </span>
          </div>
        </div>
      </div>
      {/* グラフ出自と検索対象の不整合は静かに壊れる（KG自動オフ）ため常時警告する */}
      {mismatch && (
        <div className="border-b border-amber-300 bg-amber-50 px-6 py-1.5 text-center text-xs text-amber-800">
          ⚠️ ナレッジグラフは <b className="font-mono">{health!.provenance.graph_collection}</b> 用ですが、
          検索対象は <b className="font-mono">{health!.collection}</b> です — QAではKGが自動オフになっています。
          <Link to="/settings" className="ml-1 underline underline-offset-2">
            設定タブでコレクションを切替
          </Link>
        </div>
      )}
    </header>
  )
}
