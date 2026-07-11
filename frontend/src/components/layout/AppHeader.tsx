import { useHealth } from "@/hooks/useHealth"
import { cn } from "@/lib/utils"

export function AppHeader() {
  const { data: health } = useHealth()
  const dot = health?.ok ? "bg-emerald-400" : "bg-amber-400"
  return (
    <header className="brand-gradient text-white">
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
    </header>
  )
}
