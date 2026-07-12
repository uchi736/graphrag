import { useState } from "react"
import { Circle } from "lucide-react"
import { useHealth, useModels } from "./hooks/useAppMeta"
import { GenerateTab } from "./components/GenerateTab"
import { EditTab } from "./components/EditTab"

type Tab = "generate" | "edit"

export function App() {
  const [tab, setTab] = useState<Tab>("generate")
  const health = useHealth()
  const models = useModels()

  const comfyOk = health?.comfyui_reachable
  const healthLabel =
    health == null ? "API 未接続" : comfyOk ? "ComfyUI 接続OK" : "ComfyUI 未接続"
  const healthColor =
    health == null ? "text-red-400" : comfyOk ? "text-emerald-400" : "text-red-400"

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="flex items-center justify-between border-b border-slate-800 px-6 py-3.5">
        <h1 className="text-base font-semibold">画像生成・編集</h1>
        <span
          className={`flex items-center gap-1.5 text-xs ${healthColor}`}
          title={health?.comfyui_url}
        >
          <Circle size={9} fill="currentColor" /> {healthLabel}
        </span>
      </header>

      <nav className="flex gap-1 px-6 pt-4">
        {(["generate", "edit"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`rounded-t-lg border border-b-0 px-5 py-2 text-sm ${
              tab === t
                ? "border-slate-800 bg-slate-900 text-slate-100"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            {t === "generate" ? "生成" : "編集"}
          </button>
        ))}
      </nav>

      <main className="mx-auto max-w-5xl px-6 py-5">
        {models.note && (
          <p className="mb-4 rounded-lg border border-amber-900/50 bg-amber-950/30 px-3 py-2 text-xs text-amber-300">
            {models.note}
          </p>
        )}
        {tab === "generate" ? (
          <GenerateTab checkpoints={models.checkpoints} />
        ) : (
          <EditTab checkpoints={models.checkpoints} />
        )}
      </main>
    </div>
  )
}
