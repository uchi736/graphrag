import type { ReactNode } from "react"
import { AlertTriangle, RefreshCw } from "lucide-react"
import { useHealth } from "@/hooks/useHealth"

/**
 * 起動ゲート（旧 sidebar.py の st.stop() ハードガード相当）。
 * - API到達不可 / env不足 / Neo4j不通 → 全画面エラー + 再試行
 * - degraded でも graph/neo4j 以外の理由なら警告バナー付きで通す余地は今後判断
 */
export function HealthGate({ children }: { children: ReactNode }) {
  const { data, isLoading, isError, error, refetch, isFetching } = useHealth()

  if (isLoading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center text-muted-foreground">
        <RefreshCw className="mr-2 h-5 w-5 animate-spin" />
        バックエンドに接続しています…
      </div>
    )
  }

  const problems: string[] = []
  if (isError) {
    problems.push(`APIに到達できません: ${error instanceof Error ? error.message : String(error)}`)
  } else if (data && !data.ok) {
    if (!data.checks.env.ok)
      problems.push(`環境変数が不足しています: ${data.checks.env.missing.join(", ")}`)
    if (!data.checks.neo4j.ok)
      problems.push(`Neo4j接続エラー: ${data.checks.neo4j.error ?? "不明"}`)
    if (data.startup_error) problems.push(`初期化エラー: ${data.startup_error}`)
    if (problems.length === 0) problems.push("バックエンドが degraded 状態です")
  }

  if (problems.length > 0) {
    return (
      <div className="mx-auto mt-16 max-w-xl rounded-lg border border-destructive/40 bg-card p-8 shadow-sm">
        <div className="flex items-center gap-2 text-destructive">
          <AlertTriangle className="h-6 w-6" />
          <h2 className="text-lg font-bold">バックエンドを利用できません</h2>
        </div>
        <ul className="mt-4 list-disc space-y-1 pl-6 text-sm text-foreground/90">
          {problems.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
        <p className="mt-4 text-xs text-muted-foreground">
          .env の設定と Neo4j / PostgreSQL / vLLM の稼働を確認してください。
        </p>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="mt-6 inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
        >
          <RefreshCw className={isFetching ? "h-4 w-4 animate-spin" : "h-4 w-4"} />
          再試行
        </button>
      </div>
    )
  }

  return <>{children}</>
}
