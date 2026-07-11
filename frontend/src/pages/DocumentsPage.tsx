import { RefreshCw } from "lucide-react"
import { useDocuments } from "@/hooks/useGraphData"

export default function DocumentsPage() {
  const { data, isLoading, isError, error, refetch, isFetching } = useDocuments()

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
        読み込み中…
      </div>
    )
  }
  if (isError) {
    return (
      <p className="rounded-md bg-red-50 px-4 py-3 text-sm text-destructive">
        取得エラー: {error instanceof Error ? error.message : String(error)}
      </p>
    )
  }
  if (!data || data.documents.length === 0) {
    return (
      <div className="rounded-lg border bg-card p-10 text-center text-sm text-muted-foreground">
        登録されたドキュメントはありません（「🛠️ 構築 / 取り込み」から追加してください）
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <div className="rounded-lg border bg-card px-5 py-3 shadow-sm">
          <p className="text-xs text-muted-foreground">総チャンク数</p>
          <p className="font-mono text-2xl font-bold">{data.total_chunks.toLocaleString()}</p>
        </div>
        <div className="rounded-lg border bg-card px-5 py-3 shadow-sm">
          <p className="text-xs text-muted-foreground">ドキュメント数</p>
          <p className="font-mono text-2xl font-bold">{data.documents.length}</p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="ml-auto inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs hover:bg-muted disabled:opacity-50"
        >
          <RefreshCw className={isFetching ? "h-3.5 w-3.5 animate-spin" : "h-3.5 w-3.5"} />
          再取得
        </button>
      </div>

      <div className="overflow-x-auto rounded-lg border bg-card shadow-sm">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/60 text-left text-xs text-muted-foreground">
              <th className="px-4 py-2.5 font-medium">ソースファイル</th>
              <th className="px-4 py-2.5 text-right font-medium">チャンク数</th>
            </tr>
          </thead>
          <tbody>
            {data.documents.map((d) => (
              <tr key={d.source} className="border-b last:border-0 hover:bg-muted/40">
                <td className="px-4 py-2 font-mono text-xs">{d.source}</td>
                <td className="px-4 py-2 text-right font-mono">{d.chunk_count.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
