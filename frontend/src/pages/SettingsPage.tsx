import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"
import { AlertTriangle, Database, RotateCcw } from "lucide-react"
import { apiGet, apiSend } from "@/api/client"
import type { CollectionsResponse } from "@/api/types"
import { useHealth, useSettingsInfo } from "@/hooks/useHealth"
import { useSettingsStore } from "@/stores/settingsStore"
import { useQaStore } from "@/stores/qaStore"

export default function SettingsPage() {
  const s = useSettingsStore()
  const { data: info } = useSettingsInfo()
  const { data: health } = useHealth()
  const clearHistory = useQaStore((st) => st.clear)
  const qc = useQueryClient()
  const [confirmText, setConfirmText] = useState("")

  // コレクション切替（実行時・永続化。KG出自との不整合防止のため一覧に出自を明示）
  const { data: cols } = useQuery({
    queryKey: ["collections"],
    queryFn: () => apiGet<CollectionsResponse>("/api/admin/collections"),
    staleTime: 30_000,
  })
  const [pendingCol, setPendingCol] = useState<string | null>(null)
  const switchCol = useMutation({
    mutationFn: (name: string) =>
      apiSend<{ new: string; provenance: { status: string } }>(
        "POST", "/api/admin/collection", { name }),
    onSuccess: (r) => {
      if (r.provenance.status === "match") {
        toast.success(`コレクションを ${r.new} に切り替えました（グラフ出自と一致・KG有効）`)
      } else {
        toast.warning(
          `コレクションを ${r.new} に切り替えました — グラフ出自と不一致のためKGは自動オフです`)
      }
      setPendingCol(null)
      qc.invalidateQueries()
    },
    onError: (e) => toast.error(e instanceof Error ? e.message : String(e)),
  })

  const clearDb = useMutation({
    mutationFn: () => apiSend("POST", "/api/admin/clear-database", { confirm: confirmText }),
    onSuccess: () => {
      toast.success("データベースをクリアしました")
      setConfirmText("")
      clearHistory()
      qc.invalidateQueries()
    },
    onError: (e) => toast.error(e instanceof Error ? e.message : String(e)),
  })

  const collection = health?.collection ?? info?.collection ?? ""

  return (
    <div className="max-w-2xl space-y-6">
      {/* 検索対象コレクション（実行時切替・再起動不要） */}
      {cols && (
        <section className="rounded-lg border bg-card p-5 shadow-sm">
          <h2 className="mb-1 flex items-center gap-2 text-sm font-bold">
            <Database className="h-4 w-4 text-primary" />
            検索対象コレクション
          </h2>
          <p className="mb-3 text-xs text-muted-foreground">
            現在: <b className="font-mono">{cols.current}</b>
            {cols.graph_collection && (
              <>
                　/　ナレッジグラフの出自: <b className="font-mono">{cols.graph_collection}</b>
                {cols.graph_collection !== cols.current && (
                  <span className="ml-1 text-amber-600">（不一致 → KG自動オフ中）</span>
                )}
              </>
            )}
            　切替は即時反映され、再起動後も維持されます（.env は変更しません）
          </p>
          <div className="flex flex-wrap gap-2">
            <select
              value={pendingCol ?? cols.current}
              onChange={(e) => setPendingCol(e.target.value)}
              className="rounded-md border bg-background px-3 py-1.5 font-mono text-sm"
            >
              {cols.collections.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}（{c.chunks.toLocaleString()}チャンク）
                  {c.name === cols.graph_collection ? " ★グラフ出自" : ""}
                </option>
              ))}
            </select>
            <button
              onClick={() => pendingCol && switchCol.mutate(pendingCol)}
              disabled={!pendingCol || pendingCol === cols.current || switchCol.isPending}
              className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
            >
              {switchCol.isPending ? "切替中…" : "切り替える"}
            </button>
          </div>
        </section>
      )}

      {/* KG詳細設定（QAリクエストに載る値。localStorage永続） */}
      <section className="rounded-lg border bg-card p-5 shadow-sm">
        <h2 className="mb-4 text-sm font-bold">🔧 ナレッジグラフ詳細設定</h2>
        <div className="space-y-4 text-sm">
          <label className="flex items-center justify-between gap-4">
            <span>
              KGソースチャンクを回答コンテキストに含める
              <span className="block text-xs text-muted-foreground">
                グラフ関係の抽出元チャンク本文を最大5件追加します
              </span>
            </span>
            <input
              type="checkbox"
              checked={s.include_kg_source_chunks}
              onChange={(e) => s.set("include_kg_source_chunks", e.target.checked)}
              className="h-4 w-4 accent-[var(--color-primary)]"
            />
          </label>

          <label className="block">
            <span className="mb-1 flex justify-between">
              <span>グラフ探索ホップ数</span>
              <span className="font-mono text-muted-foreground">{s.graph_hop_count}</span>
            </span>
            <input
              type="range"
              min={1}
              max={3}
              value={s.graph_hop_count}
              onChange={(e) => s.set("graph_hop_count", Number(e.target.value))}
              className="w-full accent-[var(--color-primary)]"
            />
          </label>

          <label className="flex items-center justify-between gap-4">
            <span>
              エンティティベクトル検索
              <span className="block text-xs text-muted-foreground">
                表記揺れをベクトル類似で補完してノード照合します
              </span>
            </span>
            <input
              type="checkbox"
              checked={s.enable_entity_vector}
              onChange={(e) => s.set("enable_entity_vector", e.target.checked)}
              className="h-4 w-4 accent-[var(--color-primary)]"
            />
          </label>

          <label className="block">
            <span className="mb-1 flex justify-between">
              <span>エンティティ類似度閾値</span>
              <span className="font-mono text-muted-foreground">
                {s.entity_similarity_threshold.toFixed(2)}
              </span>
            </span>
            <input
              type="range"
              min={0.5}
              max={1}
              step={0.01}
              value={s.entity_similarity_threshold}
              disabled={!s.enable_entity_vector}
              onChange={(e) => s.set("entity_similarity_threshold", Number(e.target.value))}
              className="w-full accent-[var(--color-primary)] disabled:opacity-40"
            />
          </label>

          {info && (
            <button
              onClick={() => {
                s.resetToDefaults(info.qa_defaults)
                toast.info("サーバ既定値に戻しました")
              }}
              className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs hover:bg-muted"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              検索設定をサーバ既定に戻す
            </button>
          )}
        </div>
      </section>

      {/* 接続情報（読み取り専用） */}
      {info && (
        <section className="rounded-lg border bg-card p-5 text-sm shadow-sm">
          <h2 className="mb-3 text-sm font-bold">🩺 接続情報</h2>
          <dl className="grid grid-cols-[140px_1fr] gap-y-1.5 text-xs">
            <dt className="text-muted-foreground">コレクション</dt>
            <dd className="font-mono">{info.collection}</dd>
            <dt className="text-muted-foreground">LLM</dt>
            <dd className="font-mono">{info.llm.model}（{info.llm.provider}）</dd>
            <dt className="text-muted-foreground">Embedding</dt>
            <dd className="font-mono">{info.embedding_provider}</dd>
            <dt className="text-muted-foreground">リランカー</dt>
            <dd>{info.reranker_enabled ? "有効" : "無効"}</dd>
            <dt className="text-muted-foreground">Sudachi</dt>
            <dd>{info.sudachi_available ? "利用可能" : "未導入"}</dd>
          </dl>
        </section>
      )}

      {/* Danger Zone */}
      <section className="rounded-lg border border-destructive/40 bg-card p-5 shadow-sm">
        <h2 className="mb-2 flex items-center gap-2 text-sm font-bold text-destructive">
          <AlertTriangle className="h-4 w-4" />
          データベースをクリア
        </h2>
        <p className="text-xs text-muted-foreground">
          Neo4j の全グラフと PGVector のコレクション「{collection}」を削除します。
          この操作は取り消せません。実行するには下にコレクション名を入力してください。
        </p>
        <div className="mt-3 flex gap-2">
          <input
            value={confirmText}
            onChange={(e) => setConfirmText(e.target.value)}
            placeholder={collection}
            className="flex-1 rounded-md border bg-background px-3 py-1.5 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-destructive"
          />
          <button
            onClick={() => clearDb.mutate()}
            disabled={confirmText !== collection || clearDb.isPending}
            className="rounded-md bg-destructive px-4 py-1.5 text-sm font-medium text-white hover:opacity-90 disabled:opacity-40"
          >
            完全に削除する
          </button>
        </div>
      </section>
    </div>
  )
}
