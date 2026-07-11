import { useEffect, useRef, useState } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"

export interface JobProgress {
  stage: string
  message: string
  current: number | null
  total: number | null
  percent: number | null
  ok: number
  err: number
  level?: string
}

export interface JobState {
  jobId: string | null
  state: "idle" | "running" | "succeeded" | "failed" | "cancelled"
  progress: JobProgress | null
  warnings: string[]
  result: Record<string, unknown> | null
  error: string | null
}

const initial: JobState = { jobId: null, state: "idle", progress: null, warnings: [], result: null, error: null }
const STORAGE_KEY = "graphrag.active-job"

/** ジョブ進捗の購読（EventSource + リロード後の再購読）。 */
export function useJobProgress() {
  const [job, setJob] = useState<JobState>(initial)
  const esRef = useRef<EventSource | null>(null)
  const qc = useQueryClient()

  const attach = (jobId: string) => {
    esRef.current?.close()
    sessionStorage.setItem(STORAGE_KEY, jobId)
    setJob({ ...initial, jobId, state: "running" })

    const es = new EventSource(`/api/jobs/${jobId}/events`)
    esRef.current = es

    es.addEventListener("progress", (e) => {
      const data = JSON.parse((e as MessageEvent).data) as JobProgress
      setJob((j) => ({
        ...j,
        progress: data,
        warnings: data.level === "warning" ? [...j.warnings.slice(-19), data.message] : j.warnings,
      }))
    })
    es.addEventListener("state", (e) => {
      const data = JSON.parse((e as MessageEvent).data) as {
        state: JobState["state"] | "running"
        result?: Record<string, unknown>
        error?: string
      }
      if (data.state === "running") return
      setJob((j) => ({ ...j, state: data.state, result: data.result ?? null, error: data.error ?? null }))
      es.close()
      sessionStorage.removeItem(STORAGE_KEY)
      // 完了したらサーバ状態を更新
      qc.invalidateQueries({ queryKey: ["health"] })
      qc.invalidateQueries({ queryKey: ["documents"] })
      qc.invalidateQueries({ queryKey: ["graph-overview"] })
      qc.invalidateQueries({ queryKey: ["graph-status"] })
      if (data.state === "succeeded") {
        const r = (data.result ?? {}) as { ok?: number; err?: number; total?: number }
        if ((r.err ?? 0) > 0 && (r.ok ?? 0) > 0)
          toast.warning(`部分的に完了: ${r.ok}/${r.total} 成功、${r.err}件失敗`)
        else toast.success("ジョブが完了しました")
      } else if (data.state === "failed") {
        toast.error(`ジョブ失敗: ${data.error ?? "不明なエラー"}`)
      } else if (data.state === "cancelled") {
        toast.info("ジョブをキャンセルしました")
      }
    })
    es.onerror = () => {
      // SSE切断時: ジョブが終わっていれば snapshot で確定させる
      fetch(`/api/jobs/${jobId}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((snap) => {
          if (snap && snap.state !== "running" && snap.state !== "queued") {
            setJob((j) => ({ ...j, state: snap.state, result: snap.result, error: snap.error }))
            es.close()
            sessionStorage.removeItem(STORAGE_KEY)
          }
        })
        .catch(() => {})
    }
  }

  // リロード後の再購読
  useEffect(() => {
    const saved = sessionStorage.getItem(STORAGE_KEY)
    if (saved) attach(saved)
    return () => esRef.current?.close()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const cancel = async () => {
    if (job.jobId) await fetch(`/api/jobs/${job.jobId}/cancel`, { method: "POST" })
  }

  return { job, attach, cancel, busy: job.state === "running" }
}
