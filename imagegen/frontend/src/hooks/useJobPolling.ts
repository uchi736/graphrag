import { useCallback, useEffect, useRef, useState } from "react"
import { api } from "../api"
import type { JobStatus } from "../types"

const POLL_MS = 1500
const TERMINAL = new Set(["done", "failed", "cancelled"])

interface State {
  job: JobStatus | null
  error: string | null
  submitting: boolean
}

/** ジョブ投入 → 完了までポーリング表示するフック。 */
export function useJobPolling() {
  const [state, setState] = useState<State>({ job: null, error: null, submitting: false })
  const timer = useRef<number | null>(null)

  const clear = () => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current)
      timer.current = null
    }
  }
  useEffect(() => clear, [])

  const poll = useCallback((id: string) => {
    const tick = async () => {
      try {
        const job = await api.job(id)
        setState((s) => ({ ...s, job }))
        if (!TERMINAL.has(job.state)) {
          timer.current = window.setTimeout(tick, POLL_MS)
        }
      } catch (e) {
        setState((s) => ({ ...s, error: (e as Error).message }))
      }
    }
    void tick()
  }, [])

  /** submitFn は job_id を返す投入呼び出し。 */
  const run = useCallback(
    async (submitFn: () => Promise<{ job_id: string }>) => {
      clear()
      setState({ job: null, error: null, submitting: true })
      try {
        const { job_id } = await submitFn()
        poll(job_id)
      } catch (e) {
        setState({ job: null, error: (e as Error).message, submitting: false })
        return
      }
      setState((s) => ({ ...s, submitting: false }))
    },
    [poll],
  )

  return { ...state, run }
}
