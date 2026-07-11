import { useCallback, useRef, useState } from "react"
import { postSse } from "@/api/sse"
import type { QaDefaults, QaEvidence } from "@/api/types"
import { useQaStore, summarizeConfig } from "@/stores/qaStore"

export type QaStatus = "idle" | "retrieving" | "streaming" | "done" | "error"

export interface QaStreamState {
  status: QaStatus
  answer: string
  evidence: QaEvidence | null
  error: string | null
  timing: { retrieval: number; generation: number } | null
}

const initial: QaStreamState = {
  status: "idle",
  answer: "",
  evidence: null,
  error: null,
  timing: null,
}

/** SSE QA の状態機械: idle → retrieving → streaming → done | error */
export function useQaStream() {
  const [state, setState] = useState<QaStreamState>(initial)
  const handleRef = useRef<{ abort: () => void } | null>(null)
  const pushHistory = useQaStore((s) => s.push)

  const submit = useCallback(
    (question: string, config: QaDefaults) => {
      handleRef.current?.abort()
      setState({ ...initial, status: "retrieving" })

      const handle = postSse("/api/qa/stream", { question, config }, (event, data) => {
        if (event === "retrieval") {
          setState((s) => ({ ...s, status: "streaming", evidence: data as QaEvidence }))
        } else if (event === "token") {
          const { delta } = data as { delta: string }
          setState((s) => ({ ...s, status: "streaming", answer: s.answer + delta }))
        } else if (event === "done") {
          const d = data as { answer: string; timing_ms: { retrieval: number; generation: number } }
          setState((s) => ({ ...s, status: "done", answer: d.answer, timing: d.timing_ms }))
          pushHistory({
            question,
            answer: d.answer,
            at: Date.now(),
            config_summary: summarizeConfig(config),
          })
        } else if (event === "error") {
          const d = data as { stage: string; message: string }
          setState((s) => ({ ...s, status: "error", error: `[${d.stage}] ${d.message}` }))
        }
      })
      handleRef.current = handle
      handle.finished.catch((e: unknown) => {
        if (e instanceof DOMException && e.name === "AbortError") {
          setState((s) => (s.status === "done" ? s : { ...s, status: "idle" }))
        } else {
          setState((s) => ({
            ...s,
            status: "error",
            error: e instanceof Error ? e.message : String(e),
          }))
        }
      })
    },
    [pushHistory],
  )

  const cancel = useCallback(() => {
    handleRef.current?.abort()
  }, [])

  const busy = state.status === "retrieving" || state.status === "streaming"
  return { ...state, busy, submit, cancel }
}
