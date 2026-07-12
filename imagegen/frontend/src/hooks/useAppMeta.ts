import { useEffect, useState } from "react"
import { api } from "../api"
import type { Health, ModelList } from "../types"

/** ヘルス（ComfyUI 疎通）を定期取得。 */
export function useHealth(intervalMs = 15000) {
  const [health, setHealth] = useState<Health | null>(null)
  useEffect(() => {
    let alive = true
    const tick = () =>
      api
        .health()
        .then((h) => alive && setHealth(h))
        .catch(() => alive && setHealth(null))
    tick()
    const id = window.setInterval(tick, intervalMs)
    return () => {
      alive = false
      window.clearInterval(id)
    }
  }, [intervalMs])
  return health
}

/** 利用可能な checkpoint 一覧。 */
export function useModels() {
  const [models, setModels] = useState<ModelList>({ checkpoints: [], note: null })
  useEffect(() => {
    let alive = true
    api
      .models()
      .then((m) => alive && setModels(m))
      .catch(() => alive && setModels({ checkpoints: [], note: "取得失敗" }))
    return () => {
      alive = false
    }
  }, [])
  return models
}
