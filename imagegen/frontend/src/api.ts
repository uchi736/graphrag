import type {
  EditParams,
  GenerateParams,
  Health,
  JobAccepted,
  JobStatus,
  ModelList,
} from "./types"

// 同一オリジン配信が既定。別オリジン運用時は VITE_API_BASE で上書き。
const BASE = import.meta.env.VITE_API_BASE ?? ""

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail: string
    try {
      const body = await res.json()
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body)
    } catch {
      detail = res.statusText
    }
    throw new Error(`HTTP ${res.status}: ${detail}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => fetch(`${BASE}/health`).then(asJson<Health>),

  models: () => fetch(`${BASE}/models`).then(asJson<ModelList>),

  job: (id: string) => fetch(`${BASE}/jobs/${id}`).then(asJson<JobStatus>),

  generate: (params: GenerateParams) =>
    fetch(`${BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }).then(asJson<JobAccepted>),

  edit: (params: EditParams, baseImage: File) => {
    const fd = new FormData()
    fd.append("base_image", baseImage)
    fd.append("prompt", params.prompt)
    fd.append("negative_prompt", params.negative_prompt)
    fd.append("strength", String(params.strength))
    fd.append("steps", String(params.steps))
    fd.append("cfg", String(params.cfg))
    if (params.seed !== null) fd.append("seed", String(params.seed))
    if (params.model) fd.append("model", params.model)
    return fetch(`${BASE}/edit`, { method: "POST", body: fd }).then(asJson<JobAccepted>)
  },
}

export function imageUrl(path: string): string {
  return `${BASE}${path}`
}
