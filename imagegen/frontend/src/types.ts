// backend/schemas.py に対応

export type JobState = "queued" | "running" | "done" | "failed" | "cancelled"

export interface JobStatus {
  job_id: string
  kind: "generate" | "edit"
  state: JobState
  created_at: number
  started_at: number | null
  finished_at: number | null
  prompt_id: string | null
  error: string | null
  images: string[]
  image_urls: string[]
  params: Record<string, unknown>
}

export interface JobAccepted {
  job_id: string
  state: JobState
}

export interface ModelList {
  checkpoints: string[]
  note: string | null
}

export interface Health {
  ok: boolean
  comfyui_url: string
  comfyui_reachable: boolean
}

export interface GenerateParams {
  prompt: string
  negative_prompt: string
  width: number
  height: number
  steps: number
  cfg: number
  seed: number | null
  model: string | null
}

export interface EditParams {
  prompt: string
  negative_prompt: string
  strength: number
  steps: number
  cfg: number
  seed: number | null
  model: string | null
}
