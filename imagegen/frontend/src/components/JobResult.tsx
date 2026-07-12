import { Loader2, AlertTriangle, ImageOff } from "lucide-react"
import { imageUrl } from "../api"
import type { JobStatus } from "../types"

const LABEL: Record<string, string> = {
  queued: "待機中",
  running: "生成中",
  done: "完了",
  failed: "失敗",
  cancelled: "キャンセル",
}

const STATE_COLOR: Record<string, string> = {
  queued: "text-sky-400",
  running: "text-sky-400",
  done: "text-emerald-400",
  failed: "text-red-400",
  cancelled: "text-red-400",
}

export function JobResult({
  job,
  error,
  submitting,
}: {
  job: JobStatus | null
  error: string | null
  submitting: boolean
}) {
  const active = job && (job.state === "queued" || job.state === "running")

  return (
    <div className="flex min-h-[240px] flex-col gap-3 rounded-xl border border-slate-800 bg-slate-900 p-4">
      {error && (
        <p className="flex items-center gap-2 text-sm text-red-400">
          <AlertTriangle size={16} /> {error}
        </p>
      )}

      {!job && !error && (
        <p className="m-auto flex flex-col items-center gap-2 text-sm text-slate-500">
          {submitting ? (
            <>
              <Loader2 className="animate-spin" size={20} /> 投入中…
            </>
          ) : (
            <>
              <ImageOff size={22} /> 結果がここに表示されます
            </>
          )}
        </p>
      )}

      {job && (
        <>
          <p className={`flex items-center gap-2 text-sm font-medium ${STATE_COLOR[job.state]}`}>
            {active && <Loader2 className="animate-spin" size={16} />}
            {LABEL[job.state] ?? job.state}
            <span className="font-mono text-xs text-slate-500">job {job.job_id}</span>
          </p>

          {job.state === "failed" && job.error && (
            <pre className="overflow-x-auto whitespace-pre-wrap rounded-lg bg-slate-950 p-3 text-xs text-red-300">
              {job.error}
            </pre>
          )}

          {job.state === "done" &&
            job.image_urls.map((u) => (
              <a key={u} href={imageUrl(u)} target="_blank" rel="noopener noreferrer">
                <img
                  src={imageUrl(u)}
                  alt="結果画像"
                  className="w-full rounded-lg border border-slate-700"
                />
              </a>
            ))}

          {Object.keys(job.params).length > 0 && (
            <p className="mt-auto font-mono text-[11px] text-slate-500">
              seed={String(job.params.seed)} · steps={String(job.params.steps)} · model=
              {String(job.params.model ?? "既定")}
            </p>
          )}
        </>
      )}
    </div>
  )
}
