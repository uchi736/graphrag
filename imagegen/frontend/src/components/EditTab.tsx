import { useState } from "react"
import { Wand2, Upload } from "lucide-react"
import { api } from "../api"
import { useJobPolling } from "../hooks/useJobPolling"
import { Field, ModelSelect, TextArea, TextInput } from "./Field"
import { JobResult } from "./JobResult"

export function EditTab({ checkpoints }: { checkpoints: string[] }) {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [prompt, setPrompt] = useState("")
  const [negative, setNegative] = useState("")
  const [strength, setStrength] = useState(0.75)
  const [steps, setSteps] = useState(20)
  const [cfg, setCfg] = useState(7)
  const [seed, setSeed] = useState("")
  const [model, setModel] = useState("")

  const { job, error, submitting, run } = useJobPolling()
  const busy = submitting || (job?.state === "queued" || job?.state === "running")

  const onFile = (f: File | null) => {
    setFile(f)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(f ? URL.createObjectURL(f) : null)
  }

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return
    void run(() =>
      api.edit(
        {
          prompt,
          negative_prompt: negative,
          strength,
          steps,
          cfg,
          seed: seed === "" ? null : Number(seed),
          model: model || null,
        },
        file,
      ),
    )
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <form
        onSubmit={submit}
        className="flex flex-col gap-3.5 rounded-xl border border-slate-800 bg-slate-900 p-5"
      >
        <Field label="ベース画像">
          <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-300 hover:border-sky-500">
            <Upload size={16} />
            {file ? file.name : "画像を選択"}
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => onFile(e.target.files?.[0] ?? null)}
            />
          </label>
        </Field>
        {previewUrl && (
          <img
            src={previewUrl}
            alt="ベース画像プレビュー"
            className="max-h-64 w-full rounded-lg border border-slate-700 object-contain"
          />
        )}
        <Field label="指示プロンプト">
          <TextArea
            rows={3}
            required
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="どう書き換えるか自然言語で指示"
          />
        </Field>
        <Field label="ネガティブプロンプト">
          <TextArea
            rows={2}
            value={negative}
            onChange={(e) => setNegative(e.target.value)}
            placeholder="除外したい要素（任意）"
          />
        </Field>
        <Field label={`変化の強さ (strength): ${strength.toFixed(2)}`}>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={strength}
            onChange={(e) => setStrength(Number(e.target.value))}
            className="accent-sky-500"
          />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="ステップ数">
            <TextInput
              type="number"
              min={1}
              max={150}
              value={steps}
              onChange={(e) => setSteps(Number(e.target.value))}
            />
          </Field>
          <Field label="CFG">
            <TextInput
              type="number"
              min={0}
              max={30}
              step={0.5}
              value={cfg}
              onChange={(e) => setCfg(Number(e.target.value))}
            />
          </Field>
          <Field label="シード">
            <TextInput
              type="number"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="空でランダム"
            />
          </Field>
          <Field label="モデル">
            <ModelSelect value={model} onChange={setModel} checkpoints={checkpoints} />
          </Field>
        </div>
        <button
          type="submit"
          disabled={busy || !file}
          className="mt-1 flex items-center justify-center gap-2 rounded-lg bg-sky-600 py-2.5 text-sm font-medium text-white hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <Wand2 size={16} /> 編集する
        </button>
      </form>

      <JobResult job={job} error={error} submitting={submitting} />
    </div>
  )
}
