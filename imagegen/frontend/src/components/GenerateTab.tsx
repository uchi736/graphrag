import { useState } from "react"
import { Sparkles } from "lucide-react"
import { api } from "../api"
import { useJobPolling } from "../hooks/useJobPolling"
import { Field, ModelSelect, TextArea, TextInput } from "./Field"
import { JobResult } from "./JobResult"

export function GenerateTab({ checkpoints }: { checkpoints: string[] }) {
  const [prompt, setPrompt] = useState("")
  const [negative, setNegative] = useState("")
  const [width, setWidth] = useState(1024)
  const [height, setHeight] = useState(1024)
  const [steps, setSteps] = useState(20)
  const [cfg, setCfg] = useState(7)
  const [seed, setSeed] = useState("")
  const [model, setModel] = useState("")

  const { job, error, submitting, run } = useJobPolling()
  const busy = submitting || (job?.state === "queued" || job?.state === "running")

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    void run(() =>
      api.generate({
        prompt,
        negative_prompt: negative,
        width,
        height,
        steps,
        cfg,
        seed: seed === "" ? null : Number(seed),
        model: model || null,
      }),
    )
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <form
        onSubmit={submit}
        className="flex flex-col gap-3.5 rounded-xl border border-slate-800 bg-slate-900 p-5"
      >
        <Field label="プロンプト">
          <TextArea
            rows={3}
            required
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="生成したい画像を記述"
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
        <div className="grid grid-cols-2 gap-3">
          <Field label="幅">
            <TextInput
              type="number"
              min={64}
              max={4096}
              step={8}
              value={width}
              onChange={(e) => setWidth(Number(e.target.value))}
            />
          </Field>
          <Field label="高さ">
            <TextInput
              type="number"
              min={64}
              max={4096}
              step={8}
              value={height}
              onChange={(e) => setHeight(Number(e.target.value))}
            />
          </Field>
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
          disabled={busy}
          className="mt-1 flex items-center justify-center gap-2 rounded-lg bg-sky-600 py-2.5 text-sm font-medium text-white hover:bg-sky-500 disabled:cursor-progress disabled:opacity-60"
        >
          <Sparkles size={16} /> 生成する
        </button>
      </form>

      <JobResult job={job} error={error} submitting={submitting} />
    </div>
  )
}
