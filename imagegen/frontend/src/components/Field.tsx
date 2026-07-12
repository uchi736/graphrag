import type { ReactNode } from "react"

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5 text-xs text-slate-400">
      <span>{label}</span>
      {children}
    </label>
  )
}

const inputCls =
  "rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 " +
  "outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500"

export function TextInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={`${inputCls} ${props.className ?? ""}`} />
}

export function TextArea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} className={`${inputCls} resize-y ${props.className ?? ""}`} />
}

export function Select(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={`${inputCls} ${props.className ?? ""}`} />
}

export function ModelSelect({
  value,
  onChange,
  checkpoints,
}: {
  value: string
  onChange: (v: string) => void
  checkpoints: string[]
}) {
  return (
    <Select value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">
        {checkpoints.length ? "（ワークフロー既定）" : "（モデル未取得）"}
      </option>
      {checkpoints.map((c) => (
        <option key={c} value={c}>
          {c}
        </option>
      ))}
    </Select>
  )
}
