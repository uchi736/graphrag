export function PagePlaceholder({ title, note }: { title: string; note: string }) {
  return (
    <div className="rounded-lg border bg-card p-10 text-center shadow-sm">
      <h2 className="text-lg font-bold">{title}</h2>
      <p className="mt-2 text-sm text-muted-foreground">{note}</p>
    </div>
  )
}
