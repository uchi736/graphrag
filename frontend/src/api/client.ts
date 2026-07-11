/**
 * 薄いfetchラッパ。P5 で openapi-fetch + 生成型へ移行するまでの暫定。
 * エラーは Error(message) に正規化（FastAPI の detail を吸い上げる）。
 */

export class ApiError extends Error {
  status: number
  detail: unknown
  constructor(status: number, message: string, detail?: unknown) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (res.ok) return (await res.json()) as T
  let detail: unknown
  let message = `${res.status} ${res.statusText}`
  try {
    const body = await res.json()
    detail = body?.detail ?? body
    if (typeof detail === "string") message = detail
    else if (detail && typeof detail === "object" && "message" in detail)
      message = String((detail as { message: unknown }).message)
  } catch {
    /* JSONでないエラー本文は無視 */
  }
  throw new ApiError(res.status, message, detail)
}

export async function apiGet<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(path, window.location.origin)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined) url.searchParams.set(k, String(v))
    }
  }
  const res = await fetch(url.pathname + url.search, { headers: { Accept: "application/json" } })
  return handle<T>(res)
}

export async function apiSend<T>(
  method: "POST" | "PUT" | "DELETE",
  path: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  return handle<T>(res)
}
