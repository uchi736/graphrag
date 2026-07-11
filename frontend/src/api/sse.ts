/**
 * POST 対応の SSE クライアント（EventSource は GET 専用のため自前実装）。
 * fetch + ReadableStream + TextDecoder で `event:`/`data:` 行をパースする。
 */

export interface SseHandle {
  abort: () => void
  finished: Promise<void>
}

export function postSse(
  path: string,
  body: unknown,
  onEvent: (event: string, data: unknown) => void,
): SseHandle {
  const controller = new AbortController()

  const finished = (async () => {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    if (!res.ok || !res.body) {
      let message = `${res.status} ${res.statusText}`
      try {
        const err = await res.json()
        if (err?.detail) message = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail)
      } catch {
        /* ignore */
      }
      throw new Error(message)
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder("utf-8")
    let buffer = ""
    let eventName = ""
    let dataLines: string[] = []

    const dispatch = () => {
      if (eventName && dataLines.length > 0) {
        try {
          onEvent(eventName, JSON.parse(dataLines.join("\n")))
        } catch {
          /* 不正なJSONは無視 */
        }
      }
      eventName = ""
      dataLines = []
    }

    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).replace(/\r$/, "")
        buffer = buffer.slice(idx + 1)
        if (line === "") {
          dispatch() // 空行 = イベント境界
        } else if (line.startsWith("event:")) {
          eventName = line.slice(6).trim()
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trimStart())
        }
        // ":" で始まる行はコメント（ping）→ 無視
      }
    }
    dispatch()
  })()

  return { abort: () => controller.abort(), finished }
}
