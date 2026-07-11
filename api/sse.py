"""SSE（Server-Sent Events）整形ヘルパ。

sync generator を StreamingResponse に渡す想定（starlette が threadpool で
イテレートするため、コアの blocking generator をそのまま使える）。
"""
from __future__ import annotations

import json
from typing import Iterator

from fastapi.responses import StreamingResponse

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 将来 nginx を挟んだ場合のバッファ無効化
}


def format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_response(events: Iterator[tuple[str, dict]]) -> StreamingResponse:
    """(event名, payload) のイテレータを SSE ストリームに変換する。"""

    def _gen():
        # 接続確立を早めるコメント行（プロキシのバッファ検出対策）
        yield ": ping\n\n"
        for name, data in events:
            yield format_sse(name, data)

    return StreamingResponse(_gen(), media_type="text/event-stream", headers=SSE_HEADERS)
