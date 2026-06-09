"""
langfuse_utils.py
==================
Langfuse tracing helpers (SDK v4).
環境変数 LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST が設定されている場合のみ有効化。
未設定時は get_langfuse_config() が None を返し、ゼロオーバーヘッドで動作する。

v4 階層トレーシング:
  @observe() デコレータでトレース階層を自動構築。
  内部で CallbackHandler() を生成すると現在のSpanを自動継承。
"""

import os
import logging
from typing import Optional, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_LANGFUSE_AVAILABLE = False
try:
    from langfuse.langchain import CallbackHandler
    from langfuse import observe as _observe, propagate_attributes as _propagate_attributes
    _LANGFUSE_AVAILABLE = True
except ImportError:
    CallbackHandler = None  # type: ignore
    _observe = None
    _propagate_attributes = None


# ── No-op フォールバック（langfuse未インストール時） ─────────────────

def _noop_observe(func=None, *, name=None, capture_input=None,
                  capture_output=None, **kwargs):
    """langfuse未インストール時の @observe no-op デコレータ"""
    if func is not None:
        return func
    return lambda f: f


@contextmanager
def _noop_propagate_attributes(**kwargs):
    """langfuse未インストール時の propagate_attributes no-op"""
    yield


# ── Public API ────────────────────────────────────────────────────

observe = _observe if _LANGFUSE_AVAILABLE else _noop_observe
propagate_attributes = _propagate_attributes if _LANGFUSE_AVAILABLE else _noop_propagate_attributes


def is_langfuse_enabled() -> bool:
    """Langfuseトレーシングが有効かどうかを判定"""
    return _LANGFUSE_AVAILABLE and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))


def get_langfuse_callback() -> Optional[dict]:
    """@observe()コンテキスト内で使用。現在のトレースを継承するCallbackHandlerを返す。

    @observe() デコレータ内で呼ぶと、生成されるCallbackHandlerは
    現在のSpan/Traceの子として自動登録される。

    Returns:
        {"callbacks": [CallbackHandler()]} or None
    """
    if not is_langfuse_enabled():
        return None
    return {"callbacks": [CallbackHandler()]}


def get_langfuse_config(
    trace_name: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Optional[dict]:
    """LangChain の .invoke() に渡す config dict を生成（後方互換用）。

    @observe() を使わない箇所（build_kg.py）向け。
    独立したトレースを生成する旧パターン。

    Returns:
        {"callbacks": [CallbackHandler(...)]} or None
    """
    if not is_langfuse_enabled():
        return None

    # v4: trace_context dict で名前・セッション等を渡す
    trace_ctx = {}
    if trace_name is not None:
        trace_ctx["name"] = trace_name
    if session_id is not None:
        trace_ctx["session_id"] = session_id
    if user_id is not None:
        trace_ctx["user_id"] = user_id
    if tags is not None:
        trace_ctx["tags"] = tags

    handler = CallbackHandler(trace_context=trace_ctx if trace_ctx else None)
    return {"callbacks": [handler]}


def update_current_span(**kwargs) -> None:
    """@observe()コンテキスト内で現在のSpanのinput/outputを手動セットする。

    capture_input=False と併用し、ベクトル以外の有用な情報だけを記録する。
    langfuse未インストール or 無効時は何もしない。

    Usage:
        @observe(name="my_func", capture_input=False)
        def my_func(query_text, query_vector, k):
            update_current_span(input={"query_text": query_text, "k": k})
            ...
    """
    if not is_langfuse_enabled():
        return
    try:
        from langfuse import get_client
        get_client().update_current_span(**kwargs)
    except Exception:
        pass  # トレース更新失敗は無視
