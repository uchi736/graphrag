"""FastAPI 依存性: AppState 取得と ready ゲート。"""
from __future__ import annotations

from fastapi import HTTPException, Request

from api.state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.ctx


def require_ready(request: Request) -> AppState:
    """接続系が揃っていなければ 503（/api/health だけは素通し）。"""
    st: AppState = request.app.state.ctx
    if not st.ready:
        detail = {
            "message": "バックエンドが初期化されていません（/api/health を確認）",
            "missing_env": st.env_report.get("missing", []),
            "startup_error": st.startup_error,
        }
        raise HTTPException(status_code=503, detail=detail)
    return st
