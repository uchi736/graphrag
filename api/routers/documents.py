"""登録ドキュメント一覧エンドポイント。"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import require_ready
from api.state import AppState
from graphrag_core.services.documents import list_registered_documents

router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/documents")
def documents(st: AppState = Depends(require_ready)) -> dict:
    """ソース別チャンク数の集計。"""
    return list_registered_documents(st.settings.pg_conn, st.settings.pg_collection)
