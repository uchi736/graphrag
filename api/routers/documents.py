"""登録ドキュメント一覧・チャンク閲覧エンドポイント。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.deps import require_ready
from api.state import AppState
from graphrag_core.services.documents import list_document_chunks, list_registered_documents

router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/documents")
def documents(st: AppState = Depends(require_ready)) -> dict:
    """ソース別チャンク数の集計。"""
    return list_registered_documents(st.settings.pg_conn, st.settings.pg_collection)


@router.get("/documents/chunks")
def document_chunks(
    source: str = Query(..., description="ソース文書名（documents一覧のsource）"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    focus: str | None = Query(None, description="このチャンクIDを含むページに offset を自動調整"),
    st: AppState = Depends(require_ready),
) -> dict:
    """指定文書のチャンク本文一覧（ページング）。focus指定時は該当ページを返す。"""
    return list_document_chunks(st.settings.pg_conn, st.settings.pg_collection,
                                source, limit=limit, offset=offset, focus_id=focus)
