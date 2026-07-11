"""専門用語辞書（名寄せ）エンドポイント。"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.deps import require_ready
from api.jobs import JobBusy
from api.state import AppState
from graphrag_core.services.dictionary import (
    dictionary_report, save_dictionary_file)

router = APIRouter(prefix="/api/dictionary", tags=["dictionary"])


class DictEntry(BaseModel):
    canonical: str = Field(min_length=1)
    aliases: List[str] = []
    category: Optional[str] = ""
    definition: Optional[str] = ""


class DictSaveBody(BaseModel):
    entries: List[DictEntry] = Field(min_length=1)


class DictApplyBody(BaseModel):
    merge: bool = True


@router.get("")
def get_dictionary(st: AppState = Depends(require_ready)) -> dict:
    """辞書エントリ一覧＋グラフ内マッチ状況（統合候補/一致/未マッチ）。"""
    return dictionary_report(st.graph, st.settings.pg_collection)


@router.put("")
def put_dictionary(body: DictSaveBody, st: AppState = Depends(require_ready)) -> dict:
    """辞書を保存（.bakバックアップ付き）。"""
    try:
        return save_dictionary_file([e.model_dump() for e in body.entries],
                                    st.settings.pg_collection)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/apply", status_code=202)
def apply_dictionary_endpoint(body: DictApplyBody = DictApplyBody(),
                              st: AppState = Depends(require_ready)) -> dict:
    """辞書適用ジョブ（名寄せ統合→プロパティ付与→search_keys再計算）。LLM不要。"""
    graph = st.graph
    collection = st.settings.pg_collection

    def run(progress, should_cancel):
        from graphrag_core.services.dictionary import apply_dictionary_full
        result = apply_dictionary_full(graph, collection, merge=body.merge,
                                       progress=progress, should_cancel=should_cancel)
        st.invalidate_retrieval()
        return result

    try:
        job = st.jobs.submit("dict_apply", run)
    except JobBusy as e:
        raise HTTPException(409, {"message": str(e), "running_job_id": e.running_job_id})
    return {"job_id": job.id}
