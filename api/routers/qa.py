"""QA エンドポイント（一括 / SSEストリーミング）。"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.deps import require_ready
from api.sse import sse_response
from api.state import AppState
from graphrag_core.config import build_pipeline_config
from graphrag_core.services.qa import (
    QADeps,
    answer_question,
    answer_question_events,
    serialize_qa_result,
)

router = APIRouter(prefix="/api", tags=["qa"])


class QAConfig(BaseModel):
    """UI検索設定9キー。None は Settings 既定を使う（build_pipeline_config が None-skip）。"""
    retrieval_top_k: Optional[int] = Field(None, ge=1, le=20)
    enable_rerank: Optional[bool] = None
    enable_japanese_search: Optional[bool] = None
    search_mode: Optional[Literal["hybrid", "vector", "keyword"]] = None
    enable_knowledge_graph: Optional[bool] = None
    include_kg_source_chunks: Optional[bool] = None
    graph_hop_count: Optional[int] = Field(None, ge=1, le=3)
    enable_entity_vector: Optional[bool] = None
    entity_similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class QARequest(BaseModel):
    question: str = Field(min_length=1)
    config: QAConfig = QAConfig()


def _deps_and_config(st: AppState, req: QARequest) -> tuple[QADeps, dict]:
    deps = QADeps(
        graph=st.graph,
        llm=st.llm,
        embeddings=st.embeddings,
        vector_store=st.vector_store,
        pg_conn=st.settings.pg_conn,
        pg_collection=st.settings.pg_collection,
    )
    config = build_pipeline_config(st.settings, **req.config.model_dump())
    return deps, config


@router.post("/qa")
def qa(req: QARequest, st: AppState = Depends(require_ready)) -> dict:
    """一括QA（テスト・バッチ・外部連携用）。"""
    deps, config = _deps_and_config(st, req)
    result = answer_question(req.question, deps, config)
    return serialize_qa_result(result)


@router.post("/qa/stream")
def qa_stream(req: QARequest, st: AppState = Depends(require_ready)):
    """SSEストリーミングQA。

    イベント順序: meta → retrieval（全根拠先行）→ token（差分×N）→ done | error
    """
    deps, config = _deps_and_config(st, req)
    events = answer_question_events(req.question, deps, config)
    return sse_response((ev.type, ev.data) for ev in events)
