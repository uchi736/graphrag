"""QA サービス: 検索（retriever_and_merge）+ 回答生成の一元実装。

ui/system.py の restore版/build版に重複していたネストクロージャ
（retriever_and_merge ラッパ + generate_with_sources）をここに統合する。
- Streamlit からは make_qa_chain()（LCEL Runnable、現行 chain と同形）
- FastAPI からは answer_question()（一括） / answer_question_events()（SSE用）

graph_paths は常に含める（restore版のみ返していた差異を統一。
qa_tab.py は .get("graph_paths", []) 読みのため後方互換）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from graphrag_core.llm.factory import create_chat_llm
from graphrag_core.llm.langfuse_utils import get_langfuse_callback, observe
from graphrag_core.prompts import QA_PROMPT
from graphrag_core.retrieval.pipeline import retriever_and_merge as _pipeline_retrieve
from graphrag_core.graph.provenance import graph_matches_collection
from graphrag_core.services.retrievers import create_vector_retriever


# ── 依存の束（Streamlit の ctx / FastAPI の AppState 双方から作れる） ──
@dataclass
class QADeps:
    graph: Any
    llm: Any
    embeddings: Any
    vector_store: Any
    pg_conn: str
    pg_collection: str


# ── KGゲート ──────────────────────────────────────────────────────
def gate_graph(deps: QADeps, config: Dict) -> tuple[Any, Optional[str]]:
    """KGを使うべきか判定し (graph|None, skip_reason) を返す。

    - config["enable_knowledge_graph"] が明示 False → スキップ
    - グラフ出自コレクションと現コレクションの不整合（provenance）→ スキップ
    """
    if deps.graph is None:
        return None, "graph_not_connected"
    if config.get("enable_knowledge_graph") is False:
        return None, "disabled_by_config"
    if not graph_matches_collection(deps.graph, deps.pg_collection):
        return None, "provenance_mismatch"
    return deps.graph, None


# ── 検索 ──────────────────────────────────────────────────────────
def _annotate_triple_types(graph, triples) -> None:
    """graph_sources の各トリプルに実ノードラベル start_type/end_type を付与する。

    QA参照グラフの色分け用（付与しないと全ノードが Unknown=灰色になる）。
    1クエリのみ・失敗しても検索結果には影響させない。
    """
    if graph is None or not triples:
        return
    ids = sorted({t.get(k) for t in triples if isinstance(t, dict)
                  for k in ("start", "end") if t.get(k)})
    if not ids:
        return
    try:
        rows = graph.query(
            "MATCH (n) WHERE n.id IN $ids RETURN n.id AS id, labels(n)[0] AS t",
            {"ids": ids}) or []
        tmap = {r["id"]: r["t"] for r in rows if r.get("t")}
        for t in triples:
            if not isinstance(t, dict):
                continue
            if t.get("start") in tmap:
                t["start_type"] = tmap[t["start"]]
            if t.get("end") in tmap:
                t["end_type"] = tmap[t["end"]]
    except Exception:
        pass


def run_retrieval(question: str, deps: QADeps, config: Dict) -> Dict:
    """検索本体。graphゲート + retriever生成込みで pipeline を呼ぶ。

    返り値は pipeline の dict に kg_used / kg_skip_reason を追加したもの。
    """
    graph, skip_reason = gate_graph(deps, config)
    top_k = int(config.get("retrieval_top_k") or 5)
    vector_retriever = create_vector_retriever(deps.vector_store, top_k)
    result = _pipeline_retrieve(
        question, graph, deps.llm, deps.embeddings, vector_retriever,
        deps.pg_conn, deps.pg_collection, config,
    )
    result["kg_used"] = graph is not None
    result["kg_skip_reason"] = skip_reason
    _annotate_triple_types(graph, result.get("graph_sources") or [])
    return result


# ── 回答生成 ──────────────────────────────────────────────────────
def make_answer_chain(llm=None):
    """QA_PROMPT | LLM | StrOutputParser の生成チェーン。"""
    prompt = PromptTemplate.from_template(QA_PROMPT)
    return prompt | (llm or create_chat_llm(temperature=0)) | StrOutputParser()


@observe(name="answer_generation")
def generate_with_sources(data: Dict, llm_chain=None) -> Dict:
    """検索結果 dict から回答を生成し、根拠と共に返す（旧2クロージャの統一版）。"""
    chain = llm_chain or make_answer_chain()
    answer = chain.invoke(
        {"question": data["question"], "context": data["context"]},
        config=get_langfuse_callback(),
    )
    return {
        "answer": answer,
        "vector_sources": data.get("vector_sources", []),
        "kg_source_chunks": data.get("kg_source_chunks", []),
        "graph_sources": data.get("graph_sources", []),
        "graph_paths": data.get("graph_paths", []),
        "extracted_entities": data.get("extracted_entities", {}),
        "kg_used": data.get("kg_used", False),
        "kg_skip_reason": data.get("kg_skip_reason"),
    }


# ── 一括実行（非ストリーミング） ──────────────────────────────────
def answer_question(question: str, deps: QADeps, config: Dict,
                    llm_chain=None) -> Dict:
    data = run_retrieval(question, deps, config)
    return generate_with_sources(data, llm_chain=llm_chain)


# ── SSE用イベント列（ストリーミング） ─────────────────────────────
@dataclass
class QAEvent:
    type: str          # meta | retrieval | token | done | error
    data: Dict


def _doc_to_dto(d) -> Dict:
    meta = getattr(d, "metadata", None) or {}
    return {
        "id": meta.get("id"),
        "source": meta.get("source"),
        "page": meta.get("page"),
        "text": getattr(d, "page_content", str(d)),
    }


def serialize_qa_result(data: Dict, include_answer: bool = True) -> Dict:
    """pipeline/generate の結果を JSON安全な DTO に変換する。"""
    out = {
        "vector_sources": [_doc_to_dto(d) for d in data.get("vector_sources", [])],
        "kg_source_chunks": [_doc_to_dto(d) for d in data.get("kg_source_chunks", [])],
        "graph_sources": [
            {"start": t.get("start"), "type": t.get("type"), "end": t.get("end"),
             "start_type": t.get("start_type"), "end_type": t.get("end_type")}
            for t in data.get("graph_sources", []) if isinstance(t, dict)
        ],
        "graph_paths": [
            (p if isinstance(p, dict) else {"path_text": str(p)})
            for p in data.get("graph_paths", [])
        ],
        "extracted_entities": data.get("extracted_entities", {}),
        "kg_used": data.get("kg_used", False),
        "kg_skip_reason": data.get("kg_skip_reason"),
    }
    if include_answer:
        out["answer"] = data.get("answer", "")
    return out


def answer_question_events(question: str, deps: QADeps, config: Dict,
                           llm_chain=None) -> Iterator[QAEvent]:
    """SSE用: meta → retrieval(全根拠) → token(差分) → done | error。

    sync generator（FastAPI の StreamingResponse が threadpool で回す）。
    """
    t0 = time.time()
    try:
        yield QAEvent("meta", {
            "question": question,
            "effective_config": {
                k: config.get(k) for k in (
                    "retrieval_top_k", "search_mode", "enable_rerank",
                    "enable_japanese_search", "enable_knowledge_graph",
                    "include_kg_source_chunks", "graph_hop_count",
                    "enable_entity_vector", "entity_similarity_threshold",
                )
            },
        })
        data = run_retrieval(question, deps, config)
        t_retrieval = time.time() - t0
        yield QAEvent("retrieval", serialize_qa_result(data, include_answer=False))
    except Exception as e:
        yield QAEvent("error", {"stage": "retrieval", "message": f"{type(e).__name__}: {e}"})
        return

    try:
        chain = llm_chain or make_answer_chain()
        t1 = time.time()
        parts = []
        for token in chain.stream(
            {"question": data["question"], "context": data["context"]},
            config=get_langfuse_callback(),
        ):
            parts.append(token)
            yield QAEvent("token", {"delta": token})
        yield QAEvent("done", {
            "answer": "".join(parts),
            "timing_ms": {
                "retrieval": round(t_retrieval * 1000),
                "generation": round((time.time() - t1) * 1000),
            },
        })
    except Exception as e:
        yield QAEvent("error", {"stage": "generation", "message": f"{type(e).__name__}: {e}"})


# ── Streamlit 互換チェーン ────────────────────────────────────────
def make_qa_chain(deps: QADeps, config_provider: Callable[[], Dict]):
    """現行 st.session_state.chain と同形の LCEL Runnable を返す。

    config_provider はクエリ毎に呼ばれる（ctx.build_config を渡すことで
    「設定変更が再構築なしで即反映」の現行挙動を維持）。
    """
    llm_chain = make_answer_chain()

    def _retrieve(question: str):
        return run_retrieval(question, deps, config_provider())

    def _generate(data: Dict):
        return generate_with_sources(data, llm_chain=llm_chain)

    return RunnablePassthrough() | RunnableLambda(_retrieve) | RunnableLambda(_generate)
