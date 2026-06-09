"""Cross-encoder reranker client (vLLM /v1/score 互換)

vLLM上の reranker モデル（BAAI/bge-reranker-v2-m3 等）を HTTP で呼び、
(query, candidate) ペアの関連度スコアを返す。LLM rerank より 10-100倍速い。
"""

from __future__ import annotations

import logging
from typing import List, Optional

import requests

from graphrag_core.config import get_settings

logger = logging.getLogger(__name__)


def is_reranker_enabled() -> bool:
    s = get_settings()
    return bool(s.reranker_enabled and s.vllm_reranker_endpoint)


def score_candidates(query: str, candidates: List[str], timeout: float = 30.0) -> Optional[List[float]]:
    """各候補の関連度スコア（大きいほど関連あり）を返す

    Returns:
        List[float]: candidates と同じ長さ。失敗時は None。
    """
    if not candidates:
        return []

    s = get_settings()
    url = s.vllm_reranker_endpoint.rstrip("/") + "/score"
    headers = {"Content-Type": "application/json"}
    if s.vllm_reranker_api_key and s.vllm_reranker_api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {s.vllm_reranker_api_key}"

    payload = {
        "model": s.vllm_reranker_model,
        "text_1": query,
        "text_2": list(candidates),
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()

        # vLLM の /v1/score レスポンス: data[i].score (index順とは限らないのでsort)
        entries = data.get("data", [])
        if not entries or len(entries) != len(candidates):
            logger.warning("reranker response length mismatch: got %d, expected %d",
                           len(entries), len(candidates))
            return None

        scores = [0.0] * len(candidates)
        for entry in entries:
            idx = entry.get("index")
            score = entry.get("score")
            if idx is None or score is None:
                continue
            if 0 <= idx < len(candidates):
                scores[idx] = float(score)
        return scores
    except Exception as e:
        logger.warning("reranker scoring failed: %s", e)
        return None


def rerank_by_score(query: str, items: list, text_fn, top_k: int) -> list:
    """items を reranker でスコアリングして上位 top_k を返す

    Args:
        query: 検索クエリ
        items: 任意のオブジェクトのリスト
        text_fn: items[i] -> テキスト表現 に変換する関数
        top_k: 上位件数

    Returns:
        items から上位 top_k を抜き出したリスト（スコア降順）
        reranker失敗時は items[:top_k] を返す
    """
    if not items:
        return []
    texts = [text_fn(x) for x in items]
    scores = score_candidates(query, texts)
    if scores is None:
        return items[:top_k]
    ranked = sorted(zip(scores, items), key=lambda t: -t[0])
    return [x for _, x in ranked[:top_k]]
