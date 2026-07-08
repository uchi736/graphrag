#!/usr/bin/env python
"""plant_v15 予測の採点。

- 回答評価: LLM judge（gemma @ JUDGE_BASE_URL、Fujitsu benchと同一プロンプト）
- 参照評価: doc_id カバレッジ（full-coverage = 正解doc_idを全て検索できたか / match-rate）
- 軸別スライス（retrieval_level / answer_level / kg_query_type）

Usage:
    JUDGE_BASE_URL=http://192.168.0.250:8000/v1 JUDGE_MODEL=google/gemma-4-26B-A4B-it \
      python _bench/plant/eval_plant.py --pred _bench/plant/pred_plant_retrieval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "google/gemma-4-31B-it")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://192.168.0.250:8000/v1")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "dummy")

PROMPT = """あなたは質問応答の採点官です。
与えられた「質問」と「正しい回答」を踏まえて、「回答」の「正確性」を判定してください。
判定結果は、正確なら"1" 不正確なら"0" のどちらか1語だけを出力してください。

質問: {question}
正しい回答: {reference_answer}
回答: {answer}
"""


def judge(items, backend="vllm"):
    if backend == "azure":
        from graphrag_core.llm.factory import create_chat_llm
        chat = create_chat_llm(temperature=0, max_tokens=8)
        try:
            chat.request_timeout = 60
            chat.max_retries = 3
        except Exception:
            pass
    else:
        chat = ChatOpenAI(model=JUDGE_MODEL, base_url=JUDGE_BASE_URL,
                          api_key=JUDGE_API_KEY, temperature=0, max_tokens=8)

    def one(it):
        try:
            r = chat.invoke(PROMPT.format(question=it["question"],
                                          reference_answer=it["correct_answer"],
                                          answer=it["predicted_answer"]))
            raw = r.content.strip()
            return 1 if raw.startswith("1") else (0 if raw.startswith("0") else (1 if "1" in raw[:5] else 0))
        except Exception:
            return -1

    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(one, items))


def ref_coverage(pred, correct):
    """doc_id full-coverage: 正解doc_idを全て予測が含むか"""
    pset = set(pred)
    if not correct:
        return None
    return 1.0 if all(c in pset for c in correct) else 0.0


def ref_match_rate(pred, correct):
    if not correct:
        return None
    pset = set(pred)
    return sum(1 for c in correct if c in pset) / len(correct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--judge-backend", choices=["vllm", "azure"], default="vllm",
                    help="vllm=gemma@JUDGE_BASE_URL, azure=gpt-4.1-mini(create_chat_llm)")
    args = ap.parse_args()

    data = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    items = [d for d in data if d.get("success")]
    scores = judge(items, backend=args.judge_backend)
    for it, sc in zip(items, scores):
        it["_judge"] = sc

    n = len(items)
    ans_ok = sum(1 for s in scores if s == 1)
    covs = [ref_coverage(it["predicted_references"], it["correct_references"]) for it in items]
    covs = [c for c in covs if c is not None]
    mrs = [ref_match_rate(it["predicted_references"], it["correct_references"]) for it in items]
    mrs = [m for m in mrs if m is not None]

    print(f"=== plant_v15 evaluation ({Path(args.pred).name}) ===")
    print(f"  judge: {JUDGE_MODEL}")
    print(f"  回答 accuracy      : {100*ans_ok/n:.1f}% ({ans_ok}/{n})")
    print(f"  参照 full-coverage : {100*sum(covs)/len(covs):.1f}% ({int(sum(covs))}/{len(covs)})")
    print(f"  参照 match-rate avg: {100*sum(mrs)/len(mrs):.1f}%")

    # 軸別スライス（回答accuracy）
    for axis in ("retrieval_level", "answer_level", "kg_query_type"):
        buckets = defaultdict(lambda: [0, 0])
        for it in items:
            v = it.get(axis)
            if v is None:
                continue
            buckets[str(v)][0] += 1 if it["_judge"] == 1 else 0
            buckets[str(v)][1] += 1
        if buckets:
            print(f"\n  [{axis}]")
            for k in sorted(buckets):
                ok, tot = buckets[k]
                print(f"    {k:14s} {100*ok/tot:5.1f}% ({ok}/{tot})")

    out = Path(args.pred).with_suffix(".eval.json")
    out.write_text(json.dumps({
        "answer_accuracy": round(100*ans_ok/n, 2),
        "ref_full_coverage": round(100*sum(covs)/len(covs), 2),
        "ref_match_rate": round(100*sum(mrs)/len(mrs), 2),
        "n": n,
        "details": [{"qa_id": it.get("qa_id"), "judge": it["_judge"],
                     "pred_refs": it["predicted_references"],
                     "correct_refs": it["correct_references"]} for it in items],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  detail: {out}")


if __name__ == "__main__":
    main()
