#!/usr/bin/env python
"""J-RAGBench (neoai-inc/Japanese-RAG-Generator-Benchmark) を Gemma 4 で回す

Retrieverを切り離して「Generator LLM」だけの精度を測る。
positive + negative を context として渡し、回答生成 → LLM-as-judgeで採点。

Usage:
    python _bench/run_jragbench.py                          # 全114問
    python _bench/run_jragbench.py --limit 20                # 先頭20問
    python _bench/run_jragbench.py --no-negatives             # クリーン条件 (positiveのみ)
    python _bench/run_jragbench.py --concurrency 8            # 並列実行
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from graphrag_core.config import reset_settings
from graphrag_core.llm.factory import create_chat_llm


GENERATION_PROMPT = """あなたは与えられた【参考文書】に基づいて質問に答えるアシスタントです。

【参考文書】
{context}

【質問】
{question}

【回答ルール】
- 参考文書に書かれている内容だけを根拠に答えてください
- 参考文書に答えるための情報が無い場合は「回答に必要な情報源が存在しないため回答不可」と答えてください
- 推測や一般常識による補完はしないでください
- 回答は簡潔に、結論を最初に述べてください

【回答】"""


def categorize(example: dict) -> str:
    """ヒューリスティック分類: 拒否 / Table / Integration / Reasoning / Single"""
    n_pos = len(example.get("positive") or [])
    answer = example.get("answer", "")

    if n_pos == 0 or "回答不可" in answer or "情報源が存在しない" in answer:
        return "Refusal"

    pos_text = "\n".join(example.get("positive") or [])
    # 表(Markdown table): | が連続する行が3行以上
    table_lines = sum(1 for line in pos_text.split("\n") if line.count("|") >= 3)
    if table_lines >= 3:
        return "Table"

    # 多段推論っぽい: positive 4件以上 OR 数値計算/年代計算が必要っぽい質問
    if n_pos >= 4 or re.search(r"(西暦|何年|何歳|何%|計算|合計|平均|差|引き算|割り算)", example.get("question", "")):
        return "Reasoning"

    if n_pos >= 2:
        return "Integration"

    return "Single"


def build_context(example: dict, include_negatives: bool) -> str:
    """positive + (optional) negative を結合したcontextテキスト"""
    pos = example.get("positive") or []
    neg = example.get("negative") or [] if include_negatives else []
    # 順序: pos先頭、neg後ろ (本番想定では retriever がスコアでまぜる)
    parts = []
    idx = 1
    for p in pos:
        parts.append(f"[文書{idx}]\n{p}")
        idx += 1
    for n in neg:
        parts.append(f"[文書{idx}]\n{n}")
        idx += 1
    return "\n\n".join(parts)


def run_one(example: dict, idx: int, llm, include_negatives: bool) -> dict:
    """1問処理して回答を返す"""
    ctx = build_context(example, include_negatives)
    prompt = GENERATION_PROMPT.format(context=ctx, question=example["question"])
    t0 = time.time()
    try:
        resp = llm.invoke(prompt)
        answer = resp.content.strip()
        err = ""
    except Exception as e:
        answer = ""
        err = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0

    return {
        "idx": idx,
        "category": categorize(example),
        "n_positive": len(example.get("positive") or []),
        "n_negative": len(example.get("negative") or []),
        "question": example["question"],
        "gold": example["answer"],
        "prediction": answer,
        "error": err,
        "elapsed_sec": round(elapsed, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="先頭N問だけ実行 (デバッグ用)")
    ap.add_argument("--no-negatives", action="store_true", help="positiveのみcontextに含める (クリーン条件)")
    ap.add_argument("--concurrency", type=int, default=4, help="vLLM並列リクエスト数")
    ap.add_argument("--output", default="_bench/jragbench_predictions.csv", help="予測CSVの出力先")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("neoai-inc/Japanese-RAG-Generator-Benchmark", split="train")
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"=== J-RAGBench Generator評価 ===")
    print(f"  問題数: {n}/{len(ds)}")
    print(f"  context: positive + {'negative' if not args.no_negatives else '(negative無し: クリーン条件)'}")
    print(f"  並列度: {args.concurrency}")

    reset_settings()
    llm = create_chat_llm(temperature=0)

    results = [None] * n
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, ds[i], i, llm, not args.no_negatives): i for i in range(n)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {"idx": i, "error": f"executor: {e}",
                              "question": ds[i]["question"], "gold": ds[i]["answer"],
                              "prediction": "", "category": categorize(ds[i]),
                              "n_positive": len(ds[i].get("positive") or []),
                              "n_negative": len(ds[i].get("negative") or []),
                              "elapsed_sec": 0.0}
            done += 1
            if done % 10 == 0 or done == n:
                avg = (time.time() - t_start) / done
                eta = avg * (n - done)
                print(f"  {done}/{n}  avg={avg:.1f}s ETA={eta:.0f}s")

    cols = ["idx", "category", "n_positive", "n_negative", "elapsed_sec", "question", "gold", "prediction", "error"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            if r:
                w.writerow({k: r.get(k, "") for k in cols})

    total = time.time() - t_start
    print(f"\n完了: {total:.1f}秒, {args.output}")

    # カテゴリ分布
    from collections import Counter
    cat_dist = Counter(r["category"] for r in results if r)
    print(f"\nカテゴリ分布: {dict(cat_dist)}")


if __name__ == "__main__":
    main()
