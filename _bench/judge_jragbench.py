#!/usr/bin/env python
"""J-RAGBench予測結果を LLM-as-judge で採点 + カテゴリ別集計

Usage:
    python _bench/judge_jragbench.py
    python _bench/judge_jragbench.py --input _bench/jragbench_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from graphrag_core.config import reset_settings
from graphrag_core.llm.factory import create_chat_llm


JUDGE_PROMPT = """あなたは質問応答の採点者です。「正解」を基準に「モデル回答」を採点してください。

【質問】
{question}

【正解】
{gold}

【モデル回答】
{prediction}

【カテゴリ】
{category}

【採点ルール】
- 0: 不正解 (正解と矛盾、または見当違いの回答)
- 1: 部分正解 (本質的な要素の一部欠落、または余計な情報を含む)
- 2: 完全正解 (意味的に正解と同等)

【重要】
- 「回答拒否」カテゴリ (正解が「回答に必要な情報源が存在しないため回答不可」等):
    * モデル回答も拒否系なら 2
    * モデル回答が誤った具体回答を提示していれば 0 (幻覚)
- 「回答拒否ではないカテゴリ」 (正解が具体的な内容):
    * モデル回答が「回答不可」「情報がない」等の拒否なら 0 (情報あるのに見逃した)
    * 部分的に正しいが核心が外れていれば 1
- 文言の完全一致は不要。意味が同じなら正解扱い

JSON 1行で次の形式のみ返してください (前置きや説明は禁止):
{{"score": 0|1|2, "reason": "50字以内の日本語"}}
"""


def llm_judge(row: dict, llm) -> dict:
    """1問の採点"""
    prompt = JUDGE_PROMPT.format(
        question=row["question"], gold=row["gold"],
        prediction=row["prediction"] or "(空)",
        category=row["category"],
    )
    try:
        resp = llm.invoke(prompt)
        raw = resp.content.strip()
        m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", raw, re.DOTALL)
        if not m:
            return {"judge_score": "", "judge_reason": f"JSON抽出失敗: {raw[:80]}"}
        data = json.loads(m.group(0))
        score = int(data.get("score"))
        if score not in (0, 1, 2):
            return {"judge_score": "", "judge_reason": f"score範囲外: {score}"}
        return {"judge_score": str(score), "judge_reason": str(data.get("reason", "")).strip()[:80]}
    except Exception as e:
        return {"judge_score": "", "judge_reason": f"judge error: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="_bench/jragbench_predictions.csv")
    ap.add_argument("--output", default="_bench/jragbench_judged.csv")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"採点開始: {len(rows)}問, 並列={args.concurrency}")

    reset_settings()
    llm = create_chat_llm(temperature=0)

    t0 = time.time()
    judged = [None] * len(rows)

    def _work(i):
        r = dict(rows[i])
        out = llm_judge(r, llm)
        r.update(out)
        return i, r

    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(_work, i) for i in range(len(rows))]
        for fut in as_completed(futs):
            i, r = fut.result()
            judged[i] = r
            done += 1
            if done % 20 == 0 or done == len(rows):
                avg = (time.time() - t0) / done
                eta = avg * (len(rows) - done)
                print(f"  {done}/{len(rows)}  avg={avg:.1f}s ETA={eta:.0f}s")

    cols = list(rows[0].keys()) + ["judge_score", "judge_reason"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in judged:
            w.writerow({k: r.get(k, "") for k in cols})

    # 集計
    print()
    print(f"=== Overall ===")
    scores_all = [int(r["judge_score"]) for r in judged if r["judge_score"].isdigit()]
    judge_ok = len(scores_all)
    print(f"  採点成功: {judge_ok}/{len(rows)}")
    print(f"  Avg score: {sum(scores_all)/len(scores_all):.2f} / 2.0")
    sc_dist = Counter(scores_all)
    print(f"  Score分布: 2={sc_dist[2]} (完全正解={sc_dist[2]/len(rows)*100:.1f}%), "
          f"1={sc_dist[1]}, 0={sc_dist[0]}, judge失敗={len(rows)-judge_ok}")

    # カテゴリ別
    print()
    print(f"=== Category breakdown ===")
    cat_groups = defaultdict(list)
    for r in judged:
        if r["judge_score"].isdigit():
            cat_groups[r["category"]].append(int(r["judge_score"]))
    print(f"  {'Category':12s} {'N':>4s} {'Avg':>6s} {'Full正':>8s} {'部分':>6s} {'誤':>4s}")
    for cat, scores in sorted(cat_groups.items()):
        full = sum(1 for s in scores if s == 2)
        partial = sum(1 for s in scores if s == 1)
        wrong = sum(1 for s in scores if s == 0)
        avg = sum(scores) / len(scores)
        print(f"  {cat:12s} {len(scores):4d} {avg:6.2f} {full:4d}({full/len(scores)*100:.0f}%) {partial:4d} {wrong:4d}")

    print(f"\n出力: {args.output}")


if __name__ == "__main__":
    main()
