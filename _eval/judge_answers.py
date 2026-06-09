#!/usr/bin/env python
"""質問応答評価: key_facts 照合 + LLM-as-judge による採点

入力: batch_eval.py の出力CSV (`expected_answer`, `key_facts`, `question`, `answer` 列を含む)
出力: 採点済みCSV + サマリ

採点ルール:
  - key_facts (;区切り) の網羅率: 0..1
  - LLM-as-judge: 0..2 (0=誤り, 1=部分正解, 2=完全正解) + 根拠
  - これを掛け合わせて total スコアにせず、別々に報告
"""

from __future__ import annotations
import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from graphrag_core.config import get_settings


def normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+", "", s)


def fact_coverage(answer: str, key_facts: str) -> tuple[float, list[str], list[str]]:
    """key_facts のうち answer に出現しているものの割合"""
    facts = [f.strip() for f in key_facts.split(";") if f.strip()]
    if not facts:
        return 1.0, [], []
    ans_norm = normalize(answer)
    hit = [f for f in facts if normalize(f) in ans_norm]
    miss = [f for f in facts if normalize(f) not in ans_norm]
    return len(hit) / len(facts), hit, miss


JUDGE_PROMPT = """あなたは質問応答の採点者です。「期待される回答」を基準に「実際の回答」を採点してください。

【質問】
{question}

【期待される回答】
{expected}

【必須キーファクト (含まれていればよい情報、;区切り)】
{key_facts}

【実際の回答】
{answer}

【採点ルール】
- 0: 不正解、または期待される回答と無関係/矛盾
- 1: 部分正解。重要な要素の一部が欠落、または冗長で核心がぼやけている
- 2: 完全正解。期待される回答と同等の情報量と正確性

文言の完全一致は不要。意味的に同等であればOK。
言い回しの違いはペナルティにしない。
ただし、列挙すべきものが明確に欠けている、誤った断定がある場合は減点。

JSON で次の形式のみ返してください (説明や前置きは禁止):
{{"score": 0|1|2, "reason": "短い日本語の理由 (50字以内)"}}
"""


def llm_judge(question: str, expected: str, key_facts: str, answer: str, llm) -> tuple[Optional[int], str]:
    """LLMで採点。失敗時は (None, error_msg)"""
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate.from_template(JUDGE_PROMPT)
    chain = prompt | llm | StrOutputParser()
    try:
        raw = chain.invoke({
            "question": question,
            "expected": expected,
            "key_facts": key_facts,
            "answer": answer,
        })
        # JSON抽出 (LLMが余計な文字を付けても拾えるように)
        m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", raw, re.DOTALL)
        if not m:
            return None, f"JSON抽出失敗: {raw[:120]}"
        data = json.loads(m.group(0))
        score = int(data.get("score"))
        reason = str(data.get("reason", "")).strip()
        if score not in (0, 1, 2):
            return None, f"score範囲外: {score}"
        return score, reason
    except Exception as e:
        return None, f"judge error: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers_csv", help="batch_eval.py の出力CSV")
    ap.add_argument("-o", "--output", default=None, help="採点結果CSVの出力先")
    ap.add_argument("--no-llm-judge", action="store_true", help="LLM-as-judge をスキップ (key_facts照合のみ)")
    args = ap.parse_args()

    in_path = Path(args.answers_csv)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_judged.csv")

    rows = []
    with open(in_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for r in reader:
            rows.append(r)

    needed = {"question", "answer", "expected_answer", "key_facts"}
    missing = needed - set(cols)
    if missing:
        print(f"CSV に必要な列がありません: {missing}")
        sys.exit(1)

    llm = None
    if not args.no_llm_judge:
        from graphrag_core.llm.factory import create_chat_llm
        llm = create_chat_llm(temperature=0)

    out_cols = list(cols) + ["fact_coverage", "facts_hit", "facts_miss", "judge_score", "judge_reason"]

    print(f"採点開始: {len(rows)}問")
    sum_cov, sum_judge, judge_n = 0.0, 0, 0
    score_dist = {0: 0, 1: 0, 2: 0, None: 0}
    detail_rows = []

    for i, r in enumerate(rows, 1):
        cov, hit, miss = fact_coverage(r["answer"], r["key_facts"])
        r["fact_coverage"] = f"{cov:.2f}"
        r["facts_hit"] = "; ".join(hit)
        r["facts_miss"] = "; ".join(miss)
        sum_cov += cov

        if llm is not None:
            score, reason = llm_judge(
                r["question"], r["expected_answer"], r["key_facts"], r["answer"], llm,
            )
            r["judge_score"] = str(score) if score is not None else ""
            r["judge_reason"] = reason
            if score is not None:
                sum_judge += score
                judge_n += 1
                score_dist[score] += 1
            else:
                score_dist[None] += 1
        else:
            r["judge_score"] = ""
            r["judge_reason"] = ""

        detail_rows.append(r)
        print(f"  Q{i:2d} cov={cov:.2f} judge={r.get('judge_score','-')}  miss={r['facts_miss'][:40]}")

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        for r in detail_rows:
            w.writerow(r)

    print()
    print(f"=== Summary ({len(rows)}問) ===")
    print(f"  Key fact coverage 平均: {sum_cov / len(rows):.2%}")
    if judge_n:
        print(f"  Judge score 平均: {sum_judge / judge_n:.2f} / 2.0  (採点成功 {judge_n}/{len(rows)})")
        print(f"  内訳: score 2={score_dist[2]}件, 1={score_dist[1]}件, 0={score_dist[0]}件, 失敗={score_dist[None]}件")
    print(f"  出力: {out_path}")


if __name__ == "__main__":
    main()
