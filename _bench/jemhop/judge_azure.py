#!/usr/bin/env python
"""JEMHopQA予測をAzure gpt-4.1-miniで採点（string-match採点のYES/NO過小評価を是正）。

短答(YES/NO・固有名詞・日付)を正解と照合。type別内訳付き。
Usage:
    python _bench/jemhop/judge_azure.py --pred _bench/jemhop/pred_jemhop_kg.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))
from dotenv import load_dotenv
load_dotenv()

PROMPT = """質問への「回答」が「正解」と本質的に一致するか判定してください。
YES/NO質問では肯定/否定が合っていれば正解。固有名詞・日付は表記揺れを許容し意味一致で判定。
出力は1(正解) か 0(不正解) の1文字のみ。

質問: {q}
正解: {gold}
回答: {pred}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    items = [d for d in data if d.get("success")]

    from graphrag_core.llm.factory import create_chat_llm
    llm = create_chat_llm(temperature=0, max_tokens=4)
    try:
        llm.request_timeout = 60
        llm.max_retries = 3
    except Exception:
        pass

    def judge(d):
        try:
            r = llm.invoke(PROMPT.format(q=d["question"], gold=d.get("gold_answer", ""),
                                         pred=d.get("predicted_answer", "")[:400]))
            return 1 if r.content.strip().startswith("1") else 0
        except Exception:
            return -1

    with ThreadPoolExecutor(max_workers=4) as ex:
        scores = list(ex.map(judge, items))

    by_type = defaultdict(lambda: [0, 0])
    ok = n = 0
    for d, s in zip(items, scores):
        if s < 0:
            continue
        n += 1
        ok += s
        by_type[d.get("type", "?")][0] += s
        by_type[d.get("type", "?")][1] += 1

    print(f"=== {Path(args.pred).name} (Azure judge) ===")
    print(f"  回答 accuracy: {ok}/{n} = {100*ok/n:.1f}%")
    for t, (c, cnt) in sorted(by_type.items()):
        print(f"    {t:14s}: {c}/{cnt} ({100*c/cnt:.0f}%)")
    # 参照カバレッジ(決定的)も再掲
    ref_full = sum(1 for d in items if set(d.get("gold_references") or [])
                   and set(d["gold_references"]) <= set(d.get("predicted_references") or []))
    print(f"  参照 full-coverage: {ref_full}/{len(items)} = {100*ref_full/len(items):.1f}%")


if __name__ == "__main__":
    main()
