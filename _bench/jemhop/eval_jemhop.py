#!/usr/bin/env python
"""JEMHopQA 予測の採点。

短答(実体名/YES-NO/数値)なので:
- answer: 正規化EM + 包含(gold⊆pred) + 文字F1。YES/NO は はい/いいえ/yes/no を正規化吸収。
- reference: gold page_ids(2件)を取得チャンクの doc_id がカバーできたか（full / partial）。
- type別(comparison/compositional)の内訳。

Usage:
    python _bench/jemhop/eval_jemhop.py --pred _bench/jemhop/pred_jemhop.json
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = re.sub(r"[\s　、。,.\"'’”｢｣「」『』（）\(\)・]+", "", s)
    # YES/NO 正規化
    yes = {"yes", "はい", "そうです", "正しい", "両方とも", "同じ"}
    no = {"no", "いいえ", "違います", "異なる", "ちがう"}
    if s in yes:
        return "__yes__"
    if s in no:
        return "__no__"
    return s


def char_f1(pred: str, gold: str) -> float:
    p, g = list(pred), list(gold)
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def gold_in_pred(gold_raw: str, pred_raw: str) -> bool:
    """正規化後に gold が pred に包含されるか（YES/NOは完全一致扱い）"""
    g, p = norm(gold_raw), norm(pred_raw)
    if g.startswith("__"):  # YES/NO
        # predの先頭付近で yes/no を判定
        return norm(pred_raw[:20]) == g or g.strip("_") in p[:30]
    return g in p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    items = [d for d in data if d.get("success")]

    em = contain = 0
    f1_sum = 0.0
    ref_full = ref_partial = 0
    by_type = defaultdict(lambda: [0, 0, 0])  # type -> [contain, ref_full, n]
    details = []
    for d in items:
        gold, pred = d.get("gold_answer", ""), d.get("predicted_answer", "")
        ng, np_ = norm(gold), norm(pred)
        is_em = (ng == np_) or (ng != "" and ng == norm(pred[:len(gold) + 5]))
        is_contain = gold_in_pred(gold, pred)
        f1 = char_f1(np_.strip("_"), ng.strip("_"))
        em += int(is_em)
        contain += int(is_contain)
        f1_sum += f1
        cref = set(d.get("gold_references") or [])
        pref = set(d.get("predicted_references") or [])
        full = bool(cref) and cref <= pref
        partial = bool(cref & pref)
        ref_full += int(full)
        ref_partial += int(partial)
        t = d.get("type", "?")
        by_type[t][0] += int(is_contain)
        by_type[t][1] += int(full)
        by_type[t][2] += 1
        details.append({"qid": d.get("qid"), "type": t, "contain": is_contain,
                        "em": is_em, "f1": round(f1, 3), "ref_full": full,
                        "gold": gold, "pred": pred[:80]})

    n = len(items)
    print(f"=== JEMHopQA dev eval (KG-off) ===")
    print(f"  N={n}")
    print(f"  Answer 包含(gold⊆pred): {contain}/{n} = {100*contain/n:.1f}%")
    print(f"  Answer 正規化EM       : {em}/{n} = {100*em/n:.1f}%")
    print(f"  Answer 文字F1平均      : {100*f1_sum/n:.1f}%")
    print(f"  参照 full-coverage(2件): {ref_full}/{n} = {100*ref_full/n:.1f}%")
    print(f"  参照 partial(1件以上)  : {ref_partial}/{n} = {100*ref_partial/n:.1f}%")
    print(f"  --- type別 (包含 / 参照full / N) ---")
    for t, (c, rf, cnt) in sorted(by_type.items()):
        print(f"    {t:14s}: 包含 {c}/{cnt} ({100*c/cnt:.0f}%) | 参照full {rf}/{cnt} ({100*rf/cnt:.0f}%)")

    out = Path(args.pred).with_suffix(".eval.json")
    out.write_text(json.dumps({
        "n": n, "contain_pct": round(100*contain/n, 1), "em_pct": round(100*em/n, 1),
        "f1_pct": round(100*f1_sum/n, 1), "ref_full_pct": round(100*ref_full/n, 1),
        "ref_partial_pct": round(100*ref_partial/n, 1), "details": details,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  detail: {out}")


if __name__ == "__main__":
    main()
