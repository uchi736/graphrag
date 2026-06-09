#!/usr/bin/env python
"""Fujitsu RAG Hard Bench の予測結果を、YAML の 4軸メタデータでスライス集計

各軸の各値について、条件別 (hybrid / rerank / KG など) の回答精度を比較し、
「KG が効くカテゴリ / 効かないカテゴリ」を可視化する。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]


def load_tasks_meta(yaml_path: Path) -> list[dict]:
    import yaml
    tasks = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["tasks"]

    flat = []
    for t in tasks:
        meta = {
            "question": t["question"],
            "question_type": t.get("question_type", ""),
            "retrieval_level": t.get("retrieval_level", ""),
            "answer_level": t.get("answer_level", ""),
            "tag": t.get("tag", ""),
        }

        # Reasoning Complexity
        rc = t.get("Reasoning Complexity", {}) or {}
        meta["reasoning_depth"] = (rc.get("Reasoning Depth (Multi-step Reasoning)", {}) or {}).get("value")
        meta["quantitative"] = (rc.get("Quantitative Operation", {}) or {}).get("value")
        meta["negation"] = (rc.get("Negation Question", {}) or {}).get("value")
        meta["cause_effect"] = (rc.get("Cause and Effect", {}) or {}).get("value")
        meta["comparison"] = (rc.get("Comparison (and Conditional Judgment)", {}) or {}).get("value")
        meta["temporal"] = (rc.get("Temporal Specification", {}) or {}).get("value")
        meta["output_type"] = (rc.get("Type of Output Processing", {}) or {}).get("value")

        # Retrieval Difficulty
        rd = t.get("Retrieval Difficulty", {}) or {}
        meta["multi_document"] = (rd.get("multi-document", {}) or {}).get("value")
        meta["multi_chunk"] = (rd.get("multi-chunk", {}) or {}).get("value")
        meta["low_locality"] = (rd.get("Low Locality", {}) or {}).get("value")
        meta["remote_reference"] = (rd.get("Remote Reference", {}) or {}).get("value")
        meta["abstraction_discrepancy"] = (rd.get("Abstraction Discrepancy", {}) or {}).get("value")
        meta["vocabulary_mismatch"] = (rd.get("Vocabulary Mismatch", {}) or {}).get("value")

        # Source Structure & Modality
        ss = t.get("Source Structure & Modality", {}) or {}
        meta["tables_charts"] = (ss.get("Tables/Charts", {}) or {}).get("value")
        meta["complex_layout"] = (ss.get("Complex Layout", {}) or {}).get("value")
        meta["specific_area_ref"] = (ss.get("Specific Area Reference", {}) or {}).get("value")
        meta["logical_nesting"] = (ss.get("Logical Nesting", {}) or {}).get("value")
        meta["large_enumeration"] = (ss.get("Large Enumeration", {}) or {}).get("value")
        meta["redundancy"] = (ss.get("Redundancy", {}) or {}).get("value")

        # Explainability
        ex = t.get("Explainability Requirement", {}) or {}
        meta["explainability"] = (ex.get("Strictness of Evidence Presentation", {}) or {}).get("value")

        flat.append(meta)
    return flat


def load_eval(path: Path) -> dict[int, int]:
    """eval JSON から question_number(1-indexed) → 0/1 (correct/incorrect)"""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for d in data.get("details", []):
        qn = d["question_number"]
        out[qn] = 1 if d["final_evaluation"] == "correct" else 0
    return out


def crosstab(meta: list[dict], evals: dict[str, dict[int, int]], axis: str):
    """軸 axis の値別に、各条件の accuracy を集計"""
    bucket = defaultdict(lambda: {cond: [0, 0] for cond in evals})  # value -> cond -> [correct, total]
    for i, m in enumerate(meta, start=1):
        v = m.get(axis)
        if v is None:
            continue
        # 値を文字列正規化
        key = str(v) if not isinstance(v, str) else v
        for cond, ev in evals.items():
            if i in ev:
                bucket[key][cond][0] += ev[i]
                bucket[key][cond][1] += 1
    return bucket


def fmt_row(label: str, conds: list[str], bucket_row: dict, ref_cond: str | None) -> str:
    parts = [f"  {label:25s}"]
    n_max = max((bucket_row[c][1] for c in conds), default=0)
    parts.append(f"N={n_max:3d}  ")
    accs = {}
    for c in conds:
        cor, tot = bucket_row[c]
        if tot == 0:
            parts.append(f"{c}:  ---  ")
            continue
        acc = cor / tot * 100
        accs[c] = acc
        parts.append(f"{c}:{acc:5.1f}% ")
    if ref_cond and ref_cond in accs and len(accs) > 1:
        # 最右 vs ref_cond の差
        last_cond = conds[-1]
        if last_cond != ref_cond and last_cond in accs:
            delta = accs[last_cond] - accs[ref_cond]
            parts.append(f"  Δ({last_cond}−{ref_cond})={delta:+.1f}")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="../Fujitsu-RAG-Hard-Benchmark/dataset/FJ_KGQA_Hard.yaml")
    ap.add_argument("--evals", nargs="+", required=True,
                    help="cond_name:path/to/eval_result.json の複数指定")
    ap.add_argument("--ref", default=None, help="差分計算の基準条件名 (デフォルト: 1番目)")
    args = ap.parse_args()

    yaml_path = Path(args.yaml)
    if not yaml_path.is_absolute():
        yaml_path = (_proj / args.yaml).resolve()
    meta = load_tasks_meta(yaml_path)
    print(f"Tasks: {len(meta)}")

    evals: dict[str, dict[int, int]] = {}
    for spec in args.evals:
        name, path = spec.split(":", 1)
        evals[name] = load_eval(Path(path))
    conds = list(evals.keys())
    ref = args.ref or conds[0]
    print(f"Conditions: {conds} (ref={ref})")
    print()

    # 全体 accuracy
    print("=" * 90)
    print("Overall")
    print("=" * 90)
    for c, ev in evals.items():
        if ev:
            acc = sum(ev.values()) / len(ev) * 100
            print(f"  {c:12s}  {acc:5.1f}%  ({sum(ev.values())}/{len(ev)})")
    print()

    # 軸ごと
    axes_groups = [
        ("Reasoning Complexity", [
            "reasoning_depth", "quantitative", "negation",
            "cause_effect", "comparison", "temporal", "output_type",
        ]),
        ("Retrieval Difficulty", [
            "multi_document", "multi_chunk", "low_locality",
            "remote_reference", "abstraction_discrepancy", "vocabulary_mismatch",
        ]),
        ("Source Structure & Modality", [
            "tables_charts", "complex_layout", "specific_area_ref",
            "logical_nesting", "large_enumeration", "redundancy",
        ]),
        ("Basic", ["question_type", "retrieval_level", "answer_level"]),
        ("Explainability", ["explainability"]),
    ]

    for group_name, group_axes in axes_groups:
        print("=" * 90)
        print(f"{group_name}")
        print("=" * 90)
        for axis in group_axes:
            bucket = crosstab(meta, evals, axis)
            if not bucket:
                continue
            print(f"\n[{axis}]")
            for value in sorted(bucket.keys()):
                print(fmt_row(f"{value}", conds, bucket[value], ref))
        print()


if __name__ == "__main__":
    main()
