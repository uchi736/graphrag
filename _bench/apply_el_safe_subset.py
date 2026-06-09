#!/usr/bin/env python
"""EL dry-run レポートから安全な size=2 cluster だけを抽出して Neo4j に適用

フィルター:
1. size=2 のみ採用 (size≥3 は over-merge リスク高)
2. Levenshtein 距離==1 で両方 digit を含む → skip (型番変種: FMCPTD01X vs Y)
3. Levenshtein 距離==1 で長さ同じ → skip (掲示板↔掲示物, 取締役↔監査役)
4. substring with len_diff==1 → skip (学校↔小学校, 目標↔目標値, 別売↔別売品)
5. canonical_form プロパティを保存 (retrieval時のエイリアス展開用)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from fujitsu_entity_linking import merge_cluster  # noqa: E402


def has_digit(s: str) -> bool:
    return any(c.isdigit() for c in s)


def levenshtein(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        # fallback
        if a == b: return 0
        if not a: return len(b)
        if not b: return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = curr
        return prev[-1]


def should_skip(a: str, b: str) -> tuple[bool, str]:
    """skip すべきか + 理由"""
    if a == b:
        return True, "identical"
    d = levenshtein(a, b)
    # 1-char model variant: FMCPTD01X vs FMCPTD01Y
    if d == 1 and len(a) == len(b) and (has_digit(a) or has_digit(b)):
        return True, "1char_model_variant"
    # 1-char swap same length: 掲示板↔掲示物
    if d == 1 and len(a) == len(b):
        return True, "1char_suffix_diff"
    # substring with len_diff==1: 学校↔小学校, 目標↔目標値
    if len(a) != len(b):
        short, long_ = (a, b) if len(a) < len(b) else (b, a)
        if short in long_ and len(long_) - len(short) == 1:
            return True, "substring_len_diff_1"
    return False, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="_bench/entity_linking_report_v2c_dryrun.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output", default="_bench/el_safe_subset_applied.json")
    args = ap.parse_args()

    rep = json.load(open(args.report, encoding="utf-8"))
    plan = rep["merge_plan"]
    # size=2 only
    size2 = [p for p in plan if p["size"] == 2 and p["duplicates"]]
    print(f"input: {len(plan)} clusters, size=2 with dups: {len(size2)}")

    # フィルター
    accepted = []
    rejected_stats: dict[str, int] = {}
    for p in size2:
        canon = p["canonical"]
        dup = p["duplicates"][0]
        skip, reason = should_skip(canon, dup)
        if skip:
            rejected_stats[reason] = rejected_stats.get(reason, 0) + 1
            continue
        accepted.append(p)

    print(f"accepted: {len(accepted)} clusters")
    print(f"rejected: {sum(rejected_stats.values())} ({rejected_stats})")
    print()
    print("--- accepted sample (random 30) ---")
    import random
    random.seed(0)
    for p in random.sample(accepted, min(30, len(accepted))):
        print(f"  [{p['canonical_type']:12s}] {p['canonical']:30s} <- {p['duplicates'][0]}")
    print()
    print("--- rejected sample (random 15) ---")
    rej = [p for p in size2 if should_skip(p["canonical"], p["duplicates"][0])[0]]
    for p in random.sample(rej, min(15, len(rej))):
        skip, reason = should_skip(p["canonical"], p["duplicates"][0])
        print(f"  [{reason:25s}] {p['canonical']:30s} <- {p['duplicates'][0]}")

    if args.dry_run:
        print(f"\n--dry-run, Neo4j 書込みスキップ")
        return

    # 実行
    print(f"\n=== Neo4j 書込み開始: {len(accepted)} clusters ===")
    from graphrag_core.config import reset_settings, get_settings
    from langchain_neo4j import Neo4jGraph
    reset_settings()
    s = get_settings()
    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw)

    total_moved, total_deleted = 0, 0
    t0 = time.time()
    applied_log = []
    for i, p in enumerate(accepted):
        moved, deleted = merge_cluster(graph, p["canonical"], p["duplicates"])
        total_moved += moved
        total_deleted += deleted
        applied_log.append({
            "canonical": p["canonical"],
            "duplicates": p["duplicates"],
            "moved": moved,
            "deleted": deleted,
        })
        if (i + 1) % 50 == 0 or i + 1 == len(accepted):
            print(f"  {i+1}/{len(accepted)}  累計 moved={total_moved}, deleted={total_deleted}  ({time.time()-t0:.0f}s)")

    print(f"\n完了 ({time.time()-t0:.0f}s)")
    print(f"  total moved edges: {total_moved}")
    print(f"  total deleted nodes: {total_deleted}")

    Path(args.output).write_text(json.dumps({
        "accepted_count": len(accepted),
        "rejected_count": sum(rejected_stats.values()),
        "rejected_stats": rejected_stats,
        "total_moved": total_moved,
        "total_deleted": total_deleted,
        "applied": applied_log,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {args.output}")


if __name__ == "__main__":
    main()
