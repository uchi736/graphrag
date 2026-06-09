#!/usr/bin/env python
"""multi-hop / multi-document 質問の retrieval 失敗を診断

各 multi-hop 質問について:
- gold pages が index に存在するか
- vector検索で何位に出るか
- top-K (5, 10, 20, 50, 100) でのrecall
- 失敗パターンの分類

これで「何位まで広げれば取れるのか」「そもそも取れないのか」が分かる
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))


def is_multihop(t: dict) -> bool:
    rc = t.get('Reasoning Complexity', {}) or {}
    rd = t.get('Retrieval Difficulty', {}) or {}
    return (
        (rc.get('Reasoning Depth (Multi-step Reasoning)', {}) or {}).get('value') == 'multi'
        or (rd.get('multi-document', {}) or {}).get('value') is True
        or (rd.get('multi-chunk', {}) or {}).get('value') is True
    )


def classify_q(t: dict) -> dict:
    rc = t.get('Reasoning Complexity', {}) or {}
    rd = t.get('Retrieval Difficulty', {}) or {}
    return {
        'multi_step': (rc.get('Reasoning Depth (Multi-step Reasoning)', {}) or {}).get('value') == 'multi',
        'multi_doc': (rd.get('multi-document', {}) or {}).get('value') is True,
        'multi_chunk': (rd.get('multi-chunk', {}) or {}).get('value') is True,
        'remote_ref': (rd.get('Remote Reference', {}) or {}).get('value') is True,
        'low_locality': (rd.get('Low Locality', {}) or {}).get('value') is True,
    }


def main():
    import yaml
    from graphrag_core.config import reset_settings, get_settings
    from graphrag_core.llm.factory import create_embeddings
    from graphrag_core.retrieval.hybrid import HybridRetriever

    reset_settings()
    s = get_settings()
    tasks = yaml.safe_load(open(_proj / '../Fujitsu-RAG-Hard-Benchmark/dataset/FJ_KGQA_Hard.yaml', encoding='utf-8'))['tasks']

    multi_hop_tasks = [(i, t) for i, t in enumerate(tasks) if is_multihop(t)]
    print(f'multi-hop tasks: {len(multi_hop_tasks)}/100')
    print()

    embeddings = create_embeddings()
    hr = HybridRetriever.get_instance(s.pg_conn, collection_name='fjrag_hard')

    # 各K でのrecall集計
    K_LIST = [5, 10, 20, 50, 100]
    per_q_stats = []

    for idx, (qn_idx, t) in enumerate(multi_hop_tasks):
        question = t['question']
        # gold 参照を (pdf, page) のセットに
        gold_refs = set()
        for r in t.get('rationales') or []:
            fn = r.get('file_name', '')
            for p in r.get('pages') or []:
                gold_refs.add((fn, str(p.get('number'))))

        if not gold_refs:
            continue

        # vector search で top-100 取る
        qvec = embeddings.embed_query(question)
        try:
            results = hr.search(question, qvec, k=100, search_type='hybrid')
        except Exception as e:
            print(f'  Q{qn_idx+1}: error {e}')
            continue

        # 各結果の (pdf, page) と gold との対応
        retrieved = []
        for r in results:
            m = r.get('metadata') or {}
            retrieved.append((m.get('source'), str(m.get('page'))))

        # gold各refの順位を計算
        gold_ranks = {}
        for gref in gold_refs:
            for rank, rref in enumerate(retrieved, start=1):
                if rref == gref:
                    gold_ranks[gref] = rank
                    break
            else:
                gold_ranks[gref] = None   # top-100に無い

        # recall at K
        recalls = {}
        for K in K_LIST:
            n_in = sum(1 for r in gold_ranks.values() if r is not None and r <= K)
            recalls[K] = n_in / len(gold_refs)

        cls = classify_q(t)
        per_q_stats.append({
            'qn': qn_idx + 1,
            'question': question[:80],
            'n_gold': len(gold_refs),
            'gold_ranks': gold_ranks,
            'recalls': recalls,
            'cls': cls,
        })

    print('=== 全multi-hop 集計 ===')
    n = len(per_q_stats)
    for K in K_LIST:
        avg_r = sum(s['recalls'][K] for s in per_q_stats) / n
        full_r = sum(1 for s in per_q_stats if s['recalls'][K] == 1.0) / n
        print(f'  K={K:3d}: avg recall@K={avg_r:.2%}, 全部取れた質問={full_r:.2%}')

    print()
    print('=== gold ページの順位分布 ===')
    all_ranks = [r for s in per_q_stats for r in s['gold_ranks'].values()]
    from collections import Counter
    buckets = Counter()
    for r in all_ranks:
        if r is None: buckets['NOT_IN_TOP_100'] += 1
        elif r <= 5:  buckets['1-5'] += 1
        elif r <= 10: buckets['6-10'] += 1
        elif r <= 20: buckets['11-20'] += 1
        elif r <= 50: buckets['21-50'] += 1
        else:         buckets['51-100'] += 1
    total = sum(buckets.values())
    for b in ['1-5','6-10','11-20','21-50','51-100','NOT_IN_TOP_100']:
        c = buckets[b]
        print(f'  {b:18s}: {c:4d} ({c*100/total:.1f}%)')

    print()
    print('=== カテゴリ別 recall@5 vs recall@20 ===')
    cat_axes = ['multi_doc', 'multi_chunk', 'remote_ref', 'low_locality']
    for axis in cat_axes:
        true_q = [s for s in per_q_stats if s['cls'][axis]]
        false_q = [s for s in per_q_stats if not s['cls'][axis]]
        for label, sub in [('True', true_q), ('False', false_q)]:
            if not sub: continue
            r5 = sum(s['recalls'][5] for s in sub) / len(sub)
            r20 = sum(s['recalls'][20] for s in sub) / len(sub)
            r100 = sum(s['recalls'][100] for s in sub) / len(sub)
            print(f'  {axis:15s}={label:5s} N={len(sub):3d}: recall@5={r5:.2%}, @20={r20:.2%}, @100={r100:.2%}')

    print()
    print('=== サンプル: gold が top-100 外の質問 (本質的に取れない) ===')
    not_in_top = [s for s in per_q_stats if any(r is None for r in s['gold_ranks'].values())]
    print(f'  該当数: {len(not_in_top)} / {n}')
    for s in not_in_top[:5]:
        missing = [k for k, v in s['gold_ranks'].items() if v is None]
        found = {k: v for k, v in s['gold_ranks'].items() if v is not None}
        print(f'  Q{s["qn"]}: {s["question"]}')
        print(f'    missing: {missing}, found: {found}')

    # 出力 JSON
    Path('_bench/retrieval_diagnosis.json').write_text(
        json.dumps({
            'total': n,
            'per_q': per_q_stats,
        }, ensure_ascii=False, default=str, indent=2), encoding='utf-8'
    )


if __name__ == '__main__':
    main()
