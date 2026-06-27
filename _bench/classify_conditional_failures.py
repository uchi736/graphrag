#!/usr/bin/env python
"""Stage 0-A: 条件型失敗分類（グラフ不要・効果見積もり）

エッジqualifier(条件付き関係)の投資判断のため、現KG-offベースラインで
失敗している質問を「条件型」5分類し、qualifierで救える見込みを見積もる。

- fail源: KG-off baseline judge (fujitsu_predictions_rerank_k10_v2.azure_judge.json)
  形式 {question_text: 1|0}（0=不正解）。FJH-06のハードコードevalは使わない。
- 分類: single_condition_lookup / compound_AND / negation / temporal / non_conditional
- 各失敗質問で「回答に必要な条件→値」(qualifierが符号化する中身)をLLMに出させる
- 表/図セル依存(KG盲点, EXPERIMENTS.md)を除外フラグ化

グラフ照合(エッジ存在=対処可能 vs 抽出漏れ)は Stage 0-B（Fujitsuグラフ復元後）で行う。
本スクリプトは graph 不要 = 最安の効き代測定。
"""
from __future__ import annotations
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

YAML_PATH = _proj / '../Fujitsu-RAG-Hard-Benchmark/dataset/FJ_KGQA_Hard.yaml'
JUDGE_PATH = _proj / '_bench/fujitsu_predictions_rerank_k10_v2.azure_judge.json'
OUT_PATH = _proj / '_bench/conditional_failure_classification.json'


def axis(t: dict, group: str, key: str):
    g = t.get(group, {}) or {}
    sub = g.get(key, {}) or {}
    return sub.get('value')


CLASSIFY_PROMPT = """あなたは知識グラフQAの失敗分析者です。次の質問は現行RAGが「不正解」だったものです。
この質問が「条件付き関係(あるエッジがある条件のときだけ成立する)」をエッジqualifierで表現すれば救えるか分析してください。

質問: {q}
正解: {a}
出典: {src}
メタ軸: Comparison/Conditional={cmp} / Negation={neg} / Temporal={tmp} / Tables/Charts={tbl}

判定基準:
- single_condition_lookup: 答えが「条件Xのときの値/結論Y」一発照会（例: 深夜の割増賃金率、ある工程がCCPか）
- compound_AND: 複数条件のAND/組合せが必要（例: 食肉∧75℃未満ならCCP）
- negation: 否定・除外・「なぜ〜でないのか」型
- temporal: 時制・期間・「いつ/何年度」に依存
- non_conditional: 条件分岐と無関係（単純事実、表セルの数値読み取り、A対Bの単純比較、計算のみ）

JSON 1行で出力:
{{"conditional_type":"single_condition_lookup|compound_AND|negation|temporal|non_conditional",
"needs_condition":true/false,
"condition_payload":"回答に必要な『条件→値/結論』を簡潔に。無ければ空文字",
"table_or_figure_bound":true/false,
"qualifier_would_help":true/false,
"reason":"30字程度"}}

qualifier_would_help は「head-tail関係はグラフに在りうるが、その成立条件(数値閾値/時制/モード)が欠けているために負けている」と見込める時だけ true。
答えが表/図セルの数値そのもの(table_or_figure_bound=true)や、単純比較・計算なら qualifier_would_help=false。"""


def classify(task: dict, llm) -> dict:
    src = "; ".join(sorted({(r.get('file_name') or '') for r in (task.get('rationales') or [])}))[:120]
    prompt = CLASSIFY_PROMPT.format(
        q=task['question'], a=task['answer'], src=src or '(不明)',
        cmp=axis(task, 'Reasoning Complexity', 'Comparison (and Conditional Judgment)'),
        neg=axis(task, 'Reasoning Complexity', 'Negation Question'),
        tmp=axis(task, 'Reasoning Complexity', 'Temporal Specification'),
        tbl=axis(task, 'Source Structure & Modality', 'Tables/Charts'),
    )
    try:
        resp = llm.invoke(prompt)
        raw = resp.content if hasattr(resp, 'content') else str(resp)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    except Exception as e:
        data = {"conditional_type": "error", "reason": str(e)[:80]}
    data['no'] = task.get('no.')
    data['question'] = task['question'][:90]
    return data


def main():
    import yaml
    from graphrag_core.config import reset_settings
    from graphrag_core.llm.factory import create_chat_llm

    reset_settings()
    tasks = yaml.safe_load(open(YAML_PATH, encoding='utf-8'))['tasks']
    judge = json.load(open(JUDGE_PATH, encoding='utf-8'))  # {q: 1|0}

    # 質問文で突合 → 失敗(0)のタスク
    failed = []
    unmatched = 0
    for t in tasks:
        q = t['question']
        if q in judge:
            if judge[q] == 0:
                failed.append(t)
        else:
            unmatched += 1
    print(f"tasks={len(tasks)} / judge={len(judge)} / unmatched={unmatched}")
    print(f"KG-off baseline 失敗(=0): {len(failed)}問")

    llm = create_chat_llm(temperature=0)
    print("分類中 (concurrency=8)...")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(classify, t, llm) for t in failed]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 10 == 0 or i == len(failed):
                print(f"  {i}/{len(failed)} ({time.time()-t0:.0f}s)")

    # 集計
    by_type = Counter(r.get('conditional_type', 'error') for r in results)
    helps = [r for r in results if r.get('qualifier_would_help') is True]
    helps_not_table = [r for r in helps if not r.get('table_or_figure_bound')]
    table_bound = [r for r in results if r.get('table_or_figure_bound') is True]

    print("\n=== 失敗質問の条件型分類 ===")
    for typ, n in by_type.most_common():
        print(f"  {typ:24s}: {n}")
    print(f"\nqualifier_would_help = true        : {len(helps)}")
    print(f"  うち表/図セル依存でない(対処候補): {len(helps_not_table)}  ← Stage1の対象上限")
    print(f"表/図セル依存(KG盲点・除外)          : {len(table_bound)}")

    print("\n=== 対処候補（qualifierで救える見込み・非表）===")
    cand_types = Counter(r.get('conditional_type') for r in helps_not_table)
    for typ, n in cand_types.most_common():
        print(f"  {typ:24s}: {n}")
    print("\n--- 対処候補リスト ---")
    for r in helps_not_table:
        print(f"Q{r['no']} [{r.get('conditional_type')}] {r['question']}")
        if r.get('condition_payload'):
            print(f"    条件→値: {r['condition_payload']}")

    OUT_PATH.write_text(json.dumps({
        'n_failed': len(failed),
        'by_type': dict(by_type),
        'n_qualifier_helps': len(helps),
        'n_qualifier_helps_not_table': len(helps_not_table),
        'n_table_bound': len(table_bound),
        'addressable_candidates': helps_not_table,
        'all_results': results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n→ {OUT_PATH}")
    print("\n[GATE] 対処候補(非表)が ≥8問 で1-2型に集中なら Stage 1 へ。<5問なら STOP（投資不当）。")


if __name__ == '__main__':
    main()
