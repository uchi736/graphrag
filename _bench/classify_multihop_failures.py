#!/usr/bin/env python
"""multi-hop 失敗の (a) coref / (b) entity split / (c) extraction miss 分類

各 multi-hop 失敗質問について:
1. gold answer から key entities を LLM で抽出
2. 各 entity の graph 上の状態を確認:
   - 0ノード: PDF text に literal出現するか確認
       - 出現する → (c) extraction miss
       - しない → 同義/代名詞のみ → (a) coref miss
   - 複数ノード (表記揺れ): → (b) entity split
   - 1ノード: gold relation 存在チェック
       - 関係 missing → (c) extraction miss
3. 多数派の失敗パターン特定
"""
from __future__ import annotations
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


EXTRACT_ENT_PROMPT = """次の質問と正解から、知識グラフで参照される可能性のあるキーエンティティ (固有名詞、組織、製品、概念) を抽出してください。
代名詞や略称 (本規則、当社、彼など) は元の固有名詞に解決した形で出してください。

質問: {q}
正解: {a}

JSON 1行: {{"entities": ["..", ".."]}}
ENTITY は 2-8個、最も重要なものに絞ってください。代名詞/略称含めない。"""


def extract_key_entities(question: str, answer: str, llm) -> list[str]:
    prompt = EXTRACT_ENT_PROMPT.format(q=question, a=answer)
    try:
        resp = llm.invoke(prompt)
        raw = resp.content.strip()
        m = re.search(r"\{[^{}]*\"entities\"[^{}]*\}", raw, re.DOTALL)
        if not m:
            return []
        return json.loads(m.group(0)).get("entities", []) or []
    except Exception:
        return []


def classify_entity(graph, entity: str, pdf_texts: dict) -> dict:
    """1エンティティの graph 状態 + PDF text 状態を分析

    Returns: {
        'graph_nodes': N,
        'graph_node_names': [list],
        'in_any_pdf_text': bool,
        'classification': 'split' / 'extraction_miss' / 'coref_miss' / 'present'
    }
    """
    # graph 検索 (Term + 8 entity labels すべて)
    ENT_LABELS = ['Person', 'Organization', 'Product', 'Process',
                  'Standard', 'Indicator', 'Concept']
    rows = graph.query(f"""
        MATCH (n) WHERE n.id CONTAINS $e
          AND ANY(l IN labels(n) WHERE l IN {ENT_LABELS})
          AND NOT 'Document' IN labels(n)
          AND NOT 'ProcessedChunk' IN labels(n)
        RETURN n.id AS id, labels(n)[0] AS lbl LIMIT 30
    """, params={"e": entity})

    # 完全一致と部分一致を分けて評価
    exact = [r for r in rows if r["id"] == entity]
    partial = [r for r in rows if r["id"] != entity]

    # PDF text に literal 出現するか
    in_text = any(entity in txt for txt in pdf_texts.values())

    # 分類
    if len(exact) >= 1 and len(partial) >= 2:
        cls = 'split'   # 同義表記が分散
    elif len(exact) >= 1:
        cls = 'present'
    elif len(partial) >= 1:
        # 完全一致が無いが部分一致あり → 表記揺れで分散
        cls = 'split'
    elif in_text:
        cls = 'extraction_miss'   # text に出るが graph に無い
    else:
        cls = 'coref_miss'   # text にも無い (代名詞/別名で参照されてる可能性)

    return {
        'entity': entity,
        'graph_exact': len(exact),
        'graph_partial': len(partial),
        'graph_names': [r['id'] for r in (exact + partial)[:5]],
        'in_pdf_text': in_text,
        'classification': cls,
    }


def main():
    import yaml
    from graphrag_core.config import reset_settings, get_settings
    from graphrag_core.llm.factory import create_chat_llm
    from langchain_neo4j import Neo4jGraph

    reset_settings()
    s = get_settings()
    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw)
    llm = create_chat_llm(temperature=0)

    tasks = yaml.safe_load(open(_proj / '../Fujitsu-RAG-Hard-Benchmark/dataset/FJ_KGQA_Hard.yaml', encoding='utf-8'))['tasks']

    # KG_old (FJH-04) の評価結果を使う - multi-hop 失敗を取得
    # 現状 fjrag_hard コレクションは FJH-06 だが、Term 構造は似てるので analysis 用に流用
    eval_kg = json.load(open(_proj / '_bench/results/evaluation_result_20260607_214652.json', encoding='utf-8'))
    fail_qns = {d['question_number'] for d in eval_kg['details'] if d['final_evaluation'] == 'incorrect'}

    # multi-hop かつ KG_v3 失敗の質問
    targets = []
    for i, t in enumerate(tasks):
        qn = i + 1
        if is_multihop(t) and qn in fail_qns:
            targets.append((qn, t))
    print(f'multi-hop & KG失敗: {len(targets)}問')

    # PDFs を pp化済みのtextで一括ロード
    print('PDF text load 中...')
    pdf_texts = {}
    pp_dir = (_proj / '_bench/_pp').resolve()
    for sub in pp_dir.iterdir():
        if not sub.is_dir(): continue
        txt = sub / 'extracted_text.txt'
        if txt.exists():
            pdf_texts[f'{sub.name}.pdf'] = txt.read_text(encoding='utf-8')
    print(f'  loaded {len(pdf_texts)} PDF texts')

    # gold で参照されてるPDF だけ絞る効率化用
    def relevant_texts(task):
        pdfs = set()
        for r in task.get('rationales') or []:
            fn = r.get('file_name', '')
            if fn: pdfs.add(fn)
        return {k: v for k, v in pdf_texts.items() if k in pdfs} or pdf_texts

    # 並列処理
    def _analyze(target):
        qn, t = target
        ents = extract_key_entities(t['question'], t['answer'], llm)
        rel_text = relevant_texts(t)
        ent_states = []
        for e in ents[:8]:
            ent_states.append(classify_entity(graph, e, rel_text))
        # この質問の支配的なパターン
        cls_counts = {'split': 0, 'extraction_miss': 0, 'coref_miss': 0, 'present': 0}
        for es in ent_states:
            cls_counts[es['classification']] += 1
        # 多数決
        question_pattern = max(cls_counts, key=cls_counts.get) if any(cls_counts.values()) else 'unknown'
        return {
            'qn': qn,
            'question': t['question'][:80],
            'entities': ents,
            'entity_states': ent_states,
            'cls_counts': cls_counts,
            'question_pattern': question_pattern,
        }

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_analyze, tg) for tg in targets]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
                results.append(r)
            except Exception as e:
                print(f'  err: {e}')
            if i % 10 == 0 or i == len(targets):
                print(f'  {i}/{len(targets)}  ({time.time()-t0:.0f}s)')

    # 集計
    from collections import Counter
    pattern_dist = Counter(r['question_pattern'] for r in results)
    print()
    print('=== 質問単位の支配パターン ===')
    for p, n in pattern_dist.most_common():
        print(f'  {p:20s}: {n} ({n*100/len(results):.0f}%)')

    # entity単位の集計
    all_ent_cls = []
    for r in results:
        for es in r['entity_states']:
            all_ent_cls.append(es['classification'])
    ent_dist = Counter(all_ent_cls)
    print()
    print('=== エンティティ単位 (重複あり) ===')
    for p, n in ent_dist.most_common():
        print(f'  {p:20s}: {n} ({n*100/len(all_ent_cls):.0f}%)')

    # サンプル提示
    for pattern in ['split', 'extraction_miss', 'coref_miss']:
        sub = [r for r in results if r['question_pattern'] == pattern]
        if not sub: continue
        print()
        print(f'=== サンプル: {pattern} (5問) ===')
        for r in sub[:5]:
            print(f'Q{r["qn"]}: {r["question"]}')
            print(f'  entities: {r["entities"][:4]}')
            for es in r['entity_states'][:3]:
                names = es['graph_names'][:3]
                print(f'    [{es["classification"]:18s}] {es["entity"]:25s} (graph: exact={es["graph_exact"]}, partial={es["graph_partial"]}, names={names}, in_pdf={es["in_pdf_text"]})')

    out = _proj / '_bench/multihop_failure_classification.json'
    out.write_text(json.dumps({
        'pattern_dist': dict(pattern_dist),
        'entity_dist': dict(ent_dist),
        'results': results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
