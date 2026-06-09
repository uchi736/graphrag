#!/usr/bin/env python
"""multi-hop 失敗の v2分類 — bi-encoder ベースで「真の split」を計測

旧v1 の問題点:
- graph_partial = CONTAINS substring matching → 別実体 (熱中症 vs 熱中症警戒アラート) も
  split 扱いしていた → 過大評価

v2 の改善:
- 各 gold entity に対して、全 entity node を bi-encoder で類似度計算
- 真部分文字列ペア・数値風はフィルタ
- type違いは別カテゴリ
- 類似度 > 0.95 (LLM judge せず) が 2件以上 = 真 split
- 部分文字列で関連だが意味的に類似してない = compound_dispersion (link 候補)

カテゴリ:
- true_split: 同一実体が分散 (EL merge で救える)
- compound_dispersion: 名前共有だが別実体 (link で救える: 熱中症 → 熱中症発生件数)
- present: graph に exact 存在
- coref_miss: graph・PDF text共に無い (概念/計算メトリクス)
- extraction_miss: PDF text に有るが graph に無い
"""
from __future__ import annotations
import json
import re
import sys
import time
import unicodedata
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


# EL_v2 と同じフィルタ関数を再利用
_NUMERIC_PAT = re.compile(r"^[\d,.\-+%]+$")
_DATE_PAT = re.compile(r"^\d{1,4}[年/月日.\-]\d{1,2}")
_PERCENT_PAT = re.compile(r"^\d+(?:[\.,]\d+)?[%％]")


def is_numeric_like(s: str) -> bool:
    s = (s or '').strip()
    if not s: return False
    if _NUMERIC_PAT.match(s): return True
    if _DATE_PAT.match(s): return True
    if _PERCENT_PAT.match(s): return True
    digits = sum(1 for c in s if c.isdigit())
    return digits * 2 > len(s)


def is_substring_pair(a: str, b: str) -> bool:
    if a == b: return False
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return short in long_ and len(short) >= 2 and len(long_) > len(short) + 2


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    try:
        import neologdn
        s = neologdn.normalize(s)
    except ImportError:
        pass
    return re.sub(r"\s+", "", s).strip()


EXTRACT_ENT_PROMPT = """質問と正解から、知識グラフで参照される可能性のあるキーエンティティ (固有名詞、組織、製品、概念) を抽出してください。
代名詞や略称は元の固有名詞に解決した形で出してください。

質問: {q}
正解: {a}

JSON 1行: {{"entities": ["..", ".."]}}
ENTITY は 2-8個、最も重要なものに絞ってください。"""


def extract_key_entities(question: str, answer: str, llm) -> list[str]:
    prompt = EXTRACT_ENT_PROMPT.format(q=question, a=answer)
    try:
        resp = llm.invoke(prompt)
        raw = resp.content.strip()
        m = re.search(r"\{[^{}]*\"entities\"[^{}]*\}", raw, re.DOTALL)
        if not m: return []
        return json.loads(m.group(0)).get("entities", []) or []
    except Exception:
        return []


def classify_entity_v2(entity: str, all_terms: list[dict], term_vecs, ent_vec,
                      pdf_texts: dict, similarity_threshold: float = 0.92) -> dict:
    """1エンティティの v2分類

    Returns:
        {
            classification: 'true_split' | 'compound_dispersion' | 'present' |
                            'coref_miss' | 'extraction_miss',
            graph_exact_hits: int,
            true_split_candidates: list[name],  # bi-enc類似 + 同label + non-substring + non-numeric
            compound_partials: list[name],       # substring一致だが意味的に別実体
            in_pdf_text: bool,
        }
    """
    import numpy as np
    entity_norm = normalize_name(entity)
    is_num = is_numeric_like(entity)

    # 1. exact match (id == entity or normalized id == normalized entity)
    exact_matches = [t for t in all_terms if t['id'] == entity or t['normalized'] == entity_norm]

    # 2. 全 term との bi-encoder cosine similarity
    sims = term_vecs @ ent_vec   # (N,) ベクトル

    # 3. similar candidates (top-30) を抽出
    top_idx = np.argpartition(-sims, min(30, len(sims)-1))[:30]
    sorted_idx = sorted(top_idx, key=lambda i: -sims[i])

    true_split = []
    compound = []
    for i in sorted_idx[:30]:
        sim = float(sims[i])
        if sim < 0.5: continue
        t = all_terms[i]
        name = t['id']
        if name == entity: continue

        is_substr_pair = is_substring_pair(name, entity)
        is_num_partner = is_numeric_like(name)

        # 真の split: 高類似 + 非substring + 非数値 + 同label (もし entity の label情報があれば)
        if sim >= similarity_threshold and not is_substr_pair and not is_num_partner and not is_num:
            true_split.append((name, t['labels'][0], sim))
        # compound dispersion: substring一致しているが類似度低 (= 別実体)
        elif is_substr_pair:
            compound.append((name, t['labels'][0], sim))

    in_text = any(entity in txt for txt in pdf_texts.values())

    # 分類ロジック
    if len(true_split) >= 1:
        cls = 'true_split'
    elif len(exact_matches) >= 1:
        cls = 'present'
    elif len(compound) >= 1:
        cls = 'compound_dispersion'
    elif in_text:
        cls = 'extraction_miss'
    else:
        cls = 'coref_miss'

    return {
        'entity': entity,
        'classification': cls,
        'exact_hits': len(exact_matches),
        'true_split': [(n, l, round(s, 3)) for n, l, s in true_split[:5]],
        'compound_dispersion': [(n, l, round(s, 3)) for n, l, s in compound[:5]],
        'in_pdf_text': in_text,
    }


def main():
    import yaml
    from graphrag_core.config import reset_settings, get_settings
    from graphrag_core.llm.factory import create_chat_llm, create_embeddings
    from langchain_neo4j import Neo4jGraph
    import numpy as np

    reset_settings()
    s = get_settings()
    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw)
    llm = create_chat_llm(temperature=0)
    embeddings = create_embeddings()

    tasks = yaml.safe_load(open(_proj / '../Fujitsu-RAG-Hard-Benchmark/dataset/FJ_KGQA_Hard.yaml', encoding='utf-8'))['tasks']

    # KG_v3 (FJH-06) 評価結果から失敗質問特定
    eval_file = _proj / '_bench/results/evaluation_result_20260607_214652.json'
    eval_data = json.load(open(eval_file, encoding='utf-8'))
    fail_qns = {d['question_number'] for d in eval_data['details'] if d['final_evaluation'] == 'incorrect'}

    targets = [(i+1, t) for i, t in enumerate(tasks) if is_multihop(t) and (i+1) in fail_qns]
    print(f'multi-hop & failed: {len(targets)}問')

    # 全 Term node 取得 + ベクトル化
    ENT = ['Person','Organization','Product','Process','Standard','Indicator','Concept']
    labels_filter = ' OR '.join([f'"{l}" IN labels(t)' for l in ENT])
    print('全 Term node 取得中...')
    t0 = time.time()
    rows = graph.query(f"""
        MATCH (t) WHERE ({labels_filter}) AND NOT 'Document' IN labels(t) AND NOT 'ProcessedChunk' IN labels(t)
        RETURN t.id AS id, labels(t)[0] AS lbl, COALESCE(t.pagerank, 0.0) AS pr
        ORDER BY t.pagerank DESC LIMIT 10000
    """)
    print(f'  fetched {len(rows)} terms ({time.time()-t0:.1f}s)')
    for r in rows:
        r['labels'] = [r['lbl']]
        r['normalized'] = normalize_name(r['id'])

    print('Term 名 embedding 中...')
    t0 = time.time()
    names = [r['id'] for r in rows]
    BATCH = 64
    all_vecs = []
    for i in range(0, len(names), BATCH):
        all_vecs.extend(embeddings.embed_documents(names[i:i+BATCH]))
    term_vecs = np.array(all_vecs, dtype=np.float32)
    # L2正規化
    norms = np.linalg.norm(term_vecs, axis=1, keepdims=True); norms[norms == 0] = 1
    term_vecs = term_vecs / norms
    print(f'  embedded ({time.time()-t0:.1f}s)')

    # PDF text load
    pdf_texts = {}
    for sub in (_proj / '_bench/_pp').iterdir():
        if not sub.is_dir(): continue
        f = sub / 'extracted_text.txt'
        if f.exists():
            pdf_texts[f'{sub.name}.pdf'] = f.read_text(encoding='utf-8')

    def relevant_texts(t):
        pdfs = set(r.get('file_name', '') for r in (t.get('rationales') or []))
        return {k: v for k, v in pdf_texts.items() if k in pdfs} or pdf_texts

    # 各 target を分類
    def _analyze(target):
        qn, t = target
        ents = extract_key_entities(t['question'], t['answer'], llm)
        rel_text = relevant_texts(t)
        # entity vector
        ent_states = []
        for e in ents[:8]:
            try:
                ev = embeddings.embed_query(e)
                ev = np.array(ev, dtype=np.float32)
                ev = ev / (np.linalg.norm(ev) + 1e-12)
                state = classify_entity_v2(e, rows, term_vecs, ev, rel_text)
                ent_states.append(state)
            except Exception as ex:
                ent_states.append({'entity': e, 'classification': 'error', 'err': str(ex)[:80]})
        cls_counts = {}
        for es in ent_states:
            cls = es['classification']
            cls_counts[cls] = cls_counts.get(cls, 0) + 1
        return {
            'qn': qn,
            'question': t['question'][:80],
            'entities': ents,
            'entity_states': ent_states,
            'cls_counts': cls_counts,
            'question_pattern': max(cls_counts, key=cls_counts.get) if cls_counts else 'unknown',
        }

    print(f'\n各質問を分類中 (concurrency=8)...')
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_analyze, tg) for tg in targets]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f'  err: {e}')
            if i % 10 == 0 or i == len(targets):
                print(f'  {i}/{len(targets)} ({time.time()-t0:.0f}s)')

    from collections import Counter
    qpat = Counter(r['question_pattern'] for r in results)
    print('\n=== 質問単位 (v2分類) ===')
    for p, n in qpat.most_common():
        print(f'  {p:25s}: {n} ({n*100/len(results):.0f}%)')

    all_ent = [es['classification'] for r in results for es in r['entity_states']]
    epat = Counter(all_ent)
    print('\n=== エンティティ単位 (v2分類) ===')
    for p, n in epat.most_common():
        print(f'  {p:25s}: {n} ({n*100/len(all_ent):.0f}%)')

    # サンプル提示
    for pattern in ['true_split', 'compound_dispersion', 'present', 'coref_miss', 'extraction_miss']:
        sub = [r for r in results if r['question_pattern'] == pattern]
        if not sub: continue
        print(f'\n=== サンプル: {pattern} (3問) ===')
        for r in sub[:3]:
            print(f'Q{r["qn"]}: {r["question"]}')
            for es in r['entity_states'][:3]:
                cls = es['classification']
                ts = es.get('true_split', [])
                cd = es.get('compound_dispersion', [])
                print(f'  [{cls:22s}] {es["entity"]:30s}')
                if ts:
                    print(f'    true_split候補: {ts[:3]}')
                if cd:
                    print(f'    compound: {cd[:3]}')

    # 出力
    out = _proj / '_bench/multihop_failure_classification_v2.json'
    out.write_text(json.dumps({
        'qpat': dict(qpat), 'epat': dict(epat), 'results': results,
    }, ensure_ascii=False, default=str, indent=2), encoding='utf-8')
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
