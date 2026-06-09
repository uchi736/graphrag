#!/usr/bin/env python
"""multi_doc / remote_ref 質問に対する KG 挙動を trace

各質問について:
- LLM抽出 entity
- Vector entity 候補
- graph_paths (実際に traverse された path)
- kg_source_chunks の出元 pdf/page
- vector_sources の出元 pdf/page
- gold refs
- 正解判定

正解問題 vs 不正解問題で KG の挙動が何が違うか可視化。
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))


def main():
    import yaml
    os.environ['PG_COLLECTION'] = 'fjrag_hard'
    from graphrag_core.config import reset_settings, get_settings
    from graphrag_core.llm.factory import create_chat_llm, create_embeddings
    from graphrag_core.retrieval.pipeline import retriever_and_merge
    from graphrag_core.db.utils import add_connection_timeout
    from langchain_neo4j import Neo4jGraph
    from langchain_postgres import PGVector

    reset_settings(); s = get_settings()
    embeddings = create_embeddings()
    llm = create_chat_llm(temperature=0)
    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw, enhanced_schema=False)
    pg_conn = add_connection_timeout(s.pg_conn, timeout=30)
    vector_store = PGVector(connection=pg_conn, embeddings=embeddings, collection_name='fjrag_hard')
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 20})

    config = {
        "graph_hop_count": 2, "retrieval_top_k": 10, "enable_japanese_search": True,
        "enable_rerank": False, "enable_entity_vector": True,
        "entity_similarity_threshold": s.entity_similarity_threshold,
        "search_mode": "hybrid", "include_kg_source_chunks": True,
        "path_max_candidates": s.path_max_candidates,
        "include_graph_lines": False,
    }

    # eval result
    eval_d = json.load(open('_bench/results/evaluation_result_20260609_065349.json', encoding='utf-8'))
    eval_by_qn = {x['question_number']: x['final_evaluation'] for x in eval_d['details']}
    tasks = yaml.safe_load(open('../Fujitsu-RAG-Hard-Benchmark/dataset/FJ_KGQA_Hard.yaml', encoding='utf-8'))['tasks']

    def is_target(qn):
        t = tasks[qn-1]
        rd = t.get('Retrieval Difficulty', {}) or {}
        return ((rd.get('multi-document',{}) or {}).get('value') is True
                or (rd.get('Remote Reference',{}) or {}).get('value') is True)

    target_qns = [qn for qn in range(1, 101) if is_target(qn)]
    print(f'trace 対象: {len(target_qns)}問')

    out = []
    t0 = time.time()
    for i, qn in enumerate(target_qns, 1):
        t = tasks[qn-1]
        q = t['question']
        gold_refs = []
        for r in t.get('rationales') or []:
            fn = r.get('file_name', '')
            for p in r.get('pages') or []:
                gold_refs.append({'pdf': fn, 'page': str(p.get('number'))})

        try:
            m = retriever_and_merge(q, graph, llm, embeddings, vector_retriever, pg_conn, 'fjrag_hard', config)
        except Exception as e:
            print(f'  Q{qn} err: {e}')
            continue

        ee = m.get('extracted_entities', {}) or {}
        paths = m.get('graph_paths', []) or []
        triples = m.get('graph_sources', []) or []
        vec_docs = m.get('vector_sources') or []
        kg_docs = m.get('kg_source_chunks') or []

        vec_refs = [{'pdf': d.metadata.get('source','?'), 'page': str(d.metadata.get('page','?'))} for d in vec_docs]
        kg_refs = [{'pdf': d.metadata.get('source','?'), 'page': str(d.metadata.get('page','?'))} for d in kg_docs]

        # gold が vector / kg どちらで取れたか
        def gold_in(refs):
            hits = []
            for g in gold_refs:
                for r in refs:
                    if r['pdf'] == g['pdf'] and r['page'] == g['page']:
                        hits.append(g); break
            return hits

        vec_hit = gold_in(vec_refs)
        kg_hit = gold_in(kg_refs)
        # KG が独自に追加した gold = kg_hit にあって vec_hit に無いもの
        kg_unique = [g for g in kg_hit if g not in vec_hit]

        out.append({
            'qn': qn,
            'eval': eval_by_qn.get(qn, '?'),
            'question': q[:100],
            'llm_entities': ee.get('llm_entities', []),
            'vec_entities_top10': [e[0] for e in (ee.get('vector_entities') or [])[:10]],
            'merged_entities_count': len(ee.get('merged_entities', [])),
            'merged_entities_sample': (ee.get('merged_entities') or [])[:10],
            'path_count': len(paths),
            'triple_count': len(triples),
            'path_samples': [p.get('path_text','')[:120] for p in paths[:5]],
            'kg_chunks_count': len(kg_refs),
            'kg_chunks': kg_refs[:10],
            'vec_refs_top10': vec_refs[:10],
            'gold_refs': gold_refs,
            'vec_hit_count': len(vec_hit),
            'kg_hit_count': len(kg_hit),
            'kg_unique_gold': kg_unique,  # KG が独自に持ってきた gold
        })

        if i % 5 == 0:
            print(f'  {i}/{len(target_qns)}  ({time.time()-t0:.0f}s)')

    Path('_bench/trace_kg_subset.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8'
    )
    print(f'→ _bench/trace_kg_subset.json')
    print(f'total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
