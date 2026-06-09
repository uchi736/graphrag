#!/usr/bin/env python
"""質問1件を流して、KG検索の各フェーズの中間出力をダンプする。

Usage:
    python _eval/trace_query.py "就業規則の周知方法を列挙してください。"
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from graphrag_core.config import get_settings
from graphrag_core.llm.factory import create_chat_llm, create_embeddings


def hr(title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "就業規則の周知方法を列挙してください。"
    s = get_settings()

    hr(f"質問: {question}")
    print(f"hop_count={s.graph_hop_count}, top_k={s.retrieval_top_k}, "
          f"search_mode={s.search_mode}, entity_threshold={s.entity_similarity_threshold}")

    llm = create_chat_llm(temperature=0)
    embeddings = create_embeddings()

    # ========== Phase 1a: LLMエンティティ抽出 ==========
    from graphrag_core.prompts import ENTITY_EXTRACTION_PROMPT
    extraction_prompt = ENTITY_EXTRACTION_PROMPT.format(question=question)
    resp = llm.invoke(extraction_prompt)
    llm_entities = [e.strip() for e in resp.content.split(',') if e.strip()]
    hr("Phase 1a: LLMエンティティ抽出")
    print(f"  raw output: {resp.content!r}")
    print(f"  parsed   : {llm_entities}")

    # ========== Phase 1b: エンティティベクトル検索 ==========
    hr("Phase 1b: エンティティベクトル検索 (myapp_entities)")
    from graphrag_core.retrieval.entity_vector import EntityVectorizer
    ev = EntityVectorizer(s.pg_conn, embeddings)
    similar = ev.search_hybrid_entities(
        question, k=10,
        score_threshold=s.entity_similarity_threshold,
        search_type=s.search_mode,
    )
    print(f"  {len(similar)}件:")
    for eid, score in similar[:10]:
        print(f"    {score:.3f}  {eid}")

    # ========== merged_entities ==========
    merged = list(llm_entities)
    for eid, _ in similar:
        if eid not in merged:
            merged.append(eid)
    hr(f"merged_entities ({len(merged)})")
    for e in merged:
        marker = "L" if e in llm_entities else "V"
        print(f"  [{marker}] {e}")

    # ========== Phase 2: Cypher探索 ==========
    from langchain_neo4j import Neo4jGraph
    g = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw)

    hop = s.graph_hop_count
    _hex = "NOT n.id =~ '[0-9a-f]{32,}'"
    _hex_node = "NOT node.id =~ '[0-9a-f]{32,}'"
    limit = {1: 30, 2: 50, 3: 80}.get(hop, 50)

    if hop == 1:
        cy = ("UNWIND $entities AS entity "
              "MATCH (n) WHERE n.id CONTAINS entity AND " + _hex + " "
              "WITH collect(DISTINCT n) AS matched_nodes "
              "UNWIND matched_nodes AS start_node "
              "MATCH path = (start_node)-[r]-(end_node) "
              "WHERE type(r) <> 'MENTIONS' AND NOT end_node.id =~ '[0-9a-f]{32,}' "
              "RETURN [node IN nodes(path) | node.id] AS node_ids, "
              "[rel IN relationships(path) | {start: startNode(rel).id, type: type(rel), end: endNode(rel).id}] AS rels "
              f"LIMIT {limit}")
    else:
        cy = ("UNWIND $entities AS entity "
              "MATCH (n) WHERE n.id CONTAINS entity AND " + _hex + " "
              "WITH collect(DISTINCT n) AS matched_nodes "
              "UNWIND matched_nodes AS start_node "
              f"MATCH path = (start_node)-[*1..{hop}]-(end_node) "
              "WHERE ALL(r IN relationships(path) WHERE type(r) <> 'MENTIONS') "
              "AND ALL(node IN nodes(path) WHERE " + _hex_node + ") "
              "AND start_node <> end_node "
              "RETURN [node IN nodes(path) | node.id] AS node_ids, "
              "[rel IN relationships(path) | {start: startNode(rel).id, type: type(rel), end: endNode(rel).id}] AS rels "
              f"LIMIT {limit}")

    raw = g.query(cy, params={"entities": merged})
    hr(f"Phase 2: Cypher探索 (hop={hop}, limit={limit})")
    print(f"  hit raw rows: {len(raw)}")
    for i, r in enumerate(raw[:8]):
        nodes = r["node_ids"]; rels = r["rels"]
        chain = nodes[0]
        for k in range(len(nodes) - 1):
            chain += f" -[{rels[k]['type']}]-> {nodes[k+1]}"
        print(f"   {i+1:2d}. {chain}")
    if len(raw) > 8:
        print(f"   ... ({len(raw)-8} more)")

    # ========== Phase 3: パスのパースと重複除去 ==========
    from graphrag_core.retrieval.pipeline import parse_neo4j_paths
    parsed = parse_neo4j_paths(raw, max_candidates=s.path_max_candidates)
    hr(f"Phase 3: parse_neo4j_paths (重複除去・長さ降順)")
    print(f"  uniq paths: {len(parsed)} (max_candidates={s.path_max_candidates})")
    for i, p in enumerate(parsed[:10]):
        print(f"   {i+1:2d}. (len={p['length']}) {p['path_text']}")
    if len(parsed) > 10:
        print(f"   ... ({len(parsed)-10} more)")

    # ========== Phase 4: cross-encoder rerank ==========
    from graphrag_core.retrieval.reranker import is_reranker_enabled, score_candidates
    top_k = {1: 15, 2: 20, 3: 25}.get(hop, 15)
    hr(f"Phase 4: cross-encoder rerank (top_k={top_k})")
    print(f"  reranker enabled: {is_reranker_enabled()}")
    if parsed and is_reranker_enabled():
        # doc_context を含めずに question 単体でまず流す
        texts = [p["path_text"] for p in parsed]
        scores_q = score_candidates(question, texts)
        if scores_q:
            ranked_q = sorted(zip(scores_q, parsed), key=lambda t: -t[0])
            print("  --- query単体 (doc_context無し) ---")
            for i, (sc, p) in enumerate(ranked_q[:top_k]):
                print(f"   {i+1:2d}. {sc:+.3f}  {p['path_text']}")
        # doc_context あり (検索済みdocs先頭)
        # ここではvector検索を別途実行して上位5件の冒頭を使う
        from graphrag_core.retrieval.hybrid import HybridRetriever
        hr2 = HybridRetriever.get_instance(s.pg_conn, collection_name=s.pg_collection)
        qv = embeddings.embed_query(question)
        docs = hr2.search(question, qv, k=5, search_type=s.search_mode)
        doc_ctx = "\n---\n".join(d.get("text", "")[:200] for d in docs[:5])
        scores_d = score_candidates(question + "\n" + doc_ctx, texts)
        if scores_d:
            ranked_d = sorted(zip(scores_d, parsed), key=lambda t: -t[0])
            print("  --- query + doc_context ---")
            for i, (sc, p) in enumerate(ranked_d[:top_k]):
                print(f"   {i+1:2d}. {sc:+.3f}  {p['path_text']}")
            top_paths = [p for _, p in ranked_d[:top_k]]
        else:
            top_paths = parsed[:top_k]
    else:
        top_paths = parsed[:top_k]

    # ========== Phase 5: triple平坦化 ==========
    hr(f"Phase 5: triple平坦化 (重複除去)")
    seen = set(); flat = []
    for p in top_paths:
        for t in p["triples"]:
            key = (t["start"], t["type"], t["end"])
            if key not in seen:
                seen.add(key); flat.append(t)
    print(f"  flat triples: {len(flat)}")
    for i, t in enumerate(flat[:15]):
        print(f"   {i+1:2d}. {t['start']} -[{t['type']}]-> {t['end']}")

    # ========== Phase 6: KGソースチャンク ==========
    if s.include_kg_source_chunks and flat:
        ent_names = list(set([t['start'] for t in flat] + [t['end'] for t in flat]))
        chunk_q = """
        UNWIND $entity_names AS entity_name
        MATCH (e {id: entity_name})<-[:MENTIONS]-(doc:Document)
        RETURN DISTINCT doc.id AS chunk_id,
               substring(doc.text, 0, 300) AS text,
               doc.source AS source
        LIMIT 5
        """
        chunks = g.query(chunk_q, params={"entity_names": ent_names})
        hr(f"Phase 6: KGソースチャンク取得")
        print(f"  endpoint entities: {len(ent_names)}")
        print(f"  hit chunks: {len(chunks)}")
        for i, c in enumerate(chunks):
            print(f"   {i+1}. [{c.get('source')}] {c['text'][:120]}...")


if __name__ == "__main__":
    main()
