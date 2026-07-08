#!/usr/bin/env python
"""plant_v15 を graphrag の検索パイプラインで評価（KG-ON版）。

run_plant.py(KG-off) との違い:
- Neo4j graph を retriever_and_merge に渡す（graph!=None）
- include_kg_source_chunks=True: エンティティ→トリプル→edge.source_chunks(chunk_id)で
  抽出元チャンク本文を注入（page非依存の本命KG経路）
- predicted_references は vector_sources + kg_source_chunks の双方から
- enable_reference_follow は plant では page=None で無効なので OFF

生成器は .env の LLM_PROVIDER（gemma にするなら LLM_PROVIDER=vllm で起動）。

Usage:
    LLM_PROVIDER=vllm python _bench/plant/run_plant_kgon.py \
      --qa-path C:/work/makedataset/data/dataset_kg_v3.jsonl \
      --output _bench/plant/pred_v3_gemma_kgon.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from dotenv import load_dotenv
load_dotenv()
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from _bench.plant.run_plant import GEN_PROMPT, load_qa, correct_refs


def run_one(task, graph, llm, embeddings, vector_retriever, pg_conn, collection, config, gen_prompt):
    from graphrag_core.retrieval.pipeline import retriever_and_merge
    q = task["question"]
    try:
        merge = retriever_and_merge(q, graph, llm, embeddings, vector_retriever,
                                    pg_conn, collection, config)
        ctx = merge.get("context", "")
        # 予測参照 = ベクトル由来 + KGソースチャンク由来の doc_id
        pred_refs, seen = [], set()
        for d in (merge.get("vector_sources") or []) + (merge.get("kg_source_chunks") or []):
            did = (d.metadata or {}).get("source")
            if did and did not in seen:
                seen.add(did)
                pred_refs.append(did)
        n_kg = len(merge.get("kg_source_chunks") or [])
        resp = llm.invoke(gen_prompt.format(context=ctx, question=q))
        ans = resp.content.strip()
        ok = True
    except Exception as e:
        ans, pred_refs, ok, n_kg = f"[runner error] {e}", [], False, 0
    return {
        "qa_id": task.get("qa_id"),
        "question": q,
        "predicted_answer": ans,
        "correct_answer": task["answer"],
        "predicted_references": pred_refs,
        "correct_references": correct_refs(task),
        "retrieval_level": task.get("retrieval_level"),
        "answer_level": task.get("answer_level"),
        "kg_query_type": task.get("kg_query_type"),
        "n_kg_chunks": n_kg,
        "success": ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="plant_v15")
    ap.add_argument("--qa-path", default="C:/work/makedataset/data/dataset_kg_v3.jsonl")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--fetch-k", type=int, default=20)
    ap.add_argument("--kg-chunk-top-k", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--output", default="_bench/plant/pred_v3_gemma_kgon.json")
    args = ap.parse_args()

    os.environ["PG_COLLECTION"] = args.collection
    from graphrag_core.config import reset_settings, get_settings
    reset_settings()
    s = get_settings()

    from graphrag_core.llm.factory import create_chat_llm, create_embeddings
    from langchain_postgres import PGVector
    from langchain_neo4j import Neo4jGraph
    from graphrag_core.db.utils import add_connection_timeout

    embeddings = create_embeddings()
    llm = create_chat_llm(temperature=0)
    try:
        llm.request_timeout = 120
        llm.max_retries = 3
    except Exception:
        pass

    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw, enhanced_schema=False)

    pg = add_connection_timeout(s.pg_conn, timeout=30)
    vstore = PGVector(connection=pg, embeddings=embeddings, collection_name=args.collection)
    vretr = vstore.as_retriever(search_kwargs={"k": args.fetch_k})

    config = {
        "retrieval_top_k": args.top_k,
        "rerank_pool_size": args.fetch_k,
        "enable_japanese_search": True,
        "enable_rerank": True,
        "enable_entity_vector": False,       # search_keys 照合で足りる
        "search_mode": "hybrid",
        "include_kg_source_chunks": True,    # ★ 本命KG経路
        "kg_chunk_top_k": args.kg_chunk_top_k,
        "include_graph_lines": False,        # graph triples テキストは出さない
        "enable_reference_follow": False,    # plant は page=None で無効
    }

    tasks = load_qa(args.qa_path)
    print(f"=== plant_v15 KG-ON runner === Q={len(tasks)} qa={args.qa_path} "
          f"collection={args.collection} top_k={args.top_k} kg_chunk_top_k={args.kg_chunk_top_k} "
          f"llm={s.llm_provider}")

    results = [None] * len(tasks)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, tasks[i], graph, llm, embeddings, vretr,
                          s.pg_conn, args.collection, config, GEN_PROMPT): i
                for i in range(len(tasks))}
        done = 0
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            print(f"  {done}/{len(tasks)} ({(time.time()-t0)/done:.1f}s/q)", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    succ = sum(1 for r in results if r["success"])
    avg_kg = sum(r["n_kg_chunks"] for r in results) / max(1, len(results))
    print(f"完了: {time.time()-t0:.0f}s success={succ}/{len(results)} "
          f"avg_kg_chunks/q={avg_kg:.1f} -> {args.output}")


if __name__ == "__main__":
    main()
