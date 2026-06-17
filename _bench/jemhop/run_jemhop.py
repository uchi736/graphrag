#!/usr/bin/env python
"""JEMHopQA dev を graphrag 検索パイプライン(KG-off)で実行。

run_plant.py と同型。JEMHopQAは短答(実体名/YES-NO/数値)なので生成プロンプトは簡潔版。
gold参照 = page_ids、predicted参照 = 取得チャンクの doc_id(=page_id)。

Usage:
    python _bench/jemhop/run_jemhop.py --top-k 10 --fetch-k 20 --concurrency 2
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

HERE = Path(__file__).resolve().parent
QA_PATH = HERE / "dev.json"

GEN_PROMPT = """あなたは【参考情報】に基づいて質問に簡潔に答えるアシスタントです。

【参考情報】
{context}

【質問】
{question}

【回答ルール】
- 参考情報に書かれている内容だけを根拠に答える。推測や一般知識で補完しない。
- 比較質問（「どちらが」「ですか？」等）には、該当する固有名詞、または「はい」「いいえ」で端的に答える。
- 答えの中核（固有名詞・数値・日付・YES/NO）を最初に述べ、冗長な説明は避ける。
- 参考情報から判断できない場合のみ「情報なし」と述べる。

【回答】"""


def load_qa():
    return json.loads(QA_PATH.read_text(encoding="utf-8"))


def run_one(task, graph, llm, embeddings, vretr, pg_conn, collection, config):
    from graphrag_core.retrieval.pipeline import retriever_and_merge
    q = task["question"]
    try:
        merge = retriever_and_merge(q, graph, llm, embeddings, vretr,
                                    pg_conn, collection, config)
        ctx = merge.get("context", "")
        # KG-on時はKGソースチャンク(bridge回復分)も参照候補に含める
        docs = list(merge.get("vector_sources") or []) + list(merge.get("kg_source_chunks") or [])
        pred_refs, seen = [], set()
        for d in docs:
            did = (d.metadata or {}).get("source")
            if did and str(did) not in seen:
                seen.add(str(did))
                pred_refs.append(str(did))
        resp = llm.invoke(GEN_PROMPT.format(context=ctx, question=q))
        ans, ok = resp.content.strip(), True
    except Exception as e:
        ans, pred_refs, ok = f"[runner error] {e}", [], False
    return {
        "qid": task.get("qid"),
        "type": task.get("type"),
        "question": q,
        "predicted_answer": ans,
        "gold_answer": task.get("answer"),
        "predicted_references": pred_refs,
        "gold_references": [str(p) for p in (task.get("page_ids") or [])],
        "derivations": task.get("derivations"),
        "success": ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="jemhopqa")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--fetch-k", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--kg", action="store_true", help="KG-on: Neo4jグラフ探索+KGチャンク注入")
    ap.add_argument("--graph-hop", type=int, default=2)
    ap.add_argument("--output", default="_bench/jemhop/pred_jemhop.json")
    args = ap.parse_args()

    os.environ["PG_COLLECTION"] = args.collection
    from graphrag_core.config import reset_settings, get_settings
    reset_settings()
    s = get_settings()

    from graphrag_core.llm.factory import create_chat_llm, create_embeddings
    from langchain_postgres import PGVector
    from graphrag_core.db.utils import add_connection_timeout

    embeddings = create_embeddings()
    llm = create_chat_llm(temperature=0)
    try:
        llm.request_timeout = 90
        llm.max_retries = 3
    except Exception:
        pass
    pg = add_connection_timeout(s.pg_conn, timeout=30)
    vstore = PGVector(connection=pg, embeddings=embeddings, collection_name=args.collection)
    vretr = vstore.as_retriever(search_kwargs={"k": args.fetch_k})

    graph = None
    if args.kg:
        from langchain_neo4j import Neo4jGraph
        graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw,
                           enhanced_schema=False)

    config = {
        "retrieval_top_k": args.top_k,
        "rerank_pool_size": args.fetch_k,
        "enable_japanese_search": True,
        "enable_rerank": True,
        "enable_entity_vector": False,
        "search_mode": "hybrid",
        "graph_hop_count": args.graph_hop,
        # KG-on: グラフ探索でbridgeエンティティを辿り、KGソースチャンクを文脈注入
        "include_kg_source_chunks": bool(args.kg),
        "include_graph_lines": False,   # tripleはLLM contextに出さない(FJH知見)
        "enable_reference_follow": False,
    }

    tasks = load_qa()
    if args.limit:
        tasks = tasks[:args.limit]
    mode = "KG-on" if args.kg else "KG-off"
    print(f"=== JEMHopQA dev runner ({mode}) ===")
    print(f"  Q={len(tasks)} collection={args.collection} top_k={args.top_k} fetch_k={args.fetch_k} hop={args.graph_hop}")

    results = [None] * len(tasks)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, tasks[i], graph, llm, embeddings, vretr,
                          s.pg_conn, args.collection, config): i
                for i in range(len(tasks))}
        done = 0
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} ({(time.time()-t0)/done:.1f}s/q avg)", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    succ = sum(1 for r in results if r["success"])
    print(f"完了: {time.time()-t0:.0f}s success={succ}/{len(results)} -> {args.output}")


if __name__ == "__main__":
    main()
