#!/usr/bin/env python
"""Fujitsu-RAG-Hard-Benchmark を 完全なGraphRAGパイプライン (KG込み) で回す

fujitsu_runner.py との違い:
- HybridRetriever 直接呼び出し → retriever_and_merge() 全体に切替
- Neo4j のグラフ探索 + KG source chunks 取得 + 全 reranker 段が動く
- predicted_references は vector_sources + kg_source_chunks の双方から抽出

Usage:
    python _bench/fujitsu_runner_kg.py
    python _bench/fujitsu_runner_kg.py --top-k 5 --rerank --fetch-k 10
    python _bench/fujitsu_runner_kg.py --collection fjrag_hard --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))


GENERATION_PROMPT = """あなたは与えられた【参考情報】に基づいて質問に答えるアシスタントです。

【参考情報】
{context}

【質問】
{question}

【回答ルール】
- 参考情報に書かれている内容だけを根拠に答えてください
- まず参考情報を最後まで読み、質問に関連する記述があれば必ずそれを基に答えてください
- 完全な答えが無くても、分かる範囲・部分的な答えで構わないので必ず答えを試みてください
- 参考情報が質問と完全に無関係で何の手がかりも無い場合に限り「回答に必要な情報源が存在しないため回答不可」と答えてください
- 推測や一般常識による補完はしないでください (参考情報に書かれた根拠だけを使う)
- 結論を最初に述べ、必要なら根拠を添えてください

【特定パターンの注意】
- 「最初/最初に/初めて」「最も古い」を聞かれた場合: 関連する年代記載を全て拾い、時系列で最も古いものを特定する
- 「全て教えて/列挙して」を聞かれた場合: 該当箇所の関連項目を漏れなく列挙する。1つの見落としで不正解になる

【回答】"""


def load_tasks(yaml_path: Path) -> list[dict]:
    import yaml
    return yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["tasks"]


def extract_correct_refs(task: dict) -> list[dict]:
    refs = []
    for r in task.get("rationales") or []:
        fn = r.get("file_name")
        if not fn:
            continue
        for p in r.get("pages") or []:
            pn = p.get("number")
            if pn is None:
                continue
            refs.append({"pdf": fn, "page": str(pn)})
    return refs


def collect_refs_from_docs(docs: list) -> list[dict]:
    """vector_sources / kg_source_chunks の Document リストから (pdf, page) を抽出"""
    seen, refs = set(), []
    for d in docs:
        m = getattr(d, "metadata", None) or {}
        src = m.get("source"); pg = m.get("page")
        if src is None or pg is None:
            continue
        key = (src, str(pg))
        if key in seen:
            continue
        seen.add(key)
        refs.append({"pdf": src, "page": str(pg)})
    return refs


def run_one(task: dict, graph, llm, embeddings, vector_retriever,
            pg_conn: str, pg_collection: str, config: dict):
    """1問処理 → sample.json エントリ"""
    from graphrag_core.retrieval.pipeline import retriever_and_merge

    question = task["question"]
    try:
        merge = retriever_and_merge(
            question, graph, llm, embeddings, vector_retriever,
            pg_conn, pg_collection, config,
        )
        context = merge.get("context", "")
        vector_sources = merge.get("vector_sources") or []
        kg_chunks = merge.get("kg_source_chunks") or []

        # predicted_references: vector_sources + kg_source_chunks 両方から
        predicted_refs = collect_refs_from_docs(list(vector_sources) + list(kg_chunks))

        # 回答生成
        prompt = GENERATION_PROMPT.format(context=context, question=question)
        resp = llm.invoke(prompt)
        predicted_answer = resp.content.strip()
        success = True
    except Exception as e:
        predicted_answer = f"[runner error] {e}"
        predicted_refs = []
        success = False

    return {
        "question": question,
        "predicted_answer": predicted_answer,
        "correct_answer": task["answer"],
        "predicted_references": predicted_refs,
        "correct_references": extract_correct_refs(task),
        "success": success,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="../Fujitsu-RAG-Hard-Benchmark/dataset/FJ_KGQA_Hard.yaml")
    ap.add_argument("--collection", default="fjrag_hard")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--fetch-k", type=int, default=10, help="vector retriever の初期k (内部rerankで5に絞られる)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", default="_bench/fujitsu_predictions_kg.json")
    ap.add_argument("--graph-hop", type=int, default=2)
    ap.add_argument("--graph-lines-off", action="store_true",
                    help="KG使うが LLM context には triples を出さない (仮説2)")
    ap.add_argument("--entity-vector-off", action="store_true",
                    help="entity vector検索を無効化 (LLM抽出のみ起点)")
    args = ap.parse_args()

    yaml_path = Path(args.yaml)
    if not yaml_path.is_absolute():
        yaml_path = (_proj / args.yaml).resolve()
    if not yaml_path.exists():
        print(f"YAMLが見つかりません: {yaml_path}"); sys.exit(1)

    # Fujitsu のコレクションをパイプラインに使わせるため env を上書き
    os.environ["PG_COLLECTION"] = args.collection
    from graphrag_core.config import reset_settings, get_settings
    reset_settings()
    s = get_settings()
    print(f"PG_COLLECTION (override): {s.pg_collection}")

    tasks = load_tasks(yaml_path)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"=== Fujitsu KG-RAG Runner ===")
    print(f"  問題数: {len(tasks)}")
    print(f"  collection: {args.collection}, top_k: {args.top_k}, fetch_k: {args.fetch_k}")
    print(f"  graph_hop: {args.graph_hop}")

    # 構成
    from graphrag_core.llm.factory import create_chat_llm, create_embeddings
    from langchain_neo4j import Neo4jGraph
    from langchain_postgres import PGVector
    from graphrag_core.db.utils import add_connection_timeout

    embeddings = create_embeddings()
    llm = create_chat_llm(temperature=0)
    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw, enhanced_schema=False)
    pg_conn = add_connection_timeout(s.pg_conn, timeout=30)
    vector_store = PGVector(connection=pg_conn, embeddings=embeddings, collection_name=args.collection)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": args.fetch_k})

    # retriever_and_merge に渡す config
    config = {
        "graph_hop_count": args.graph_hop,
        "retrieval_top_k": args.top_k,
        "enable_japanese_search": True,
        "enable_rerank": False,  # cross-encoder reranker は内部で起動 (RERANKER_ENABLED env)
        "enable_entity_vector": not args.entity_vector_off,
        "entity_similarity_threshold": s.entity_similarity_threshold,
        "search_mode": "hybrid",
        "include_kg_source_chunks": True,
        "path_max_candidates": s.path_max_candidates,
        "include_graph_lines": not args.graph_lines_off,
    }

    results = [None] * len(tasks)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(run_one, tasks[i], graph, llm, embeddings, vector_retriever,
                      s.pg_conn, args.collection, config): i
            for i in range(len(tasks))
        }
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {
                    "question": tasks[i]["question"],
                    "predicted_answer": f"[outer error] {e}",
                    "correct_answer": tasks[i]["answer"],
                    "predicted_references": [],
                    "correct_references": extract_correct_refs(tasks[i]),
                    "success": False,
                }
            done += 1
            if done % 5 == 0 or done == len(tasks):
                avg = (time.time() - t0) / done
                eta = avg * (len(tasks) - done)
                print(f"  {done}/{len(tasks)} avg={avg:.1f}s ETA={eta:.0f}s")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完了: {time.time()-t0:.1f}s → {args.output}")
    success = sum(1 for r in results if r["success"])
    print(f"  success: {success}/{len(results)}")


if __name__ == "__main__":
    main()
