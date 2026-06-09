#!/usr/bin/env python
"""Fujitsu-RAG-Hard-Benchmark の100問を我々のRAGに通して sample.json 形式で出力

各問: HybridRetriever で top-K チャンク取得 → gemma4 で回答生成 →
  {question, predicted_answer, correct_answer, predicted_references, correct_references, success}
を JSONリストとして出力。

Usage:
    python _bench/fujitsu_runner.py
    python _bench/fujitsu_runner.py --top-k 5 --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))


GENERATION_PROMPT = """あなたは与えられた【参考文書】に基づいて質問に答えるアシスタントです。

【参考文書】
{context}

【質問】
{question}

【回答ルール】
- 参考文書に書かれている内容だけを根拠に答えてください
- 参考文書に答えるための情報が無い場合は「回答に必要な情報源が存在しないため回答不可」と答えてください
- 推測や一般常識による補完はしないでください
- 結論を最初に述べ、必要なら根拠を添えてください

【回答】"""


def load_tasks(yaml_path: Path) -> list[dict]:
    import yaml
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return data["tasks"]


def extract_correct_refs(task: dict) -> list[dict]:
    """rationales -> [{pdf, page (string)}, ...]"""
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


def run_one(task: dict, hybrid_retriever, llm, embeddings, top_k: int, search_mode: str,
            rerank: bool = False, fetch_k: int = 30):
    """1問処理 → sample.json エントリを返す

    rerank=True のとき: fetch_k 件取得 → cross-encoder で再ランキング → top_k 残す
    """
    question = task["question"]
    qvec = embeddings.embed_query(question)
    initial_k = fetch_k if rerank else top_k
    results = hybrid_retriever.search(
        query_text=question, query_vector=qvec,
        k=initial_k, search_type=search_mode,
    )
    # results: [{'id', 'text', 'metadata', 'score'}, ...]

    if rerank and len(results) > top_k:
        from graphrag_core.retrieval.reranker import score_candidates
        texts = [r["text"][:2000] for r in results]  # 長すぎはトリム (reranker context制約)
        scores = score_candidates(question, texts)
        if scores is not None and len(scores) == len(results):
            ranked = sorted(zip(scores, results), key=lambda t: -t[0])
            results = [r for _, r in ranked[:top_k]]
        else:
            # rerank失敗時はそのまま top_k 取る
            results = results[:top_k]
    predicted_refs = []
    seen = set()
    for r in results:
        m = r.get("metadata") or {}
        src = m.get("source"); pg = m.get("page")
        if src is None or pg is None:
            continue
        key = (src, str(pg))
        if key in seen:
            continue
        seen.add(key)
        predicted_refs.append({"pdf": src, "page": str(pg)})

    # context: 取得チャンクを結合
    ctx_parts = []
    for r in results:
        m = r.get("metadata") or {}
        src = m.get("source", "?"); pg = m.get("page", "?")
        ctx_parts.append(f"[出典: {src} p.{pg}]\n{r['text']}")
    context = "\n---\n".join(ctx_parts)

    prompt = GENERATION_PROMPT.format(context=context, question=question)
    try:
        resp = llm.invoke(prompt)
        predicted_answer = resp.content.strip()
        success = True
    except Exception as e:
        predicted_answer = f"[エラー] {e}"
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
    ap.add_argument("--search-mode", default="hybrid", choices=["hybrid", "vector", "keyword"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="先頭N問のみ")
    ap.add_argument("--output", default="_bench/fujitsu_predictions.json")
    ap.add_argument("--rerank", action="store_true",
                    help="cross-encoder (bge-reranker-v2-m3) で fetch_k → top_k 再ランキング")
    ap.add_argument("--fetch-k", type=int, default=30,
                    help="rerank時の1次検索数 (default: 30)")
    args = ap.parse_args()

    yaml_path = Path(args.yaml)
    if not yaml_path.is_absolute():
        yaml_path = (_proj / args.yaml).resolve()
    if not yaml_path.exists():
        print(f"YAMLが見つかりません: {yaml_path}"); sys.exit(1)

    tasks = load_tasks(yaml_path)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"=== Fujitsu-RAG-Hard-Bench Runner ===")
    print(f"  問題数: {len(tasks)}")
    print(f"  collection: {args.collection}, top_k: {args.top_k}, mode: {args.search_mode}")
    if args.rerank:
        print(f"  rerank: ON (fetch_k={args.fetch_k} → top_k={args.top_k}, "
              f"cross-encoder=bge-reranker-v2-m3)")

    from graphrag_core.config import reset_settings, get_settings
    reset_settings()
    s = get_settings()

    from graphrag_core.llm.factory import create_chat_llm, create_embeddings
    from graphrag_core.retrieval.hybrid import HybridRetriever
    embeddings = create_embeddings()
    llm = create_chat_llm(temperature=0)
    hr = HybridRetriever.get_instance(s.pg_conn, collection_name=args.collection)

    results = [None] * len(tasks)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(
                run_one, tasks[i], hr, llm, embeddings,
                args.top_k, args.search_mode,
                args.rerank, args.fetch_k,
            ): i
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
                    "predicted_answer": f"[runner error] {e}",
                    "correct_answer": tasks[i]["answer"],
                    "predicted_references": [],
                    "correct_references": extract_correct_refs(tasks[i]),
                    "success": False,
                }
            done += 1
            if done % 10 == 0 or done == len(tasks):
                avg = (time.time() - t0) / done
                eta = avg * (len(tasks) - done)
                print(f"  {done}/{len(tasks)} avg={avg:.1f}s ETA={eta:.0f}s")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完了: {time.time()-t0:.1f}s → {args.output}")

    # 簡易統計
    success = sum(1 for r in results if r["success"])
    print(f"  success: {success}/{len(results)}")
    # チャンクのソース別分布 (どのPDFがよくHITするか)
    from collections import Counter
    src_cnt = Counter()
    for r in results:
        for ref in r["predicted_references"]:
            src_cnt[ref["pdf"]] += 1
    print(f"\n  predicted_references の上位source:")
    for src, n in src_cnt.most_common(10):
        print(f"    {n:4d}  {src}")


if __name__ == "__main__":
    main()
