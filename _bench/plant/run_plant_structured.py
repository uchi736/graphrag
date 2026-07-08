#!/usr/bin/env python
"""plant_v15 を KG-off 実検索 + 「様式リスト構造化注入」で評価（本番測定）。

取り込み側で様式チャンクを構造化した場合と同等の効果を、実パイプラインで測る:
- 通常の KG-off 検索（BM25+vector+rerank）で context を作る
- 検索で取れた文書(source)のうち様式一覧を持つものは、per-doc 構造化テーブル
  （様式第N号 → 関係条文）を context 先頭に注入する（潰れた平文の代わり）
- 生成は gemma。judge も gemma（無料）。

Usage:
    LLM_PROVIDER=vllm VLLM_MODEL=google/gemma-4-31B-it \
      python _bench/plant/run_plant_structured.py
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_proj = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_proj))
from dotenv import load_dotenv; load_dotenv()
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from _bench.plant.run_plant import GEN_PROMPT, load_qa, correct_refs

_Z2H = str.maketrans("０１２３４５６７８９", "0123456789")
_PAT = re.compile(r"様式第([０-９0-9]+)(?:号)?（([^）]*?関係)）")


def build_yoshiki_index():
    """chunks_plant から per-doc の {doc_id: {様式第N号: 関係条文}} を構築（ルールベース）。"""
    idx = {}
    for fp in glob.glob(r"C:/work/makedataset/data/chunks_plant/*.jsonl"):
        for line in open(fp, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line); did = r.get("doc_id"); t = r.get("text", "")
            for m in _PAT.finditer(t):
                idx.setdefault(did, {})[f"様式第{m.group(1).translate(_Z2H)}号"] = f"（{m.group(2)}）"
    return idx


def structured_block(sources, idx):
    """検索で取れた文書のうち様式一覧を持つものだけ、構造化テーブルを返す。"""
    blocks = []
    for d in sorted(set(sources)):
        if d in idx:
            lines = "\n".join(f"  {y} → {rel}" for y, rel in idx[d].items())
            blocks.append(f"【{d} 様式一覧（構造化: 様式番号→関係条文）】\n{lines}")
    return "\n\n".join(blocks)


def run_one(task, llm, embeddings, vretr, pg_conn, coll, config, idx):
    from graphrag_core.retrieval.pipeline import retriever_and_merge
    q = task["question"]
    try:
        merge = retriever_and_merge(q, None, llm, embeddings, vretr, pg_conn, coll, config)
        ctx = merge.get("context", "")
        docs = merge.get("vector_sources") or []
        srcs, pred_refs, seen = [], [], set()
        for d in docs:
            did = (d.metadata or {}).get("source")
            if did:
                srcs.append(did)
                if did not in seen:
                    seen.add(did); pred_refs.append(did)
        # ★ 取れた文書の様式一覧を構造化して context 先頭に注入
        sb = structured_block(srcs, idx)
        ctx2 = (sb + "\n\n" + ctx) if sb else ctx
        ans = llm.invoke(GEN_PROMPT.format(context=ctx2, question=q)).content.strip()
        ok = True
    except Exception as e:
        ans, pred_refs, ok = f"[runner error] {e}", [], False
    return {"qa_id": task.get("qa_id"), "question": q, "predicted_answer": ans,
            "correct_answer": task["answer"], "predicted_references": pred_refs,
            "correct_references": correct_refs(task), "retrieval_level": task.get("retrieval_level"),
            "answer_level": task.get("answer_level"), "kg_query_type": task.get("kg_query_type"),
            "success": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="plant_v15")
    ap.add_argument("--qa-path", default="C:/work/makedataset/data/dataset_kg_v3.jsonl")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--fetch-k", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--output", default="_bench/plant/pred_v3_structured.json")
    args = ap.parse_args()

    os.environ["PG_COLLECTION"] = args.collection
    from graphrag_core.config import reset_settings, get_settings
    reset_settings(); s = get_settings()
    from graphrag_core.llm.factory import create_chat_llm, create_embeddings
    from langchain_postgres import PGVector
    from graphrag_core.db.utils import add_connection_timeout

    embeddings = create_embeddings()
    llm = create_chat_llm(temperature=0)
    try: llm.request_timeout = 150; llm.max_retries = 3
    except Exception: pass
    pg = add_connection_timeout(s.pg_conn, timeout=30)
    vstore = PGVector(connection=pg, embeddings=embeddings, collection_name=args.collection)
    vretr = vstore.as_retriever(search_kwargs={"k": args.fetch_k})

    config = {"retrieval_top_k": args.top_k, "rerank_pool_size": args.fetch_k,
              "enable_japanese_search": True, "enable_rerank": True, "enable_entity_vector": False,
              "search_mode": "hybrid", "include_kg_source_chunks": False,
              "include_graph_lines": False, "enable_reference_follow": False}

    idx = build_yoshiki_index()
    print(f"様式index: {len(idx)}文書, 総エントリ={sum(len(v) for v in idx.values())}")
    tasks = load_qa(args.qa_path)
    print(f"=== structured runner === Q={len(tasks)} model={os.getenv('VLLM_MODEL')} llm={s.llm_provider}")

    results = [None] * len(tasks)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, tasks[i], llm, embeddings, vretr, s.pg_conn, args.collection, config, idx): i
                for i in range(len(tasks))}
        done = 0
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result(); done += 1
            print(f"  {done}/{len(tasks)} ({(time.time()-t0)/done:.1f}s/q)", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"完了 {time.time()-t0:.0f}s success={sum(1 for r in results if r['success'])}/{len(results)} -> {args.output}")


if __name__ == "__main__":
    main()
