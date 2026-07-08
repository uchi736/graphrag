#!/usr/bin/env python
"""Closed-book ablation: 検索を一切使わず、生成LLMの知識だけで回答させる。

目的: 公開法令データの「記憶（パラメトリック知識）」で正解できてしまう
      リークがどの程度あるかを測り、open-book 70% との差で検索寄与を切り分ける。

出力は eval_plant.py がそのまま採点できる pred 形式。
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))
from dotenv import load_dotenv
load_dotenv()

# 知識を積極的に使わせる中立プロンプト（接地縛りなし）。
CB_PROMPT = """あなたは日本の法令・技術基準（ボイラー・圧力容器・溶接等）に精通した専門家です。
参考資料は与えられません。あなた自身の知識だけで、次の質問にできる限り具体的に答えてください。
条文番号・様式番号・規格名が分かる場合はそのまま示してください。分かる範囲で必ず答えを試みてください。

【質問】
{question}

【回答】"""


def load_qa(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def correct_refs(task):
    seen, out = set(), []
    for r in task.get("rationale") or []:
        did = r.get("doc_id")
        if did and did not in seen:
            seen.add(did); out.append(did)
    return out


def run_one(task, llm):
    q = task["question"]
    try:
        resp = llm.invoke(CB_PROMPT.format(question=q))
        ans = resp.content.strip(); ok = True
    except Exception as e:
        ans, ok = f"[error] {e}", False
    return {
        "qa_id": task.get("qa_id"), "question": q,
        "predicted_answer": ans, "correct_answer": task["answer"],
        "predicted_references": [],  # closed-book: 検索なし
        "correct_references": correct_refs(task),
        "retrieval_level": task.get("retrieval_level"),
        "answer_level": task.get("answer_level"),
        "kg_query_type": task.get("kg_query_type"),
        "success": ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-path", default="C:/work/makedataset/data/dataset_kg_v3.jsonl")
    ap.add_argument("--output", default="_bench/plant/pred_v3_closedbook.json")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    from graphrag_core.llm.factory import create_chat_llm
    llm = create_chat_llm(temperature=0)
    try:
        llm.request_timeout = 90; llm.max_retries = 3
    except Exception:
        pass

    tasks = load_qa(args.qa_path)
    print(f"=== closed-book runner === Q={len(tasks)} qa={args.qa_path}")
    results = [None] * len(tasks)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, tasks[i], llm): i for i in range(len(tasks))}
        done = 0
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result(); done += 1
            print(f"  {done}/{len(tasks)} ({(time.time()-t0)/done:.1f}s/q)", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"done {time.time()-t0:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
