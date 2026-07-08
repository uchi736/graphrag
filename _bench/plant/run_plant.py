#!/usr/bin/env python
"""plant_v15（25問）を graphrag の検索パイプラインで評価用に実行する。

KG-off モード（graph=None）: hybrid検索（BM25+vector RRF）+ cross-encoder rerank。
出力は採点用 JSON（predicted_answer + predicted_references[doc_id] + 正解）。

Usage:
    python _bench/plant/run_plant.py
    python _bench/plant/run_plant.py --top-k 10 --fetch-k 20
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

QA_PATH = Path("C:/work/makedataset/data/reviewed/plant_v15.jsonl")

# プロンプトトーナメント（_bench/plant、2026-06-13）で確定した graft 版。
# baseline 72% → 92%(23/25)、6失敗中5回復・退行0。
# 設計: v3の3ステップ構造 + v2の強い接地・反捏造文言。詳細は _bench/plant/RESULTS.md。
GEN_PROMPT = """あなたは法令・技術基準の条文を読み解き、与えられた【参考情報】の記載だけを根拠に質問へ答える専門アシスタントです。憶測や一般知識による補完を一切せず、参考情報に書かれた語・番号・条件をそのまま使って答えます。

【参考情報】
{context}

【質問】
{question}

【最優先の原則：接地と反捏造】
- 参考情報に明示された文言だけを根拠にする。参考情報に無い数値基準・規格名・JIS番号・別表番号・別図番号・様式番号・条文番号は、たとえ常識的に存在しそうでも創作・補完しない。
- 規格名・JIS番号・別表/別図番号・様式番号・条文番号は、参考情報に書かれているとおりに一字一句転記する（番号を推測で付け替えない／存在しない番号を足さない）。
- 原文が具体値を別の標準や条文に委譲している場合（例：「JISの規定による」「別に定める」）、勝手に具体値を当てはめず、その委譲先（標準名・条文）をそのまま答える。
- 根拠が参考情報内に無いものは「参考情報には記載がない」と述べ、作文しない。

【回答の作り方】次の3ステップの順序で組み立ててください。

▼ステップ1：準拠する規定・標準・条文を先に特定する
- まず「この質問に答える根拠となる規定」を列挙する。準拠・準用すべき条文番号、標準名（JIS番号・規格名）、別表・別図・様式番号を、参考情報に書かれているとおり逐語で挙げる。
- 「何に従うか／準拠／準用／様式」を問う質問では、具体値を並べる前に、従うべき標準名・別表/別図/JIS番号・準用関係そのものを必ず先に明示する。
- 設備名を含む見出し条文が無くても、「準用する」「…の規定による」等の準用や、一般規定が適用されるなら、それを根拠として採用する。見出しに対象設備名が無いことだけを理由に「規定が存在しない」と結論してはいけない。
- 様式を問う場合は、根拠条文に対応する「様式第N号（第N条関係）」をそのまま転記し、別目的の申請様式で代替しない。

▼ステップ2：条件分岐・例外・適用範囲を整理する
- 質問が特定の条件・対象に限定しているなら、その条件の枝だけを扱い、条件外の枝（常時適用・別条件・別対象の規則）を例外として混入させない。
- 対象が複数（例：皿形／全半球形／半だ円体形 等）で規則が異なる場合は、各対象に対応する規則（式そのまま・係数を代入・別式）を取り違えず、対象ごとに分けて示す。場合分けを網羅して問われたときに限り「その他の場合」「上記以外」等のデフォルト枝も明記する（質問が網羅を求めていない場合は該当枝だけ答える）。
- 「適用されない条件／例外／ただし書」を問われたら、原文の例外句を漏れなく逐語で列挙する。具体的には『ただし』『…は除く』『この限りでない』『…の場合に限る』『…に適合すること』等の語と、それに付く条件・条文番号をすべて拾う。
- 肯定文を否定形へ反転する際は意味を取り違えない（「適用する」⇔「適用しない」、「必要」⇔「不要」を正確に対応させる）。

▼ステップ3：結論
- ステップ1・2を踏まえ、質問に対する答えを簡潔に述べる。例外・条件・準拠先は原文の語のまま添える。
- 質問の主題（対象設備・対象事項）から逸脱せず、表面的に一致するキーワードに引きずられて無関係な規定とすり替えない。
- 分かる範囲で必ず答えを試み、根拠が参考情報に無い部分のみ「記載がない」と述べる。

【回答】"""


def load_qa(path=QA_PATH):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def correct_refs(task):
    seen, out = set(), []
    for r in task.get("rationale") or []:
        did = r.get("doc_id")
        if did and did not in seen:
            seen.add(did)
            out.append(did)
    return out


def run_one(task, llm, embeddings, vector_retriever, pg_conn, collection, config, gen_prompt):
    from graphrag_core.retrieval.pipeline import retriever_and_merge
    q = task["question"]
    try:
        merge = retriever_and_merge(q, None, llm, embeddings, vector_retriever,
                                    pg_conn, collection, config)
        ctx = merge.get("context", "")
        docs = merge.get("vector_sources") or []
        pred_refs, seen = [], set()
        for d in docs:
            did = (d.metadata or {}).get("source")
            if did and did not in seen:
                seen.add(did)
                pred_refs.append(did)
        resp = llm.invoke(gen_prompt.format(context=ctx, question=q))
        ans = resp.content.strip()
        ok = True
    except Exception as e:
        ans, pred_refs, ok = f"[runner error] {e}", [], False
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
        "success": ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="plant_v15")
    ap.add_argument("--qa-path", default=str(QA_PATH), help="評価データセット(jsonl)。未指定で内蔵QA_PATH")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--fetch-k", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--output", default="_bench/plant/pred_plant_retrieval.json")
    ap.add_argument("--prompt-file", default=None,
                    help="生成プロンプトのテンプレ（{context}/{question}必須）。未指定で内蔵GEN_PROMPT")
    args = ap.parse_args()

    gen_prompt = GEN_PROMPT
    if args.prompt_file:
        gen_prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        assert "{context}" in gen_prompt and "{question}" in gen_prompt, \
            "prompt-file は {context} と {question} を含む必要があります"

    os.environ["PG_COLLECTION"] = args.collection
    from graphrag_core.config import reset_settings, get_settings
    reset_settings()
    s = get_settings()

    from graphrag_core.llm.factory import create_chat_llm, create_embeddings
    from langchain_postgres import PGVector
    from graphrag_core.db.utils import add_connection_timeout

    embeddings = create_embeddings()
    llm = create_chat_llm(temperature=0)
    # factoryのAzureChatOpenAIはtimeout未設定（langchainデフォルト=無限待ち）。
    # ハングしたAzure呼び出しがワーカースレッドを永久ブロックするのを防ぐ
    try:
        llm.request_timeout = 90
        llm.max_retries = 3
    except Exception:
        pass
    pg = add_connection_timeout(s.pg_conn, timeout=30)
    vstore = PGVector(connection=pg, embeddings=embeddings, collection_name=args.collection)
    vretr = vstore.as_retriever(search_kwargs={"k": args.fetch_k})

    config = {
        "retrieval_top_k": args.top_k,
        "rerank_pool_size": args.fetch_k,
        "enable_japanese_search": True,
        "enable_rerank": True,      # cross-encoder（RERANKER_ENABLED）で fetch→top_k
        "enable_entity_vector": False,
        "search_mode": "hybrid",
        "include_kg_source_chunks": False,
        "include_graph_lines": False,
        "enable_reference_follow": False,
    }

    tasks = load_qa(args.qa_path)
    print(f"=== plant_v15 retrieval-only runner ===")
    print(f"  Q={len(tasks)} qa={args.qa_path} collection={args.collection} top_k={args.top_k} fetch_k={args.fetch_k}")

    results = [None] * len(tasks)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, tasks[i], llm, embeddings, vretr,
                          s.pg_conn, args.collection, config, gen_prompt): i
                for i in range(len(tasks))}
        done = 0
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            print(f"  {done}/{len(tasks)} ({(time.time()-t0)/done:.1f}s/q avg)", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    succ = sum(1 for r in results if r["success"])
    print(f"完了: {time.time()-t0:.0f}s success={succ}/{len(results)} -> {args.output}")


if __name__ == "__main__":
    main()
