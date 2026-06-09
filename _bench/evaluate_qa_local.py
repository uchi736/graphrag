"""evaluate_qa.py の vLLM 向けフォーク (Apache 2.0)

元: Fujitsu-RAG-Hard-Benchmark/evaluate/evaluate_qa.py
変更点: ChatOpenAI のエンドポイントを vLLM (OpenAI互換) に向ける
        モデル名と base_url を環境変数で指定可能に
変更点以外は元のロジックを保持 (Apache 2.0 準拠)

Usage:
    JUDGE_BASE_URL=http://192.168.0.250:8000/v1 \\
    JUDGE_MODEL=google/gemma-4-26B-A4B-it \\
    JUDGE_API_KEY=dummy \\
    python _bench/evaluate_qa_local.py \\
      --qa-results-file _bench/fujitsu_predictions.json \\
      --reference-eval-mode full-coverage
"""
import json
import logging
import argparse
import os
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from datetime import datetime

logger = logging.getLogger(__name__)

# vLLM (OpenAI互換) の設定を環境変数から
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "google/gemma-4-26B-A4B-it")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://192.168.0.250:8000/v1")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "dummy")

MODEL_SETTINGS = {"model": JUDGE_MODEL}

BASIC_ANSWER_SIMILARITY_PROMPT = (
"""
あなたは質問応答の採点官です。
与えられた「質問」と「正しい回答」を踏まえて、「回答」の「正確性」を判定してください。
判定結果は、正確なら"1" 不正確なら"0" のどちらか1語だけを出力してください。

質問: {question}
正しい回答: {reference_answer}
回答: {answer}
""")


def basic_evaluate(questions, generated_answers, target_answers):
    chat = ChatOpenAI(
        model=MODEL_SETTINGS["model"],
        base_url=JUDGE_BASE_URL,
        api_key=JUDGE_API_KEY,
        temperature=0,
        max_tokens=8,
    )

    evals = []
    for q, tgt, gen in zip(tqdm(questions), target_answers, generated_answers):
        try:
            prompt = BASIC_ANSWER_SIMILARITY_PROMPT.format(
                question=q, reference_answer=tgt, answer=gen,
            )
            result = chat.invoke(prompt)
            raw = result.content.strip()
            # 出力先頭の0 or 1 を拾う (LLMが余計な文字を付けるケースに対応)
            digit = "1" if raw.startswith("1") else "0" if raw.startswith("0") else ""
            if not digit:
                # フォールバック: 文字列内に "1" があれば 1
                digit = "1" if "1" in raw[:5] else "0"
            evals.append(int(digit))
        except Exception as e:
            logger.warning(f"llm_eval exception: {e}")
            evals.append(-1)
    return evals


def load_qa_results(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions, predicted, correct = [], [], []
    success_count = 0
    for item in data:
        if item.get("success", False):
            success_count += 1
            questions.append(item["question"])
            predicted.append(item["predicted_answer"])
            correct.append(item["correct_answer"])
    return questions, predicted, correct, success_count


def evaluate_references(predicted_refs, correct_refs):
    not_found = []
    match = 0
    total = len(correct_refs)
    for c in correct_refs:
        cp = c["pdf"].strip("'"); cpage = c["page"]
        found = False
        for p in predicted_refs:
            if p["pdf"].strip("'") == cp and p["page"] == cpage:
                found = True
                break
        if found: match += 1
        else: not_found.append({"pdf": cp, "page": cpage})
    return (match / total if total > 0 else 0.0, not_found)


def evaluate_references_full_coverage(predicted_refs, correct_refs):
    pset = {(p["pdf"].strip("'"), p["page"]) for p in predicted_refs}
    not_found = []
    for c in correct_refs:
        if (c["pdf"].strip("'"), c["page"]) not in pset:
            not_found.append({"pdf": c["pdf"].strip("'"), "page": c["page"]})
    return (1.0 if not not_found else 0.0, not_found)


def main(qa_file, reference_eval_mode):
    if not os.path.exists(qa_file):
        print(f"エラー: {qa_file} が見つかりません"); return
    os.makedirs("_bench/results", exist_ok=True)

    questions, predicted_answers, correct_answers, success_count = load_qa_results(qa_file)
    if not questions:
        print("評価対象0件"); return
    print(f"成功した質問数: {success_count}")
    print(f"Judge: model={JUDGE_MODEL}, base_url={JUDGE_BASE_URL}")

    eval_list = basic_evaluate(questions, predicted_answers, correct_answers)

    ref_fn = evaluate_references if reference_eval_mode == "match-rate" else evaluate_references_full_coverage
    print(f"参照評価モード: {reference_eval_mode}")

    total = len(eval_list)
    correct = eval_list.count(1)
    accuracy = correct / total * 100

    with open(qa_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    ref_results, ref_details = [], []
    for item in data:
        if item.get("success", False):
            res, nf = ref_fn(item["predicted_references"], item["correct_references"])
            ref_results.append(res)
            ref_details.append({
                "correct_refs": item["correct_references"],
                "predicted_refs": item["predicted_references"],
                "not_found": nf,
            })
    ref_total = len(ref_results)
    ref_correct = sum(ref_results)
    ref_accuracy = (ref_correct / ref_total * 100) if ref_total > 0 else 0

    results_json = {
        "model": MODEL_SETTINGS["model"],
        "answer_evaluation": {
            "total_question": total, "correct": correct, "accuracy": round(accuracy, 2),
        },
        "reference_evaluation": {
            "mode": reference_eval_mode, "total_question": ref_total,
            "accuracy": round(ref_accuracy, 2),
        },
        "details": [],
    }
    for i, (q, p, c, e, rr, rd) in enumerate(zip(
        questions, predicted_answers, correct_answers, eval_list, ref_results, ref_details
    )):
        results_json["details"].append({
            "question_number": i + 1,
            "final_evaluation": "correct" if e == 1 else "incorrect",
            "qa_data": {"question": q, "predicted_answer": p, "correct_answer": c},
            "reference_evaluation": {
                "result": rr,
                "correct_references": [{"pdf": r["pdf"].strip(), "page": r["page"]} for r in rd["correct_refs"]],
                "predicted_references": [{"pdf": r["pdf"].strip(), "page": r["page"]} for r in rd["predicted_refs"]],
                "not_found_references": [{"pdf": r["pdf"], "page": r["page"]} for r in rd.get("not_found", [])],
            },
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"_bench/results/evaluation_result_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    print(f"\n=== 結果 ===")
    print(f"  回答評価 accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  参照評価 accuracy: {ref_accuracy:.2f}% ({ref_correct}/{ref_total})  mode={reference_eval_mode}")
    print(f"  詳細: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-results-file", required=True)
    parser.add_argument(
        "--reference-eval-mode",
        choices=["match-rate", "full-coverage"],
        default="full-coverage",
    )
    args = parser.parse_args()
    main(args.qa_results_file, args.reference_eval_mode)
