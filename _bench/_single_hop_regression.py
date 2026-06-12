"""single-hop質問でKG-onが落とした質問の原因診断"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml

_proj = Path(__file__).resolve().parents[1]
YAML = (_proj / "../Fujitsu-RAG-Hard-Benchmark/dataset/FJ_KGQA_Hard.yaml").resolve()
tasks = yaml.safe_load(YAML.read_text(encoding="utf-8"))["tasks"]

def depth(t):
    rc = t.get("Reasoning Complexity", {}) or {}
    return (rc.get("Reasoning Depth (Multi-step Reasoning)", {}) or {}).get("value")

def gj(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {x["qa_data"]["question"]: (1 if x["final_evaluation"] == "correct" else 0)
            for x in d["details"]}

gm_off = gj("_bench/results/evaluation_result_20260611_041333.json")
gm_v2c = gj("_bench/results/evaluation_result_20260611_070457.json")
az_off = json.loads(Path("_bench/fujitsu_predictions_rerank_k10_v2.azure_judge.json").read_text(encoding="utf-8"))
az_v2c = json.loads(Path("_bench/fujitsu_predictions_kg_schema_v2c.azure_judge.json").read_text(encoding="utf-8"))

pred_off = {x["question"]: x for x in json.loads(
    Path("_bench/fujitsu_predictions_rerank_k10_v2.json").read_text(encoding="utf-8"))}
pred_v2c = {x["question"]: x for x in json.loads(
    Path("_bench/fujitsu_predictions_kg_schema_v2c.json").read_text(encoding="utf-8"))}

singles = [t["question"] for t in tasks if depth(t) == "single"]
print(f"single-hop質問: {len(singles)}問")

flips = []
for q in singles:
    off_ok = gm_off.get(q) == 1
    v2c_ok = gm_v2c.get(q) == 1
    if off_ok and not v2c_ok:
        flips.append(q)

print(f"gemma判定で off正解→v2c不正解 の single-hop: {len(flips)}問\n")
for q in flips:
    o, n = pred_off[q], pred_v2c[q]
    oref = [(r["pdf"], r["page"]) for r in o["predicted_references"]]
    nref = [(r["pdf"], r["page"]) for r in n["predicted_references"]]
    cref = [(r["pdf"].strip("'"), r["page"]) for r in o["correct_references"]]
    added = [r for r in nref if r not in oref]
    print("=" * 80)
    print(f"Q: {q[:75]}")
    print(f"  azure判定: off={az_off.get(q)} v2c={az_v2c.get(q)}  (gemma: 1→0)")
    print(f"  正解ref: {cref}")
    print(f"  refヒット: off={len([r for r in cref if r in oref])}/{len(cref)}, "
          f"v2c={len([r for r in cref if r in nref])}/{len(cref)}")
    print(f"  ref数: off={len(oref)} → v2c={len(nref)} (KG追加分: {len(added)})")
    if added:
        print(f"  KG追加ref: {added[:6]}")
    print(f"  正解: {o['correct_answer'][:90]}")
    print(f"  off回答: {o['predicted_answer'][:130]}")
    print(f"  v2c回答: {n['predicted_answer'][:130]}")
    refuse_o = "回答不可" in o["predicted_answer"] or "存在しない" in o["predicted_answer"][:60]
    refuse_n = "回答不可" in n["predicted_answer"] or "存在しない" in n["predicted_answer"][:60]
    print(f"  refusal: off={refuse_o} v2c={refuse_n}")
