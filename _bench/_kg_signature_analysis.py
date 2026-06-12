"""KGでしか解けないQAが解けているかの分析

- Azure judge での KGターゲット軸スライス (gemmaとの2judge三角測量)
- 「KG-only正解」(KG-off不正解 & KG-on正解) 質問の特定と軸タグ
- FJH-07 で特定された KGシグネチャ5問の追跡
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml

_proj = Path(__file__).resolve().parents[1]
YAML = (_proj / "../Fujitsu-RAG-Hard-Benchmark/dataset/FJ_KGQA_Hard.yaml").resolve()

tasks = yaml.safe_load(YAML.read_text(encoding="utf-8"))["tasks"]

def get_axes(t):
    rc = t.get("Reasoning Complexity", {}) or {}
    rd = t.get("Retrieval Difficulty", {}) or {}
    def v(d, k):
        return ((d.get(k, {}) or {}).get("value"))
    return {
        "depth": v(rc, "Reasoning Depth (Multi-step Reasoning)"),
        "multi_doc": v(rd, "multi-document"),
        "remote_ref": v(rd, "Remote Reference"),
        "low_loc": v(rd, "Low Locality"),
        "retrieval_level": t.get("retrieval_level"),
        "tables": v(t.get("Source Structure & Modality", {}) or {}, "Tables/Charts"),
    }

# Azure judge details: {question: 0/1}
def aj(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

az_off = aj("_bench/fujitsu_predictions_rerank_k10_v2.azure_judge.json")
az_pre = aj("_bench/fujitsu_predictions_kg_fix_k10_nolines.azure_judge.json")
az_v2c = aj("_bench/fujitsu_predictions_kg_schema_v2c.azure_judge.json")

# gemma judge details: question_number順 (YAML順と一致, 100/100成功)
def gj(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {x["qa_data"]["question"]: (1 if x["final_evaluation"] == "correct" else 0)
            for x in d["details"]}

gm_off = gj("_bench/results/evaluation_result_20260611_041333.json")
gm_pre = gj("_bench/results/evaluation_result_20260611_041254.json")
gm_v2c = gj("_bench/results/evaluation_result_20260611_070457.json")

# ── 1. Azure judge での KGターゲット軸スライス ──
def slice_acc(judge, axis_fn):
    cor = tot = 0
    for t in tasks:
        q = t["question"]
        if q not in judge or not axis_fn(get_axes(t)):
            continue
        tot += 1
        cor += judge[q]
    return f"{100*cor/tot:5.1f}% ({cor}/{tot})" if tot else "---"

AXES = [
    ("reasoning_depth=multi", lambda a: a["depth"] == "multi"),
    ("multi_document=True",   lambda a: a["multi_doc"] is True),
    ("remote_reference=True", lambda a: a["remote_ref"] is True),
    ("low_locality=True",     lambda a: a["low_loc"] is True),
    ("retrieval_level=Hard",  lambda a: a["retrieval_level"] == "Hard"),
    ("tables_charts=False",   lambda a: a["tables"] is False),
]
print("=== Azure judge: KGターゲット軸スライス ===")
print(f"{'axis':25s} {'KG-off':18s} {'KG-pre':18s} {'KG-v2c':18s}")
for name, fn in AXES:
    print(f"{name:25s} {slice_acc(az_off, fn):18s} {slice_acc(az_pre, fn):18s} {slice_acc(az_v2c, fn):18s}")

# ── 2. KG-only正解セット (off不正解 & v2c正解) ──
print("\n=== KG-onlyで正解した質問 (v2c) ===")
both_judges = []
either = []
for t in tasks:
    q = t["question"]
    if q not in az_off or q not in az_v2c:
        continue
    az_gain = az_off[q] == 0 and az_v2c[q] == 1
    gm_gain = gm_off.get(q) == 0 and gm_v2c.get(q) == 1
    if az_gain and gm_gain:
        both_judges.append((q, get_axes(t)))
    elif az_gain or gm_gain:
        either.append((q, get_axes(t), "azure" if az_gain else "gemma"))

print(f"\n両judge一致でKGのみ正解: {len(both_judges)}問")
for q, a in both_judges:
    tags = [k for k, v in [("multi", a["depth"] == "multi"), ("multi_doc", a["multi_doc"]),
                           ("remote_ref", a["remote_ref"]), ("Hard", a["retrieval_level"] == "Hard")] if v]
    print(f"  ✅ [{','.join(tags) or '-'}] {q[:65]}")
print(f"\n片judgeのみでKG正解: {len(either)}問")
for q, a, j in either:
    print(f"  ({j}) {q[:60]}")

# 逆: KGで壊した (off正解 & v2c不正解, 両judge一致)
broken = []
for t in tasks:
    q = t["question"]
    if q in az_off and az_off[q] == 1 and az_v2c.get(q) == 0 \
       and gm_off.get(q) == 1 and gm_v2c.get(q) == 0:
        broken.append(q)
print(f"\n両judge一致でKGが壊した: {len(broken)}問")
for q in broken:
    print(f"  ❌ {q[:65]}")

# ── 3. FJH-07 KGシグネチャ5問の追跡 ──
print("\n=== KGシグネチャ問題 (FJH-07でKGのみが解けた5問) ===")
SIGNATURES = {
    "Q17 時系列最少(混信件数)": "混信",
    "Q45 多文書比較(専修学校 東京vs大阪)": "専修学校",
    "Q52 列挙(女性教員の多い校種)": "女性教員",
    "Q65 数値+期間(3Q売上と前年比)": "売上収益と前年比",
    "Q67 entity-relation(副社長)": "副社長",
}
for label, pat in SIGNATURES.items():
    qs = [t["question"] for t in tasks if pat in t["question"]]
    if not qs:
        print(f"  {label}: 質問が見つからない")
        continue
    q = qs[0]
    def f(j):
        return {1: "○", 0: "×", None: "?"}[j.get(q)]
    print(f"  {label}")
    print(f"    azure: off={f(az_off)} pre={f(az_pre)} v2c={f(az_v2c)} | gemma: off={f(gm_off)} pre={f(gm_pre)} v2c={f(gm_v2c)}")
