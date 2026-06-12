"""KG-on vs KG-off のペア比較（同一質問での勝敗）"""
import json, sys
from pathlib import Path

kg_on = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))   # KG-on result
kg_off = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))  # KG-off result

def by_q(res):
    d = {}
    for item in res["details"]:
        q = item["qa_data"]["question"]
        d[q] = item["final_evaluation"] == "correct"
    return d

on = by_q(kg_on)
off = by_q(kg_off)
common = [q for q in on if q in off]

both = sum(1 for q in common if on[q] and off[q])
neither = sum(1 for q in common if not on[q] and not off[q])
on_only = [q for q in common if on[q] and not off[q]]    # KGが救った
off_only = [q for q in common if not on[q] and off[q]]   # KGが壊した

print(f"共通質問: {len(common)}")
print(f"  両方正解 : {both}")
print(f"  両方不正解: {neither}")
print(f"  KG-onのみ正解 (KGが救済): {len(on_only)}")
print(f"  KG-offのみ正解 (KGが破壊): {len(off_only)}")
print(f"  net KG効果: {len(on_only) - len(off_only):+d}問")
print(f"  KG-on accuracy : {sum(on.values())}/{len(on)}")
print(f"  KG-off accuracy: {sum(off.values())}/{len(off)}")

print("\n--- KGが救済した質問 (KG-on正解/KG-off不正解) ---")
for q in on_only:
    print(f"  ✅ {q[:70]}")
print("\n--- KGが破壊した質問 (KG-on不正解/KG-off正解) ---")
for q in off_only:
    print(f"  ❌ {q[:70]}")
