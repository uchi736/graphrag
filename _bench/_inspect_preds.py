"""予測JSONの健全性チェック"""
import json, sys
from pathlib import Path

path = sys.argv[1]
data = json.loads(Path(path).read_text(encoding="utf-8"))
n = len(data)
succ = sum(1 for r in data if r.get("success"))
refuse = sum(1 for r in data if "回答不可" in (r.get("predicted_answer") or ""))
empty_refs = sum(1 for r in data if not r.get("predicted_references"))
avg_refs = sum(len(r.get("predicted_references") or []) for r in data) / max(n, 1)
err = sum(1 for r in data if "error" in (r.get("predicted_answer") or "").lower())
print(f"{Path(path).name}")
print(f"  件数={n} success={succ} refuse={refuse} 回答エラー={err}")
print(f"  予測参照: 平均{avg_refs:.1f}件/問, 参照ゼロ={empty_refs}問")
# サンプル
print("  --- sample Q1 ---")
print(f"  Q: {data[0]['question'][:60]}")
print(f"  A: {(data[0]['predicted_answer'] or '')[:100]}")
print(f"  refs: {data[0].get('predicted_references')}")
