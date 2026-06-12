"""第2judge: Azure gpt-4.1-mini で予測ファイル群を採点（gemma自己判定の検証用）"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()

from graphrag_core.llm.factory import create_chat_llm

PROMPT = """あなたは質問応答の採点官です。
与えられた「質問」と「正しい回答」を踏まえて、「回答」の「正確性」を判定してください。
判定結果は、正確なら"1" 不正確なら"0" のどちらか1語だけを出力してください。

質問: {question}
正しい回答: {reference_answer}
回答: {answer}
"""

llm = create_chat_llm(temperature=0)

def judge_one(item):
    try:
        r = llm.invoke(PROMPT.format(
            question=item["question"],
            reference_answer=item["correct_answer"],
            answer=item["predicted_answer"],
        ))
        raw = r.content.strip()
        return 1 if raw.startswith("1") else 0
    except Exception:
        return -1

results = {}
for path in sys.argv[1:]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [d for d in data if d.get("success")]
    with ThreadPoolExecutor(max_workers=4) as ex:
        scores = list(ex.map(judge_one, items))
    ok = scores.count(1)
    n = len([s for s in scores if s >= 0])
    results[Path(path).name] = (ok, n)
    # 質問→判定 のマップも保存（ペア比較用）
    detail = {it["question"]: s for it, s in zip(items, scores)}
    out = Path(path).with_suffix(".azure_judge.json")
    out.write_text(json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{Path(path).name}: {ok}/{n} = {100*ok/n:.1f}%", flush=True)

print("\n=== Azure gpt-4.1-mini judge summary ===")
for name, (ok, n) in results.items():
    print(f"  {name}: {100*ok/n:.1f}%")
