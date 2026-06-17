#!/usr/bin/env python
"""plant条件ルール診断: reification投資判断のための2シグナル（既存データのみ・非破壊）

S1: plant_v15の条件依存問を (a)/(a')/(b) に分類（既存KG-off結果 + Azure LLM分類）
S2: 全条件ルールから条件述語を抽出し、同一述語の規則横断再帰性を集計

出力: _bench/plant/condition_diagnosis.md（UTF-8）

Usage:
    python _bench/plant/_condition_diag.py            # S2のみ（高速・LLM不要）
    python _bench/plant/_condition_diag.py --s1       # S1も（Azure LLM分類）
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))
from dotenv import load_dotenv
load_dotenv()

CHUNKS = Path("C:/work/makedataset/data/chunks_plant")
QA = Path("C:/work/makedataset/data/reviewed/plant_v15.jsonl")
PRED = _proj / "_bench/plant/pred_plant_retrieval.json"
EVAL = _proj / "_bench/plant/pred_plant_retrieval.eval.json"
OUT = _proj / "_bench/plant/condition_diagnosis.md"

# ── 条件述語抽出 ──────────────────────────────────────────────────────
_KANJI = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9}
_KUNIT = {"十": 10, "百": 100, "千": 1000}


def kanji_to_num(s: str):
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", s):
        return float(s) if "." in s else int(s)
    total, cur = 0, 0
    for ch in s:
        if ch in _KANJI:
            cur = _KANJI[ch]
        elif ch in _KUNIT:
            cur = cur or 1
            total += cur * _KUNIT[ch]
            cur = 0
        elif ch == "万":
            total = (total + cur) * 10000
            cur = 0
    return total + cur


_NUM = r"(?:[0-9]+(?:\.[0-9]+)?|[〇零一二三四五六七八九十百千万]+)"
_UNIT = (r"(?:メガパスカル|MPa|キロパスカル|kPa|パスカル|ミリメートル|mm|ミリ|"
         r"センチメートル|cm|メートル|度|℃|%|％|パーセント|リットル|ℓ|l|L|"
         r"キログラム|kg|グラム|時間|年|月|日|件|名|人|回|パーミル|‰)")
_OP = r"(以上|以下|未満|を超える|を超え|超える|超)"
_PRED_RX = re.compile(rf"({_NUM})\s*({_UNIT})\s*{_OP}")
# param: 述語直前の名詞っぽい末尾（助詞/記号を落とす）
_PARTICLE = re.compile(r"[\sのがはをにでとへやもか、。「」『』（）\(\)･・,:：;；]+$")


def op_norm(op: str) -> str:
    if op in ("以上",):
        return ">="
    if op in ("以下",):
        return "<="
    if op in ("未満",):
        return "<"
    return ">"  # を超える / 超


def extract_predicates(text: str):
    """text から (param_tail, value, unit, op_norm, raw) のリストを返す"""
    out = []
    for m in _PRED_RX.finditer(text):
        raw_num, unit = m.group(1), m.group(2)
        op = op_norm(m.group(3))
        try:
            val = kanji_to_num(raw_num)
        except Exception:
            continue
        # param: 述語直前 ~10字から末尾名詞を拾う
        pre = text[max(0, m.start() - 12):m.start()]
        pre = _PARTICLE.sub("", pre)
        ptail = re.split(r"[\s、。「」『』（）\(\)･・,:：;；0-9〇-龯ぁ-ん]+", pre)
        # 末尾の漢字/カタカナ連続を param とする（雑）
        mm = re.search(r"[一-龯ァ-ヶ]{2,10}$", pre)
        param = mm.group(0) if mm else (ptail[-1] if ptail and ptail[-1] else "?")
        out.append((param, val, unit, op, m.group(0)))
    return out


# ── S2: 条件再帰性 ────────────────────────────────────────────────────
def run_s2():
    # key_loose=(val,unit,op) / key_param=(param,val,unit,op)
    loose_docs = defaultdict(set)      # key -> set(doc_id)
    loose_chunks = defaultdict(set)    # key -> set(chunk_id)
    param_docs = defaultdict(set)
    param_examples = defaultdict(list)
    n_chunks = n_cond_chunks = n_preds = 0

    for fp in sorted(glob.glob(str(CHUNKS / "*.jsonl"))):
        for line in open(fp, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            n_chunks += 1
            preds = extract_predicates(r.get("text") or "")
            if preds:
                n_cond_chunks += 1
            for param, val, unit, op, raw in preds:
                n_preds += 1
                kl = (val, unit, op)
                kp = (param, val, unit, op)
                loose_docs[kl].add(r["doc_id"])
                loose_chunks[kl].add(r["chunk_id"])
                param_docs[kp].add(r["doc_id"])
                if len(param_examples[kp]) < 3:
                    param_examples[kp].append((r["doc_id"], raw))

    # 再帰性 = 異なるdoc_idをまたぐ条件
    loose_recur = {k: v for k, v in loose_docs.items() if len(v) >= 2}
    param_recur = {k: v for k, v in param_docs.items() if len(v) >= 2}
    total_loose = len(loose_docs)
    total_param = len(param_docs)

    lines = []
    lines.append("## S2: 条件再帰性（全条件ルール・投資判断の主シグナル）\n")
    lines.append(f"- スキャンチャンク: {n_chunks} / 条件述語を含むチャンク: {n_cond_chunks} "
                 f"({100*n_cond_chunks/max(n_chunks,1):.1f}%) / 抽出述語延べ: {n_preds}\n")
    lines.append(f"- **loose key (値×単位×演算子)**: ユニーク {total_loose} / "
                 f"複数doc再帰 {len(loose_recur)} ({100*len(loose_recur)/max(total_loose,1):.0f}%)\n")
    lines.append(f"- **param key (param×値×単位×演算子)**: ユニーク {total_param} / "
                 f"複数doc再帰 {len(param_recur)} ({100*len(param_recur)/max(total_param,1):.0f}%)\n")

    # param key の doc再帰 Top20（reificationのjoin-keyターゲット候補）
    lines.append("\n### param key 再帰Top20（異なるdoc_id数 = join-key候補）\n")
    lines.append("| param | 条件 | doc数 | 例(doc_id: raw) |\n|---|---|---|---|\n")
    top = sorted(param_recur.items(), key=lambda kv: -len(kv[1]))[:20]
    for (param, val, unit, op), docs in top:
        ex = param_examples[(param, val, unit, op)]
        exs = "; ".join(f"{d}: {raw}" for d, raw in ex[:2])
        v = int(val) if float(val).is_integer() else val
        lines.append(f"| {param} | {op}{v}{unit} | {len(docs)} | {exs} |\n")

    # loose key の doc再帰 Top10（param抽出が雑な分の上限値）
    lines.append("\n### loose key 再帰Top10（param無視・再帰の上限値）\n")
    lines.append("| 条件 | doc数 | chunk数 |\n|---|---|---|\n")
    topl = sorted(loose_recur.items(), key=lambda kv: -len(kv[1]))[:10]
    for (val, unit, op), docs in topl:
        v = int(val) if float(val).is_integer() else val
        lines.append(f"| {op}{v}{unit} | {len(docs)} | {len(loose_chunks[(val,unit,op)])} |\n")

    # 判定
    recur_rate = len(param_recur) / max(total_param, 1)
    lines.append(f"\n### S2判定\n")
    lines.append(f"- param key の複数doc再帰率: **{100*recur_rate:.0f}%**\n")
    verdict = ("高 → join-keyクエリに非自明な答えセットが立つ → reificationにターゲットあり"
               if recur_rate >= 0.15 else
               "低 → 条件はほぼ葉属性 → エッジ文字列(軽量版)で十分、reification却下寄り")
    lines.append(f"- 再帰性: **{verdict}**\n")
    return "".join(lines), {
        "recur_rate": recur_rate, "param_recur": len(param_recur),
        "total_param": total_param, "top": top[:8],
    }


# ── S1: 条件依存問の失敗モード分類 ───────────────────────────────────
_COND_KW = re.compile(r"以上|以下|未満|超|の場合|のとき|ただし|に限る|を除く")


def run_s1(use_llm: bool):
    qa = [json.loads(l) for l in open(QA, encoding="utf-8") if l.strip()]
    pred = {r["qa_id"]: r for r in json.loads(PRED.read_text(encoding="utf-8"))}
    ev = {d["qa_id"]: d for d in json.loads(EVAL.read_text(encoding="utf-8"))["details"]}

    def is_conditional(q):
        if (q.get("reasoning_complexity") or {}).get("quantitative"):
            return True
        blob = (q.get("answer") or "") + " ".join(
            (rt.get("text") or "") for rt in (q.get("rationale") or []))
        return bool(_COND_KW.search(blob))

    cond_qs = [q for q in qa if is_conditional(q)]

    llm = None
    if use_llm:
        from graphrag_core.llm.factory import create_chat_llm
        llm = create_chat_llm(temperature=0)
        try:
            llm.request_timeout = 60
            llm.max_retries = 3
        except Exception:
            pass

    RUBRIC = """次のQAを分類してください。出力はJSONのみ。
{{"conditional": true/false, "structure": "a"|"a_prime"|"b", "reason": "一文"}}
- conditional: 回答が条件・閾値（〜以上/以下/の場合/ただし等）に依存するか
- structure:
  - "a": 単一ルールの分岐選択。1つの規則を読んで条件を評価すれば答えが定まる（自己完結）
  - "a_prime": 単一ルールだが、条件評価に必要な定義/閾値が別文書（参照規格の用語定義等）にあり、リンクが要る
  - "b": 条件が結合キー。複数規則を条件で横断/逆引き/網羅する必要がある（例「炭素0.30%超を条件に持つ規定を全部」）

質問: {q}
正解: {a}
根拠抜粋: {r}
"""
    rows = []
    bucket = Counter()
    bucket_fail = Counter()   # バケツ別の失敗(回答×)数 = 真の改善ターゲット
    for q in cond_qs:
        qid = q["qa_id"]
        e = ev.get(qid)
        cref = set(r["doc_id"] for r in (q.get("rationale") or []) if r.get("doc_id"))
        pref = set(pred.get(qid, {}).get("predicted_references") or [])
        hit = "○" if cref and cref <= pref else ("△" if cref & pref else "×")
        correct = ("○" if e and e["judge"] == 1 else ("×" if e else "?"))
        struct, reason = "?", ""
        if llm is not None:
            rtxt = " / ".join((rt.get("text") or "")[:200] for rt in (q.get("rationale") or [])[:3])
            try:
                resp = llm.invoke(RUBRIC.format(q=q["question"], a=q["answer"][:400], r=rtxt[:800]))
                txt = resp.content.strip()
                mjson = re.search(r"\{.*\}", txt, re.S)
                d = json.loads(mjson.group(0)) if mjson else {}
                struct = d.get("structure", "?")
                reason = d.get("reason", "")
            except Exception as ex:
                reason = f"[llm err {type(ex).__name__}]"
        # 失敗バケツ（不正解 or b構造）
        if struct == "b":
            bk = "(b)"
        elif struct == "a_prime":
            bk = "(a')"
        else:
            bk = "(a)"
        bucket[bk] += 1
        if correct == "×":
            bucket_fail[bk] += 1
        rows.append({
            "qid": qid, "q": q["question"][:50], "hit": hit, "correct": correct,
            "struct": struct, "bucket": bk, "reason": reason[:80],
            "kg_query_type": q.get("kg_query_type"),
        })

    apk = chr(40) + chr(97) + chr(39) + chr(41)  # "(a')"
    n_fail = sum(bucket_fail.values())
    lines = ["## S1: 条件依存問の失敗モード分類（plant_v15）\n"]
    lines.append(f"- 条件依存問: {len(cond_qs)}/{len(qa)}\n")
    lines.append(f"- バケツ(件数): (a)={bucket['(a)']} (a')={bucket[apk]} (b)={bucket['(b)']}\n")
    lines.append(f"- **バケツ別の失敗(回答×)数 = 真の改善ターゲット**: "
                 f"(a)={bucket_fail['(a)']} (a')={bucket_fail[apk]} (b)={bucket_fail['(b)']} "
                 f"（全失敗{n_fail}件）\n")
    lines.append(f"- → reificationが効くのは **(b)の失敗** のみ。(b)失敗={bucket_fail['(b)']}件\n\n")
    lines.append("| qa_id | 質問(略) | 検索 | 回答 | 構造 | バケツ | kg_type | 根拠 |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        lines.append(f"| {r['qid']} | {r['q']} | {r['hit']} | {r['correct']} | "
                     f"{r['struct']} | {r['bucket']} | {r['kg_query_type']} | {r['reason']} |\n")
    return "".join(lines), {"bucket": dict(bucket), "bucket_fail": dict(bucket_fail),
                            "n_cond": len(cond_qs), "n_fail": n_fail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", action="store_true", help="S1のLLM分類も実行（Azure）")
    args = ap.parse_args()

    s2_md, s2 = run_s2()
    parts = ["# plant条件ルール診断（reification投資判断）\n\n"]
    if args.s1:
        s1_md, s1 = run_s1(use_llm=True)
        parts.append(s1_md + "\n---\n\n")
    else:
        s1 = None
    parts.append(s2_md)

    # 統合判断
    parts.append("\n## 統合判断ゲート\n")
    rr = s2["recur_rate"]
    if s1:
        bb = s1["bucket"].get("(b)", 0)
        bf = s1["bucket_fail"].get("(b)", 0)         # (b)の失敗 = 真のreificationターゲット
        apk = chr(40) + chr(97) + chr(39) + chr(41)
        aprime_fail = s1["bucket_fail"].get(apk, 0)  # (a')失敗 = 参照グラフ/リンク
        a_fail = s1["bucket_fail"].get("(a)", 0)     # (a)失敗 = retrieval/prompt
        if bf > 0 and rr >= 0.15:
            g = ("reification正当化（(b)の実失敗あり＋コーパス再帰性あり）→ "
                 "**軽量版から**着手、(b)失敗例/再帰条件を構造化対象に")
        elif bf == 0:
            g = (f"**reification却下（現時点）**: (b)型は{bb}問あるが全て検索+LLMで解けており"
                 f"(b)失敗=0。reificationの実測ターゲットが無い。"
                 f"実失敗の内訳は (a')他文書定義リンク={aprime_fail}（=参照グラフ領分）、"
                 f"(a)retrieval/prompt={a_fail}。投資先はこちら")
        else:
            g = (f"(b)失敗={bf}件 あるが S2再帰率={100*rr:.0f}%と低 → "
                 "reificationの汎用価値は限定的。個別の(b)失敗をまず精査")
        parts.append(f"- S2再帰率={100*rr:.0f}% / (b)問={bb} / **(b)失敗={bf}** / "
                     f"(a')失敗={aprime_fail} / (a)失敗={a_fail}\n")
        parts.append(f"- 判定: {g}\n")
    else:
        parts.append(f"- S2再帰率={100*rr:.0f}%（S1未実行。--s1で失敗モード分類を追加）\n")

    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"S2 param再帰率: {100*rr:.0f}% ({s2['param_recur']}/{s2['total_param']})")
    if s1:
        print(f"S1 buckets: {s1['bucket']}")


if __name__ == "__main__":
    main()
