#!/usr/bin/env python
"""Stage 1 デモ: 条件付き関係(qualifier)を reify で Neo4j に格納し、
条件起点クエリ と チャンクのみ vs 条件付与 の A/B を実証する。

- 対象: モデル就業規則 第40条 割増賃金（多変種＋複合AND＋閾値）
- 格納: 言明ノード :割増(rate属性) + 共有条件ノード :Cond + [:WHEN]
        全ノードに _demo='jpref_q' を付与し、最後に DETACH DELETE で完全撤去
        （既存 Wikipedia グラフは無傷）
"""
from __future__ import annotations
import sys, io, re
from typing import List
from pydantic import BaseModel, Field

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

import fitz
from graphrag_core.config import reset_settings, get_settings
from graphrag_core.llm.factory import create_chat_llm
from langchain_neo4j import Neo4jGraph

DEMO_TAG = "jpref_q"


# ---------- 1. 抽出スキーマ ----------
class Condition(BaseModel):
    axis: str = Field(description="条件の軸: 労働区分/時間帯/曜日区分/閾値")
    operator: str = Field(description="==, between, > など")
    value: str = Field(description="条件の値(逐語): 法定時間外/深夜(22:00-05:00)/法定休日/月60時間超")

class RateFact(BaseModel):
    label: str = Field(description="この変種の短い名前(例: 時間外, 深夜, 時間外かつ深夜)")
    rate_percent: int = Field(description="割増率(%). 2割5分=25,3割5分=35,5割=50,6割=60")
    conditions: List[Condition] = Field(description="成立条件。複合(AND)は複数要素")
    source_text: str = Field(description="根拠の逐語引用")

class Extraction(BaseModel):
    facts: List[RateFact]


def extract_facts(chunk: str, llm):
    prompt = (
        "次の就業規則の条文から『割増賃金率』の条件付き事実を全て抽出。\n"
        "各率について適用条件を構造化。複合AND(例:時間外かつ深夜)は conditions を複数要素に。\n"
        "条文に明記された率と条件のみ。\n\n条文:\n" + chunk
    )
    return llm.with_structured_output(Extraction).invoke(prompt).facts


def cond_id(c: Condition) -> str:
    return f"{c.axis}|{c.operator}|{c.value}"


def main():
    reset_settings()
    s = get_settings()
    llm = create_chat_llm(temperature=0)
    g = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw)

    # ---------- 条文取得 ----------
    full = "\n".join(p.get_text() for p in fitz.open(str(_proj / "docs/モデル就業規則.pdf")))
    i = full.find("【第４０条  割増賃金】")
    chunk = re.sub(r"\n{2,}", "\n", full[i:i+700])

    print("=" * 64)
    print("STEP 1: 構造化抽出")
    facts = extract_facts(chunk, llm)
    for f in facts:
        conds = " ∧ ".join(f"{c.axis}{c.operator}{c.value}" for c in f.conditions)
        print(f"  [{f.rate_percent}%] {f.label}  ⟵ {conds}")

    # ---------- Neo4j 格納 (reify, 隔離) ----------
    before = g.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
    print(f"\n{'='*64}\nSTEP 2: reify で Neo4j 格納 (隔離ラベル, 既存ノード数={before})")
    for k, f in enumerate(facts):
        g.query(
            "CREATE (st:割増 {id:$id, label:$label, rate:$rate, source:$src, _demo:$tag})",
            params={"id": f"rate_{k}", "label": f.label, "rate": f.rate_percent,
                    "src": f.source_text, "tag": DEMO_TAG},
        )
        for c in f.conditions:
            g.query(
                "MERGE (cd:Cond {id:$cid}) ON CREATE SET cd.axis=$ax, cd.op=$op, cd.value=$val, cd._demo=$tag "
                "WITH cd MATCH (st:割増 {id:$sid}) MERGE (st)-[:WHEN]->(cd)",
                params={"cid": cond_id(c), "ax": c.axis, "op": c.operator, "val": c.value,
                        "sid": f"rate_{k}", "tag": DEMO_TAG},
            )
    nfact = g.query("MATCH (n:割増 {_demo:$t}) RETURN count(n) AS c", {"t": DEMO_TAG})[0]["c"]
    ncond = g.query("MATCH (n:Cond {_demo:$t}) RETURN count(n) AS c", {"t": DEMO_TAG})[0]["c"]
    print(f"  言明ノード(:割増)={nfact} / 共有条件ノード(:Cond)={ncond}")
    print("  共有確認: 各条件ノードが何本の言明から参照されているか")
    for r in g.query("MATCH (st:割増 {_demo:$t})-[:WHEN]->(cd:Cond) "
                     "RETURN cd.value AS v, count(st) AS n ORDER BY n DESC", {"t": DEMO_TAG}):
        print(f"    {r['v']:32s} ← {r['n']}本の言明から共有")

    # ---------- (a) 条件起点クエリ (素のRAGでは不可能) ----------
    print(f"\n{'='*64}\nSTEP 3(a): 条件起点クエリ（チャンク検索では原理的に不可能）")
    q1 = g.query("MATCH (st:割増 {_demo:$t})-[:WHEN]->(:Cond {value:'深夜(22:00-05:00)'}) "
                 "RETURN st.label AS l, st.rate AS r", {"t": DEMO_TAG})
    # 深夜の値表記がLLM出力で揺れる可能性に対応
    if not q1:
        q1 = g.query("MATCH (st:割増 {_demo:$t})-[:WHEN]->(cd:Cond) WHERE cd.value CONTAINS '22:00' "
                     "RETURN st.label AS l, st.rate AS r", {"t": DEMO_TAG})
    print("  Q『深夜が絡む割増を全部』→", [f"{r['l']}:{r['r']}%" for r in q1])
    q2 = g.query("MATCH (st:割増 {_demo:$t}) WHERE st.rate>=35 "
                 "RETURN st.label AS l, st.rate AS r ORDER BY r", {"t": DEMO_TAG})
    print("  Q『35%以上になる場合を全部』→", [f"{r['l']}:{r['r']}%" for r in q2])
    q3 = g.query("MATCH (st:割増 {_demo:$t})-[:WHEN]->(cd:Cond) WHERE st.rate=50 "
                 "RETURN st.label AS l, collect(cd.value) AS conds", {"t": DEMO_TAG})
    print("  Q『50%になる条件を全部』→", [(r['l'], r['conds']) for r in q3])

    # ---------- (b) A/B: チャンクのみ vs 条件付与 ----------
    print(f"\n{'='*64}\nSTEP 3(b): A/B（チャンクのみ vs グラフ条件付与）")
    # グラフから構造化条件テキストを生成
    rows = g.query("MATCH (st:割増 {_demo:$t}) OPTIONAL MATCH (st)-[:WHEN]->(cd:Cond) "
                   "RETURN st.label AS l, st.rate AS r, collect(cd.value) AS conds ORDER BY r", {"t": DEMO_TAG})
    graph_ctx = "\n".join(f"- {r['l']}: {r['r']}% （条件: {' かつ '.join(r['conds'])}）" for r in rows)

    questions = [
        "割増賃金率が35%以上になるのはどんな場合ですか？該当する場合を全て挙げてください。",
        "深夜労働が条件に含まれる割増賃金率を全て挙げてください。",
    ]
    for q in questions:
        print(f"\n  Q: {q}")
        a_chunk = llm.invoke(f"次の条文だけを根拠に簡潔に答えよ。\n条文:\n{chunk}\n\n質問:{q}").content
        a_graph = llm.invoke(f"次の構造化された割増賃金率テーブルを根拠に簡潔に答えよ。\n{graph_ctx}\n\n質問:{q}").content
        print(f"    [A:チャンクのみ] {a_chunk.strip()[:160]}")
        print(f"    [B:グラフ条件 ] {a_graph.strip()[:160]}")

    # ---------- 後始末 ----------
    print(f"\n{'='*64}\nSTEP 4: 後始末（_demo タグのノードのみ削除）")
    g.query("MATCH (n) WHERE n._demo=$t DETACH DELETE n", {"t": DEMO_TAG})
    after = g.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
    print(f"  削除後ノード数={after}（格納前={before}）→ {'✅ 完全復元' if after==before else '⚠️ 不一致'}")


if __name__ == "__main__":
    main()
