"""条件付き関係(qualifier)の抽出 + reify 格納

規程・基準のような文書では「割増賃金=35%」が無条件には成立せず「深夜労働の場合」
という条件下でのみ成立する。これを無条件三つ組ではなく、言明ノード(:CondFact)＋
共有条件ノード(:Cond)＋[:WHEN] で reify して格納する。

- 数値(率/閾値/日数)は :CondFact のプロパティ value_num。決してノード化しない。
- 複合AND は 1つの :CondFact から複数の [:WHEN]。
- 条件は canonical id (axis|op|norm_value) で共有・名寄せされる。
- 全ノードに pg_collection を刻印（コーパス横断の漏れ防止 / provenance と整合）。

build 時の後処理（scripts/build_kg.py）から `build_conditional_graph` で呼ぶ。
references.py の load→extract→batched-write テンプレと同じ構造。

検証済みプロトタイプ: _bench/_qualifier_stage1_demo.py（第40条で6変種・複合・閾値を実証）。
"""
from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from pydantic import BaseModel, Field

from graphrag_core.text.japanese import normalize_entity_text
from graphrag_core.llm.factory import structured_output

logger = logging.getLogger(__name__)

# 条件が書かれていそうなチャンクだけ LLM に回すための安価な事前フィルタ
_COND_KEYWORDS = re.compile(
    r"場合|とき|時に|以上|以下|未満|を超え|超える|を除く|でない|"
    r"平日|深夜|休日|年度|勤続|継続勤務|区分|当たり|につき|割増|手当|休暇|懲戒|限り"
)

# 抽出スキーマで使う閉じた語彙（vLLM guided_json/xgrammar も enum で縛れる）
QUALIFIER_AXES = [
    "労働区分", "時間帯", "曜日区分", "閾値", "勤続年数", "対象", "期間", "状態", "その他",
]
QUALIFIER_OPS = ["eq", "gte", "lte", "gt", "lt", "between", "in"]


# ── 抽出スキーマ（Azure with_structured_output / vLLM guided_json 共用） ──

class Condition(BaseModel):
    axis: str = Field(description="条件の軸（労働区分/時間帯/曜日区分/閾値/勤続年数/対象/期間/状態）")
    operator: str = Field(description="比較演算子: eq/gte/lte/gt/lt/between/in")
    value: str = Field(description="条件の値(逐語): 法定時間外 / 深夜(22:00-05:00) / 法定休日 / 月60時間超 など")


class CondFact(BaseModel):
    label: str = Field(description="この変種の短い名前（例: 時間外, 深夜, 時間外かつ深夜）")
    predicate: str = Field(description="関係の種類を表す英大文字スネーク（例: WAGE_PREMIUM_RATE, LEAVE_DAYS）")
    value_num: int = Field(default=-1, description="結果の数値（率/閾値/日数）。2割5分=25等。無ければ-1")
    unit: str = Field(default="", description="単位（% / 日 / 円 / 時間 など）。無ければ空")
    conditions: List[Condition] = Field(default_factory=list, description="成立条件。複合(AND)は複数要素")
    source_text: str = Field(default="", description="根拠の逐語引用")


class Extraction(BaseModel):
    facts: List[CondFact] = Field(default_factory=list)


# ── 正準化 ──

def _cond_id(axis: str, op: str, value: str) -> str:
    """共有条件ノードの正準キー。NFKC+空白除去のみ（kana過剰マージは避ける）。"""
    norm = normalize_entity_text(value)
    return f"{axis}|{op}|{norm}", norm


def _fact_id(predicate: str, value: str, cond_ids: List[str]) -> str:
    """言明ノードの決定的id。同一変種は同一id → 増分再構築でも MERGE で重複しない。"""
    key = predicate + "|" + value + "|" + "&".join(sorted(cond_ids))
    return "cf_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


# ── 抽出 ──

def extract_conditional_facts(chunks, llm, *, max_workers: int = 4, min_chars: int = 120) -> List[dict]:
    """チャンク群から条件付き事実を構造化抽出する。

    Args:
        chunks: [{id,text,source,page}, ...] または [str, ...]
    Returns:
        [{predicate,label,value,value_num,unit,conditions:[{axis,op,value}],source,source_chunk,source_text}, ...]
    """
    runnable = structured_output(llm, Extraction)

    norm_chunks = []
    for c in chunks:
        if isinstance(c, str):
            norm_chunks.append({"id": None, "text": c, "source": "", "page": None})
        else:
            norm_chunks.append(c)
    # 条件語を含む十分な長さのチャンクだけ
    targets = [c for c in norm_chunks
               if c.get("text") and len(c["text"]) >= min_chars and _COND_KEYWORDS.search(c["text"])]
    logger.info("conditional extract: %d/%d chunks pass pre-filter", len(targets), len(norm_chunks))

    from graphrag_core.prompts import COND_FACT_USER_PROMPT

    def _one(c):
        out = []
        try:
            res = runnable.invoke(COND_FACT_USER_PROMPT.format(chunk=c["text"]))
            for f in (res.facts or []):
                if not f.conditions:
                    continue  # 無条件の事実は qualifier の対象外
                out.append({
                    "predicate": f.predicate.strip() or "COND_FACT",
                    "label": f.label.strip(),
                    "value": (f.source_text or f.label or "").strip()[:80] if f.value_num == -1 else str(f.value_num),
                    "value_num": f.value_num,
                    "unit": f.unit.strip(),
                    "conditions": [{"axis": cd.axis.strip(), "op": cd.operator.strip(), "value": cd.value.strip()}
                                   for cd in f.conditions],
                    "source": c.get("source", ""),
                    "source_chunk": c.get("id"),
                    "source_text": (f.source_text or "").strip()[:300],
                })
        except Exception as e:
            logger.warning("conditional extract failed for chunk %s: %s", c.get("id"), e)
        return out

    facts: List[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed([ex.submit(_one, c) for c in targets]):
            facts.extend(fut.result())
    return facts


# ── 格納（reify, 冪等） ──

def write_conditional_facts(graph, facts: List[dict], pg_collection: str, batch_size: int = 500) -> dict:
    """:CondFact + 共有 :Cond + [:WHEN] を冪等(MERGE-by-id)に書き込む。"""
    rows = []
    for f in facts:
        cond_rows = []
        cond_ids = []
        for c in f["conditions"]:
            cid, norm = _cond_id(c["axis"], c["op"], c["value"])
            cond_ids.append(cid)
            cond_rows.append({"id": cid, "axis": c["axis"], "op": c["op"], "value": c["value"], "norm": norm})
        fid = _fact_id(f["predicate"], f["value"], cond_ids)
        rows.append({
            "fid": fid, "predicate": f["predicate"], "label": f["label"], "value": f["value"],
            "value_num": f["value_num"], "unit": f["unit"], "source": f.get("source", ""),
            "source_text": f.get("source_text", ""), "source_chunk": f.get("source_chunk"),
            "conds": cond_rows,
        })

    written = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        graph.query(
            """
            UNWIND $rows AS r
            MERGE (st:CondFact {id: r.fid})
              ON CREATE SET st.extraction_count = 1
              ON MATCH  SET st.extraction_count = COALESCE(st.extraction_count, 0) + 1
            SET st.predicate = r.predicate, st.label = r.label, st.value = r.value,
                st.value_num = r.value_num, st.unit = r.unit, st.source = r.source,
                st.source_text = r.source_text, st.pg_collection = $coll,
                st.source_chunks = CASE WHEN r.source_chunk IS NULL THEN COALESCE(st.source_chunks, [])
                    WHEN r.source_chunk IN COALESCE(st.source_chunks, []) THEN st.source_chunks
                    ELSE COALESCE(st.source_chunks, []) + r.source_chunk END
            WITH st, r
            UNWIND r.conds AS c
            MERGE (cd:Cond {id: c.id})
              ON CREATE SET cd.axis = c.axis, cd.op = c.op, cd.value = c.value,
                            cd.norm_value = c.norm, cd.pg_collection = $coll
            MERGE (st)-[:WHEN]->(cd)
            """,
            {"rows": batch, "coll": pg_collection},
        )
        written += len(batch)
    nfact = graph.query("MATCH (n:CondFact {pg_collection:$c}) RETURN count(n) AS n", {"c": pg_collection})[0]["n"]
    ncond = graph.query("MATCH (n:Cond {pg_collection:$c}) RETURN count(n) AS n", {"c": pg_collection})[0]["n"]
    return {"facts_written": written, "condfact_nodes": nfact, "cond_nodes": ncond}


# ── オーケストレータ（build_kg の後処理から呼ぶ） ──

def build_conditional_graph(graph, llm, pg_collection: str, *, sources=None) -> dict:
    """既存グラフの Document チャンクから条件付き事実を抽出し reify 格納する。"""
    from graphrag_core.graph.references import load_chunks
    chunks = load_chunks(graph)
    if sources:
        chunks = [c for c in chunks if c.get("source") in set(sources)]
    facts = extract_conditional_facts(chunks, llm)
    if not facts:
        logger.info("conditional facts: 0 extracted")
        return {"facts_written": 0, "condfact_nodes": 0, "cond_nodes": 0}
    stats = write_conditional_facts(graph, facts, pg_collection)
    logger.info("conditional facts: %s", stats)
    return stats
