#!/usr/bin/env python
"""KG精度評価スクリプト

Neo4j に構築された Term/関係グラフを、人手curationの正解tripleと比較し
precision / recall / F1 を出す。

Usage:
    python _eval/evaluate_kg.py _eval/ground_truth_モデル就業規則_p2-5.json
    python _eval/evaluate_kg.py _eval/ground_truth_モデル就業規則_p2-5.json --source モデル就業規則.pdf

評価方針:
- エンティティマッチ: 完全一致 + 軽い正規化 (空白除去, 全角半角統一)
- relation マッチ: predicate名の完全一致 (Neo4j側はLLMで推定されたエッジtype)
- triple一致: (s, p, o) all-or-nothing
- must_have vs should_have を分けて報告
- 余計に抽出されたtripleは noise として量だけ集計
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from graphrag_core.config import get_settings


def normalize(s: str) -> str:
    """エンティティ名の軽い正規化。

    - NFKC で全角/半角や合字を統一
    - 前後の空白除去
    - 内部の連続空白を単一空白に
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip()
    return " ".join(s.split())


def load_ground_truth(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "triples" not in data:
        raise ValueError("ground truth JSON に 'triples' キーがありません")
    return data


def fetch_kg_triples(source_filter: str | None = None) -> list[tuple[str, str, str]]:
    """Neo4j から (subject, predicate, object) を取得。

    - Document/MENTIONS は除外（チャンク→Term の参照は構造情報）
    - SchemaMeta も除外
    - source_filter があれば、Document.source が一致するチャンクから派生した Term だけに絞る
    """
    from langchain_neo4j import Neo4jGraph

    s = get_settings()
    g = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw)

    if source_filter:
        # source が一致する Document が MENTIONS している Term ノード集合 を抽出
        # → そのTerm間のエッジ (MENTIONS以外) を triple として取得
        cypher = """
        MATCH (d:Document)-[:MENTIONS]->(t:Term)
        WHERE d.source = $src
        WITH COLLECT(DISTINCT t) AS terms
        UNWIND terms AS s
        MATCH (s)-[r]->(o)
        WHERE o IN terms AND type(r) <> 'MENTIONS' AND type(r) <> 'PART_OF_DOCUMENT'
        RETURN s.id AS subject, type(r) AS predicate, o.id AS object
        """
        rows = g.query(cypher, params={"src": source_filter})
    else:
        cypher = """
        MATCH (s:Term)-[r]->(o:Term)
        WHERE type(r) <> 'MENTIONS'
        RETURN s.id AS subject, type(r) AS predicate, o.id AS object
        """
        rows = g.query(cypher)

    triples = []
    for r in rows:
        if r["subject"] and r["predicate"] and r["object"]:
            triples.append((r["subject"], r["predicate"], r["object"]))
    return triples


def fetch_kg_entities(source_filter: str | None = None) -> set[str]:
    """source_filter 配下のTerm ID集合を取得。"""
    from langchain_neo4j import Neo4jGraph

    s = get_settings()
    g = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw)

    if source_filter:
        cypher = """
        MATCH (d:Document)-[:MENTIONS]->(t:Term)
        WHERE d.source = $src
        RETURN DISTINCT t.id AS id
        """
        rows = g.query(cypher, params={"src": source_filter})
    else:
        cypher = "MATCH (t:Term) RETURN DISTINCT t.id AS id"
        rows = g.query(cypher)

    return {r["id"] for r in rows if r.get("id")}


def evaluate(gt: dict, source_filter: str | None) -> dict:
    """正解triple集と Neo4j抽出triple集を比較。"""
    gt_triples_all = [(normalize(t["s"]), t["p"], normalize(t["o"]), t.get("criticality", "must_have")) for t in gt["triples"]]
    gt_entities_all = {normalize(e["name"]) for e in gt.get("entities", [])}

    gt_triples_set = {(s, p, o) for s, p, o, _ in gt_triples_all}
    gt_must = {(s, p, o) for s, p, o, c in gt_triples_all if c == "must_have"}
    gt_should = {(s, p, o) for s, p, o, c in gt_triples_all if c == "should_have"}

    raw_kg_triples = fetch_kg_triples(source_filter)
    raw_kg_entities = fetch_kg_entities(source_filter)

    # SAME_AS は対称関係。逆向きエッジも同一視するため、双方向に展開しておく。
    SYMMETRIC = {"SAME_AS", "RELATED_TO"}
    kg_triples_set: set[tuple[str, str, str]] = set()
    for s, p, o in raw_kg_triples:
        ns, no = normalize(s), normalize(o)
        kg_triples_set.add((ns, p, no))
        if p in SYMMETRIC:
            kg_triples_set.add((no, p, ns))
    kg_entities = {normalize(e) for e in raw_kg_entities}

    # ---------- Entity coverage ----------
    ent_hit = gt_entities_all & kg_entities
    ent_miss = gt_entities_all - kg_entities

    # ---------- Triple-level metrics ----------
    triple_hit = gt_triples_set & kg_triples_set
    triple_miss = gt_triples_set - kg_triples_set
    triple_extra = kg_triples_set - gt_triples_set  # noise / not-in-GT

    must_hit = gt_must & kg_triples_set
    must_miss = gt_must - kg_triples_set
    should_hit = gt_should & kg_triples_set

    precision = len(triple_hit) / len(kg_triples_set) if kg_triples_set else 0.0
    recall = len(triple_hit) / len(gt_triples_set) if gt_triples_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    must_recall = len(must_hit) / len(gt_must) if gt_must else 0.0
    should_recall = len(should_hit) / len(gt_should) if gt_should else 0.0

    return {
        "gt_entity_count": len(gt_entities_all),
        "kg_entity_count": len(kg_entities),
        "entity_recall": len(ent_hit) / len(gt_entities_all) if gt_entities_all else 0.0,
        "entity_missing": sorted(ent_miss),

        "gt_triple_count": len(gt_triples_set),
        "kg_triple_count": len(kg_triples_set),
        "triple_hit": len(triple_hit),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "must_recall": must_recall,
        "should_recall": should_recall,

        "must_hit_count": len(must_hit),
        "must_total": len(gt_must),
        "must_missing_triples": sorted(must_miss),
        "extra_triple_count": len(triple_extra),
        "extra_triples_sample": sorted(triple_extra)[:20],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ground_truth_json", help="正解triple JSONファイルパス")
    ap.add_argument("--source", help="Neo4jから取得時のDocument.source絞り込み (例: モデル就業規則.pdf)")
    ap.add_argument("--show-extras", type=int, default=20, help="余計に抽出されたtripleを何件表示するか")
    args = ap.parse_args()

    gt_path = Path(args.ground_truth_json)
    if not gt_path.exists():
        print(f"ERROR: 正解ファイルが見つかりません: {gt_path}")
        sys.exit(1)

    gt = load_ground_truth(gt_path)
    print(f"=== Ground truth ===")
    print(f"  source: {gt['_meta'].get('source')}")
    print(f"  scope:  {gt['_meta'].get('scope')}")
    print(f"  entities: {len(gt.get('entities', []))}")
    print(f"  triples:  {len(gt['triples'])}")
    print()

    print(f"=== Neo4j KG ===")
    print(f"  source filter: {args.source or '(none, 全Term対象)'}")
    print()

    result = evaluate(gt, args.source)

    print(f"=== Entity Coverage ===")
    print(f"  GT entities: {result['gt_entity_count']}")
    print(f"  KG entities (filtered): {result['kg_entity_count']}")
    print(f"  Entity recall: {result['entity_recall']:.2%}")
    if result["entity_missing"]:
        print(f"  Missing entities (sample 10): {result['entity_missing'][:10]}")
    print()

    print(f"=== Triple Metrics ===")
    print(f"  GT triples: {result['gt_triple_count']}")
    print(f"  KG triples: {result['kg_triple_count']}")
    print(f"  Hit: {result['triple_hit']}")
    print(f"  Precision: {result['precision']:.2%}")
    print(f"  Recall:    {result['recall']:.2%}")
    print(f"  F1:        {result['f1']:.2%}")
    print()
    print(f"  must_have recall:   {result['must_recall']:.2%}  ({result['must_hit_count']}/{result['must_total']})")
    print(f"  should_have recall: {result['should_recall']:.2%}")
    print()

    if result["must_missing_triples"]:
        print(f"=== Missed must_have triples ({len(result['must_missing_triples'])}) ===")
        for s, p, o in result["must_missing_triples"]:
            print(f"  - ({s}) -[{p}]-> ({o})")
        print()

    print(f"=== Extra triples (not in GT): {result['extra_triple_count']} ===")
    if args.show_extras and result["extra_triples_sample"]:
        for s, p, o in result["extra_triples_sample"][: args.show_extras]:
            print(f"  + ({s}) -[{p}]-> ({o})")


if __name__ == "__main__":
    main()
