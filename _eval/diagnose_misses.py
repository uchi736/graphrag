#!/usr/bin/env python
"""miss原因の内訳: ラベル違い / 方向違い / 接続無し / エンティティ無し を分類"""
from __future__ import annotations
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from graphrag_core.config import get_settings


def normalize(s: str) -> str:
    if not s:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(s)).strip().split())


def fetch_kg(source: str):
    from langchain_neo4j import Neo4jGraph
    s = get_settings()
    g = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw)
    rows = g.query("""
        MATCH (d:Document)-[:MENTIONS]->(t:Term)
        WHERE d.source = $src
        WITH COLLECT(DISTINCT t) AS terms
        UNWIND terms AS s
        MATCH (s)-[r]->(o)
        WHERE o IN terms AND type(r) <> 'MENTIONS'
        RETURN s.id AS s, type(r) AS p, o.id AS o
    """, params={"src": source})
    return [(normalize(r["s"]), r["p"], normalize(r["o"])) for r in rows]


def main():
    gt_path = Path(sys.argv[1])
    source = sys.argv[2] if len(sys.argv) > 2 else "モデル就業規則_p2-5.md"

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_triples = [(normalize(t["s"]), t["p"], normalize(t["o"])) for t in gt["triples"]]
    gt_entities = {normalize(e["name"]) for e in gt["entities"]}

    kg_triples = fetch_kg(source)
    kg_entities = set()
    for s, p, o in kg_triples:
        kg_entities.add(s); kg_entities.add(o)

    # KG側のキー: (s,o) → set(predicate)、無向ペア
    kg_directed = defaultdict(set)
    kg_undirected = defaultdict(set)
    for s, p, o in kg_triples:
        kg_directed[(s, o)].add(p)
        kg_undirected[frozenset((s, o))].add(p)

    kg_set = set(kg_triples)

    categories = {
        "exact_match": [],          # (s, p, o) 完全一致
        "wrong_direction_same_p": [],  # 方向だけ違う
        "wrong_predicate": [],      # s,o一致 (方向問わず) だがpredicate違う
        "missing_subject": [],      # subject が KG に存在しない
        "missing_object": [],       # object が KG に存在しない
        "missing_both": [],         # 両方ない
        "no_connection": [],        # 両方あるがエッジ無し
    }

    for s, p, o in gt_triples:
        if (s, p, o) in kg_set:
            categories["exact_match"].append((s, p, o))
            continue
        s_in = s in kg_entities
        o_in = o in kg_entities
        if not s_in and not o_in:
            categories["missing_both"].append((s, p, o))
        elif not s_in:
            categories["missing_subject"].append((s, p, o))
        elif not o_in:
            categories["missing_object"].append((s, p, o))
        else:
            # 両方存在
            if (o, s) in kg_directed and p in kg_directed[(o, s)]:
                categories["wrong_direction_same_p"].append((s, p, o))
            elif kg_undirected.get(frozenset((s, o))):
                got = kg_undirected[frozenset((s, o))]
                categories["wrong_predicate"].append((s, p, o, sorted(got)))
            else:
                categories["no_connection"].append((s, p, o))

    total = len(gt_triples)
    print(f"=== Diagnose miss causes (GT triples: {total}) ===\n")
    sizes = {k: len(v) for k, v in categories.items()}
    for k, n in sizes.items():
        print(f"  {k:30s} {n:3d} ({n*100/total:5.1f}%)")

    print(f"\n  KG distinct entities in scope: {len(kg_entities)}")
    print(f"  GT entities: {len(gt_entities)}")
    print(f"  GT entity coverage in KG: {len(gt_entities & kg_entities)}/{len(gt_entities)} = {len(gt_entities & kg_entities)*100/len(gt_entities):.1f}%")

    print("\n=== exact_match ===")
    for t in categories["exact_match"]:
        print(f"  ✓ ({t[0]}) -[{t[1]}]-> ({t[2]})")

    print("\n=== wrong_direction_same_p ===")
    for t in categories["wrong_direction_same_p"]:
        print(f"  ↔ ({t[0]}) -[{t[1]}]-> ({t[2]})  // KGには逆向きあり")

    print("\n=== wrong_predicate (両ノード存在&エッジあり、predicateだけ違う) ===")
    for t in categories["wrong_predicate"]:
        s, p, o, got = t
        print(f"  ≠ ({s}) -[{p}?]-> ({o})  // KGでは: {got}")

    print("\n=== no_connection (両ノードあるが繋がってない) ===")
    for t in categories["no_connection"]:
        print(f"  ─ ({t[0]}) ?? ({t[2]})  [GT: {t[1]}]")

    print("\n=== missing_subject (subject が KG に存在しない) ===")
    for t in categories["missing_subject"]:
        print(f"  - {t[0]}  (-[{t[1]}]-> {t[2]})")

    print("\n=== missing_object (object が KG に存在しない) ===")
    for t in categories["missing_object"]:
        print(f"  - {t[2]}  ({t[0]} -[{t[1]}]-)")

    if categories["missing_both"]:
        print("\n=== missing_both ===")
        for t in categories["missing_both"]:
            print(f"  - ({t[0]}, {t[2]})  [GT: {t[1]}]")


if __name__ == "__main__":
    main()
