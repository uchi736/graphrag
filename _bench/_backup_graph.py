"""Neo4j フルバックアップ（全ノード・全エッジ・全プロパティ → gzip JSONL）

restore用: 各行 {"kind":"node","labels":[...],"props":{...}} /
          {"kind":"rel","type":T,"start_id":...,"start_labels":[...],"end_id":...,"end_labels":[...],"props":{...}}
ProcessedChunk は hash、それ以外は id で復元キーにする。
"""
import gzip
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()

from langchain_neo4j import Neo4jGraph

OUT = Path(__file__).parent / "backup_full_graph_20260611.jsonl.gz"

g = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USER"),
               password=os.getenv("NEO4J_PW"), enhanced_schema=False)

BATCH = 5000
n_nodes = n_rels = 0
with gzip.open(OUT, "wt", encoding="utf-8") as f:
    # ノード
    skip = 0
    while True:
        rows = g.query(
            "MATCH (n) WITH n ORDER BY elementId(n) SKIP $skip LIMIT $batch "
            "RETURN labels(n) AS labels, properties(n) AS props",
            {"skip": skip, "batch": BATCH},
        )
        if not rows:
            break
        for r in rows:
            f.write(json.dumps({"kind": "node", "labels": r["labels"], "props": r["props"]},
                               ensure_ascii=False, default=str) + "\n")
        n_nodes += len(rows)
        print(f"  nodes {n_nodes}", flush=True)
        if len(rows) < BATCH:
            break
        skip += BATCH

    # エッジ（復元キー: id or hash）
    skip = 0
    while True:
        rows = g.query(
            "MATCH (a)-[r]->(b) WITH a, r, b ORDER BY elementId(r) SKIP $skip LIMIT $batch "
            "RETURN type(r) AS t, properties(r) AS props, "
            "COALESCE(a.id, a.hash) AS sid, labels(a) AS sl, "
            "COALESCE(b.id, b.hash) AS eid, labels(b) AS el",
            {"skip": skip, "batch": BATCH},
        )
        if not rows:
            break
        for r in rows:
            f.write(json.dumps({"kind": "rel", "type": r["t"], "props": r["props"],
                                "start_id": r["sid"], "start_labels": r["sl"],
                                "end_id": r["eid"], "end_labels": r["el"]},
                               ensure_ascii=False, default=str) + "\n")
        n_rels += len(rows)
        print(f"  rels {n_rels}", flush=True)
        if len(rows) < BATCH:
            break
        skip += BATCH

print(f"backup done: nodes={n_nodes} rels={n_rels} -> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
