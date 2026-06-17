#!/usr/bin/env python
"""build_kg_jemhop.py が退避した Fujitsu グラフを Neo4j に復元する。

_bench/jemhop/neo4j_backup_pre_jemhop.jsonl.gz から node/rel を再構築。
復元キー: COALESCE(id, hash)。現在のグラフは全削除してから流し込む。

Usage:
    python _bench/jemhop/restore_neo4j.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from dotenv import load_dotenv
load_dotenv()

BACKUP = Path(__file__).resolve().parent / "neo4j_backup_pre_jemhop.jsonl.gz"


def main():
    from collections import defaultdict
    from langchain_neo4j import Neo4jGraph
    from graphrag_core.config import get_settings
    s = get_settings()

    def connect():
        return Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw,
                          enhanced_schema=False)

    holder = {"g": connect()}

    def run(cypher, params, attempts=4):
        """接続断(IncompleteCommit/ServiceUnavailable)で再接続リトライ"""
        import time
        for a in range(attempts):
            try:
                return holder["g"].query(cypher, params)
            except Exception as e:
                if a == attempts - 1:
                    raise
                print(f"    retry({a+1}) {type(e).__name__}: {str(e)[:60]}", flush=True)
                time.sleep(1.5 * (a + 1))
                try:
                    holder["g"] = connect()
                except Exception:
                    pass

    if not BACKUP.exists():
        print(f"backup not found: {BACKUP}"); sys.exit(1)

    nodes, rels = [], []
    with gzip.open(BACKUP, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            (nodes if r["kind"] == "node" else rels).append(r)
    print(f"backup: nodes={len(nodes)} rels={len(rels)}")

    cur_nodes = run("MATCH (n) RETURN count(n) AS c", {})[0]["c"]
    # ノードが既に揃っていれば再利用（部分復元のレジューム）。エッジは全削除して張り直す
    if cur_nodes == len(nodes):
        print(f"nodes already present ({cur_nodes}) -> reuse, clear edges only")
        while True:
            r = run("MATCH ()-[r]->() WITH r LIMIT 20000 DELETE r RETURN count(r) AS c", {})
            if not r or r[0]["c"] == 0:
                break
        print("  edges cleared")
    else:
        print("full wipe + recreate nodes...")
        while True:
            r = run("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS c", {})
            if not r or r[0]["c"] == 0:
                break
        by_labels = defaultdict(list)
        for n in nodes:
            by_labels[tuple(n["labels"])].append(n["props"])
        nn = 0
        for labels, plist in by_labels.items():
            lab = "".join(f":`{l}`" for l in labels) if labels else ""
            for i in range(0, len(plist), 1000):
                run(f"UNWIND $rows AS p CREATE (n{lab}) SET n = p", {"rows": plist[i:i+1000]})
                nn += len(plist[i:i+1000])
        print(f"created {nn} nodes")

    # rels: 小バッチ(200)+型別、リトライ付き
    rr = 0
    BATCH = 200
    for i in range(0, len(rels), BATCH):
        batch = rels[i:i+BATCH]
        by_type = defaultdict(list)
        for r in batch:
            if r["start_id"] and r["end_id"]:
                by_type[r["type"]].append({"sid": r["start_id"], "eid": r["end_id"], "props": r["props"]})
        for t, xs in by_type.items():
            safe = "".join(c for c in t if c.isalnum() or c == "_")
            run(
                f"UNWIND $rows AS row "
                f"MATCH (a) WHERE COALESCE(a.id,a.hash)=row.sid "
                f"MATCH (b) WHERE COALESCE(b.id,b.hash)=row.eid "
                f"CREATE (a)-[r:`{safe}`]->(b) SET r = row.props",
                {"rows": xs})
            rr += len(xs)
        if (i // BATCH) % 25 == 0:
            print(f"  rels {rr}/{len(rels)}", flush=True)
    print(f"created {rr} rels")
    print("RESTORE COMPLETE")


if __name__ == "__main__":
    main()
