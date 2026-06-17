#!/usr/bin/env python
"""復元後のFujitsuグラフの重複エッジを除去する。

restore_neo4j.py は重複idノード(8557個)を COALESCE(id,hash) で照合するため
デカルト積でエッジが膨張(137k→235k)。同一 (start.id, type, end.id) のエッジを
1本へ集約する。関係タイプ別に処理し、接続断はリトライ。

Usage:
    python _bench/jemhop/dedup_edges.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))
from dotenv import load_dotenv
load_dotenv()


def main():
    from langchain_neo4j import Neo4jGraph
    from graphrag_core.config import get_settings
    s = get_settings()

    def connect():
        return Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw,
                          enhanced_schema=False)
    h = {"g": connect()}

    def run(cy, params=None, attempts=5):
        for a in range(attempts):
            try:
                return h["g"].query(cy, params or {})
            except Exception as e:
                if a == attempts - 1:
                    raise
                time.sleep(1.5 * (a + 1))
                try:
                    h["g"] = connect()
                except Exception:
                    pass

    before = run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    types = [r["t"] for r in run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t")]
    print(f"edges before={before}, types={len(types)}", flush=True)

    removed = 0
    for t in types:
        safe = "".join(c for c in t if c.isalnum() or c == "_")
        # 同一 (ak, bk) で2本以上ある type=t のエッジの、2本目以降を削除（バッチ）
        while True:
            r = run(
                f"MATCH (a)-[r:`{safe}`]->(b) "
                "WITH COALESCE(a.id,a.hash) AS ak, COALESCE(b.id,b.hash) AS bk, "
                "collect(elementId(r)) AS ids WHERE size(ids) > 1 "
                "WITH ids[1..] AS extra LIMIT 3000 "
                "UNWIND extra AS eid WITH collect(eid) AS eids "
                "MATCH ()-[rr]->() WHERE elementId(rr) IN eids DELETE rr "
                "RETURN count(*) AS c")
            c = r[0]["c"] if r else 0
            removed += c
            if c == 0:
                break
        print(f"  {t}: cumulative removed={removed}", flush=True)

    after = run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    print(f"edges after={after} (removed {before-after}), target~137099")
    print("DEDUP COMPLETE")


if __name__ == "__main__":
    main()
