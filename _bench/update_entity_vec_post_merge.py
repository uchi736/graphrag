#!/usr/bin/env python
"""EL merge 後、pgvector の dup entity_id を canonical に書き換える

embedding はそのまま (検索hit率維持)、metadata.entity_id だけ canonical に置換。
これで dup名で検索hit → canonical id を返却 → Neo4j で正しく見つかる。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="fjrag_hard_v3_entities",
                    help="pgvector collection name (default: fjrag_hard_v3_entities for FJH-06)")
    args = ap.parse_args()

    import psycopg
    from graphrag_core.config import reset_settings, get_settings
    from graphrag_core.db.utils import normalize_pg_connection_string
    reset_settings()
    s = get_settings()
    coll = args.collection
    print(f"target collection: {coll}")
    conn_str = normalize_pg_connection_string(s.pg_conn)

    applied = json.load(open("_bench/el_safe_subset_applied.json", encoding="utf-8"))
    # (dup_id, canonical_id) のペアを構築
    pairs = []
    for entry in applied["applied"]:
        canon = entry["canonical"]
        for dup in entry["duplicates"]:
            if dup and dup != canon:
                pairs.append((dup, canon))
    print(f"update候補: {len(pairs)} pairs")

    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            # collection UUID
            cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (coll,))
            row = cur.fetchone()
            if not row:
                print(f"collection {coll} not found"); return
            coll_uuid = row[0]

            updated = 0
            not_found = 0
            for dup_id, canon_id in pairs:
                # entity_id が dup と一致する row を canonical に書き換え
                cur.execute("""
                    UPDATE langchain_pg_embedding
                    SET cmetadata = jsonb_set(cmetadata, '{entity_id}', to_jsonb(%s::text))
                    WHERE collection_id = %s
                      AND cmetadata->>'entity_id' = %s
                    RETURNING id
                """, (canon_id, coll_uuid, dup_id))
                rows = cur.fetchall()
                if rows:
                    updated += len(rows)
                else:
                    not_found += 1
            conn.commit()
            print(f"updated: {updated} embeddings (metadata.entity_id → canonical)")
            print(f"not_found dup_ids in pgvector: {not_found}")


if __name__ == "__main__":
    main()
