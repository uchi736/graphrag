# -*- coding: utf-8 -*-
"""llm-graph-builder の Neo4j チャンクを PGVector へミラー同期するバッチ。

GB は チャンク本文(text) と ruri-v3-310m 埋め込み(embedding) を Neo4j の :Chunk に持つ。
graphrag のドキュメント検索（ベクトル + Sudachi BM25 ハイブリッド）は PGVector を読むため、
このバッチで langchain_pg_embedding へコピーする。**埋め込みは再計算しない**
（両システムとも ruri-v3-310m/768 で一致しているためベクトルをそのまま流用）。

処理:
  1. Neo4j から text+embedding を持つ :Chunk をページング取得
  2. PGVector コレクション（既定 gb_mirror）へ id=チャンクsha1 で upsert
  3. Sudachi で tokenized_content を生成（BM25用）
  4. Neo4j に存在しなくなった行を削除（--no-delete-orphans で抑止）
  5. HNSW / GIN インデックス保証

冪等: 同一チャンクは同一idでupsertされる。再実行安全。
実行タイミング: GBの取り込み・削除・再処理の後（enrich_external_graph.py とセットで）。

使い方:
  python scripts/mirror_gb_chunks.py \
      --neo4j-uri neo4j://192.168.0.250:7688 --neo4j-user neo4j --neo4j-pw *** \
      --pg-collection gb_mirror
  # 接続情報省略時は .env (NEO4J_*, PG_CONN) を使用
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("mirror_gb_chunks")

FETCH_BATCH = 500


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", choices=["gb", "native"], default="gb",
                   help="gb: KG_CHUNK_LABEL=Chunk / KG_CHUNK_EDGE=HAS_ENTITY（既定）")
    p.add_argument("--neo4j-uri", default=None)
    p.add_argument("--neo4j-user", default=None)
    p.add_argument("--neo4j-pw", default=None)
    p.add_argument("--pg-conn", default=None, help="省略時は PG_CONN")
    p.add_argument("--pg-collection", default="gb_mirror")
    p.add_argument("--no-delete-orphans", action="store_true",
                   help="Neo4jに無くなった行の削除をスキップ")
    p.add_argument("--no-tokenize", action="store_true", help="Sudachiトークン化をスキップ")
    p.add_argument("--dry-run", action="store_true", help="件数確認のみ（書き込みなし）")
    return p.parse_args()


def main():
    args = parse_args()

    # env はimportより先に確定（プロンプト等がimport時評価のため）
    if args.profile == "gb":
        os.environ.setdefault("KG_CHUNK_LABEL", "Chunk")
        os.environ.setdefault("KG_CHUNK_EDGE", "HAS_ENTITY")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import psycopg
    from langchain_neo4j import Neo4jGraph
    from langchain_postgres import PGVector
    from graphrag_core.config import get_settings
    from graphrag_core.graph.schema import chunk_label
    from graphrag_core.llm.factory import create_embeddings
    from graphrag_core.db.utils import (
        add_connection_timeout, normalize_pg_connection_string,
        ensure_tokenized_schema, ensure_embedding_id_unique,
        ensure_schema_compatibility, ensure_hnsw_index,
    )

    s = get_settings()
    uri = args.neo4j_uri or s.neo4j_uri
    user = args.neo4j_user or s.neo4j_user
    pw = args.neo4j_pw or s.neo4j_pw
    pg_conn = add_connection_timeout(args.pg_conn or s.pg_conn, timeout=30)
    collection = args.pg_collection

    graph = Neo4jGraph(url=uri, username=user, password=pw, enhanced_schema=False)
    label = chunk_label()

    total = graph.query(
        f"MATCH (c:{label}) WHERE c.text IS NOT NULL AND c.embedding IS NOT NULL "
        "RETURN count(c) AS c")[0]["c"]
    total_no_embed = graph.query(
        f"MATCH (c:{label}) WHERE c.text IS NOT NULL AND c.embedding IS NULL "
        "RETURN count(c) AS c")[0]["c"]
    logger.info("Neo4j(:%s): 同期対象 %d 件 / 埋め込み未生成 %d 件", label, total, total_no_embed)
    if total_no_embed:
        logger.warning("埋め込み未生成のチャンクがあります。GB側で抽出（埋め込み生成）を先に完了させてください")
    if total == 0:
        logger.error("同期対象がありません")
        sys.exit(1)
    if args.dry_run:
        logger.info("--dry-run のため終了")
        return

    # スキーマ保証 → PGVectorコレクション初期化（embeddingsは検索時用に同モデルを渡す）
    ensure_schema_compatibility(pg_conn)
    embeddings = create_embeddings()
    vs = PGVector(connection=pg_conn, embeddings=embeddings,
                  collection_name=collection, use_jsonb=True)
    vs.create_collection()
    ensure_embedding_id_unique(pg_conn)
    ensure_tokenized_schema(pg_conn)

    tokenizer = None
    if not args.no_tokenize:
        try:
            from graphrag_core.text.japanese import get_japanese_processor
            tokenizer = get_japanese_processor()
        except Exception as e:
            logger.warning("Sudachi初期化失敗（tokenized_contentなしで続行）: %s", e)

    raw_conn = normalize_pg_connection_string(pg_conn)
    synced_ids: set = set()
    skip = 0
    while True:
        rows = graph.query(
            f"MATCH (c:{label}) WHERE c.text IS NOT NULL AND c.embedding IS NOT NULL "
            "RETURN c.id AS id, c.text AS text, c.embedding AS embedding, "
            "c.fileName AS source, c.page_number AS page, c.position AS position "
            "ORDER BY c.id SKIP $skip LIMIT $batch",
            params={"skip": skip, "batch": FETCH_BATCH},
        )
        if not rows:
            break
        ids = [r["id"] for r in rows]
        texts = [r["text"] for r in rows]
        metadatas = [
            {"id": r["id"], "source": r["source"] or "(unknown)",
             "page": r["page"], "position": r["position"]}
            for r in rows
        ]
        vectors = [list(r["embedding"]) for r in rows]
        # 埋め込みは再計算せずコピー（add_embeddings は id で upsert）
        vs.add_embeddings(texts=texts, embeddings=vectors, metadatas=metadatas, ids=ids)

        if tokenizer is not None:
            params = []
            for r in rows:
                try:
                    params.append((tokenizer.tokenize(r["text"]), r["id"]))
                except Exception:
                    continue
            if params:
                with psycopg.connect(raw_conn) as conn:
                    with conn.cursor() as cur:
                        cur.executemany(
                            "UPDATE langchain_pg_embedding SET tokenized_content = %s "
                            "WHERE cmetadata->>'id' = %s", params)
                    conn.commit()

        synced_ids.update(ids)
        skip += FETCH_BATCH
        logger.info("同期 %d / %d", min(skip, total), total)

    # 孤児行の削除（GBで文書削除・再処理された分）
    deleted = 0
    if not args.no_delete_orphans:
        with psycopg.connect(raw_conn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COALESCE(e.cmetadata->>'id', e.id)
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                """, (collection,))
                pg_ids = {r[0] for r in cur.fetchall() if r[0]}
                orphans = sorted(pg_ids - synced_ids)
                if orphans:
                    cur.execute("""
                        DELETE FROM langchain_pg_embedding e
                        USING langchain_pg_collection c
                        WHERE e.collection_id = c.uuid AND c.name = %s
                          AND COALESCE(e.cmetadata->>'id', e.id) = ANY(%s)
                    """, (collection, orphans))
                    deleted = cur.rowcount
            conn.commit()

    ensure_hnsw_index(pg_conn)
    logger.info("done. upsert=%d件 / 孤児削除=%d件 / collection=%s",
                len(synced_ids), deleted, collection)


if __name__ == "__main__":
    main()
