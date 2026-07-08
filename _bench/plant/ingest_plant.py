#!/usr/bin/env python
"""makedataset の事前チャンク済み plant コーパスを PGVector に投入する。

- 入力: C:/work/makedataset/data/chunks_plant/*.jsonl（チャンク済み: doc_id/chunk_id/text）
- 出力: PGVector コレクション（デフォルト plant_v15）。doc_id を source メタデータに保持。
- BM25 用に tokenized_content も付与。
- 既存 Fujitsu 等のコレクションには触れない（新規 collection_id で分離）。

Usage:
    python _bench/plant/ingest_plant.py
    python _bench/plant/ingest_plant.py --collection plant_v15 --fresh
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from dotenv import load_dotenv
load_dotenv()

CHUNKS_DIR = Path("C:/work/makedataset/data/chunks_plant")


def load_plant_chunks():
    from langchain_core.documents import Document
    docs = []
    for fp in sorted(glob.glob(str(CHUNKS_DIR / "*.jsonl"))):
        doc_id = os.path.basename(fp).replace(".jsonl", "")
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = (r.get("text") or "").strip()
            if not text:
                continue
            sec = " > ".join(r.get("section_path") or [])
            docs.append(Document(
                page_content=(f"[{sec}]\n{text}" if sec else text),
                metadata={
                    "id": r.get("chunk_id"),
                    "source": r.get("doc_id", doc_id),
                    "chunk_id": r.get("chunk_id"),
                    "page": r.get("page"),
                },
            ))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="plant_v15")
    ap.add_argument("--fresh", action="store_true", help="コレクションを作り直す")
    ap.add_argument("--chunks-dir", default=None, help="チャンクJSONLディレクトリ（既定: chunks_plant）")
    args = ap.parse_args()
    if args.chunks_dir:
        global CHUNKS_DIR
        CHUNKS_DIR = Path(args.chunks_dir)

    os.environ["PG_COLLECTION"] = args.collection
    from graphrag_core.config import reset_settings, get_settings
    reset_settings()
    s = get_settings()

    from graphrag_core.llm.factory import create_embeddings
    from graphrag_core.db.utils import (
        ensure_embedding_id_unique, ensure_schema_compatibility, ensure_hnsw_index,
        ensure_tokenized_schema, batch_pgvector_from_documents, batch_update_tokenized,
        add_connection_timeout,
    )
    from graphrag_core.text.japanese import get_japanese_processor

    docs = load_plant_chunks()
    print(f"loaded {len(docs)} chunks from {CHUNKS_DIR}")
    n_docs = len({d.metadata['source'] for d in docs})
    print(f"distinct doc_id: {n_docs}")

    embeddings = create_embeddings()
    dim = len(embeddings.embed_query("テスト"))
    print(f"embedding provider={s.embedding_provider} dim={dim} collection={args.collection}")

    pg = add_connection_timeout(s.pg_conn, timeout=30)
    ensure_embedding_id_unique(s.pg_conn)
    ensure_schema_compatibility(s.pg_conn)
    ensure_hnsw_index(s.pg_conn)

    batch_pgvector_from_documents(
        docs, embeddings, connection=pg, collection_name=args.collection,
        pre_delete_collection=args.fresh,
        progress_callback=lambda i, t, n: print(f"  PGVector: {i+n}/{t}") if (i // 500) % 4 == 0 else None,
    )
    print(f"PGVector ingest done: {len(docs)} chunks (fresh={args.fresh})")

    proc = get_japanese_processor()
    if proc and s.enable_japanese_search:
        ensure_tokenized_schema(s.pg_conn)
        for d in docs:
            try:
                d.metadata["tokenized_content"] = proc.tokenize(d.page_content)
            except Exception:
                d.metadata["tokenized_content"] = None
        updated = batch_update_tokenized(s.pg_conn, docs)
        print(f"tokenized: {updated} chunks")

    print("INGEST COMPLETE")


if __name__ == "__main__":
    main()
