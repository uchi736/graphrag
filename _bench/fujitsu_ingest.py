#!/usr/bin/env python
"""Fujitsu-RAG-Hard-Benchmark のPDF群を page-aware で PGVector に投入

各ページを1チャンクとして扱い、metadata に source(PDFファイル名) と page(1-indexed) を保持。
別コレクション (デフォルト: fjrag_hard) を使い既存データと混じらないようにする。

Usage:
    python _bench/fujitsu_ingest.py
    python _bench/fujitsu_ingest.py --collection fjrag_hard --batch-size 100
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from graphrag_core.config import reset_settings, get_settings


def load_page_chunks(pdf_dir: Path, max_chars: int = 1000, overlap: int = 80) -> list[dict]:
    """全PDFを page-aware で読み込み chunks 化。

    1ページ = 基本1チャンク。ただし `max_chars` を超えたページは分割する
    (Ruri 1024トークン制約を char ベースで近似)。
    page metadata は分割しても同一page番号を保持。
    """
    import fitz
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars, chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )

    chunks = []
    long_page_count = 0
    for fn in sorted(pdf_dir.iterdir()):
        if not fn.suffix.lower() == ".pdf":
            continue
        try:
            doc = fitz.open(str(fn))
        except Exception as e:
            print(f"  ⚠️  {fn.name}: open失敗 {e}")
            continue
        for i in range(len(doc)):
            text = doc[i].get_text("text", sort=True).strip()
            if not text:
                continue
            page_num = i + 1
            if len(text) <= max_chars:
                cid = hashlib.sha256(f"{fn.name}|{page_num}|0|{text[:200]}".encode("utf-8")).hexdigest()
                chunks.append({"id": cid, "text": text, "source": fn.name, "page": page_num, "sub": 0})
            else:
                long_page_count += 1
                parts = splitter.split_text(text)
                for j, part in enumerate(parts):
                    cid = hashlib.sha256(f"{fn.name}|{page_num}|{j}|{part[:200]}".encode("utf-8")).hexdigest()
                    chunks.append({"id": cid, "text": part, "source": fn.name, "page": page_num, "sub": j})
        doc.close()
    if long_page_count:
        print(f"  ℹ️  長いページを分割: {long_page_count} ページ")
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="../Fujitsu-RAG-Hard-Benchmark/dataset/PDFs",
                    help="PDFフォルダ (graphragルート基準)")
    ap.add_argument("--collection", default="fjrag_hard",
                    help="PGVectorコレクション名 (既存と分離)")
    ap.add_argument("--batch-size", type=int, default=100)
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = (_proj / args.pdf_dir).resolve()
    if not pdf_dir.exists():
        print(f"PDFフォルダが見つかりません: {pdf_dir}")
        sys.exit(1)

    print(f"PDF dir: {pdf_dir}")
    print(f"Collection: {args.collection}")

    t0 = time.time()
    chunks = load_page_chunks(pdf_dir)
    print(f"全 {len(chunks)} ページチャンクを抽出 ({time.time()-t0:.1f}s)")

    # PGVector投入
    from langchain_core.documents import Document
    from graphrag_core.llm.factory import create_embeddings
    from graphrag_core.db.utils import (
        ensure_embedding_id_unique, ensure_schema_compatibility, ensure_hnsw_index,
        add_connection_timeout, batch_pgvector_from_documents,
        ensure_tokenized_schema, batch_update_tokenized,
    )
    from graphrag_core.text.japanese import get_japanese_processor

    reset_settings()
    s = get_settings()
    embeddings = create_embeddings()

    docs = [
        Document(
            page_content=c["text"],
            metadata={
                "id": c["id"], "source": c["source"],
                "page": c["page"], "sub": c.get("sub", 0),
            },
        )
        for c in chunks
    ]

    ensure_embedding_id_unique(s.pg_conn)
    ensure_schema_compatibility(s.pg_conn)
    ensure_hnsw_index(s.pg_conn)
    pg = add_connection_timeout(s.pg_conn, timeout=30)

    print(f"Embedding + PGVector 書込中 (batch={args.batch_size})...")
    t1 = time.time()
    batch_pgvector_from_documents(
        docs, embeddings,
        connection=pg, collection_name=args.collection,
        pre_delete_collection=True,
        batch_size=args.batch_size,
        progress_callback=lambda i, total, n: print(f"  {i+n}/{total}") if (i//100 != (i+n)//100 or i+n == total) else None,
    )
    print(f"PGVector OK ({time.time()-t1:.1f}s)")

    # 日本語トークン化 (ハイブリッド検索用)
    jp = get_japanese_processor()
    if jp and s.enable_japanese_search:
        print(f"日本語トークン化中...")
        t2 = time.time()
        ensure_tokenized_schema(s.pg_conn)
        for d in docs:
            try:
                d.metadata["tokenized_content"] = jp.tokenize(d.page_content)
            except Exception:
                d.metadata["tokenized_content"] = None
        n_upd = batch_update_tokenized(s.pg_conn, docs)
        print(f"  {n_upd}件 ({time.time()-t2:.1f}s)")

    print(f"\n完了: {len(docs)} chunks → collection={args.collection}  total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
