#!/usr/bin/env python
"""Fujitsu preprocessing済みmarkdown → page-aware chunks → PGVector

`_bench/_pp/{pdf_stem}/extracted_text.txt` を読み、`[ページ N]` マーカーで分割。
ページ内テキストには markdown構造 (## 見出し、表) が含まれるので、
MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter で構造尊重チャンキング。

各チャンクは metadata に source/page/sub/section を保持する。

Usage:
    python _bench/fujitsu_ingest_md.py
    python _bench/fujitsu_ingest_md.py --collection fjrag_hard_md --max-chars 1000
"""
from __future__ import annotations
import argparse, hashlib, re, sys, time
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))


# `[ページ N]` 行で分割するためのパターン
PAGE_PAT = re.compile(r'\n\[ページ (\d+)\]\n-+\n', re.MULTILINE)


def parse_md_to_page_chunks(md_text: str, pdf_name: str, pdf_path: Path = None,
                            max_chars: int = 1000, overlap: int = 100) -> list[dict]:
    """preprocessing済み markdown を [ページ N] で分割し、各ページを
    markdown-aware にチャンク化する。

    preprocessing_optimizer が「画像」と判定したページは markdown 出力が
    [画像: ...] マーカーだけになるが、実際は PyMuPDF で text 抽出可能なケースが多い。
    `pdf_path` 指定時、各ページの markdown コンテンツが薄い (本文 <100字) なら
    PyMuPDF で text を補完する。
    """
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter,
    )
    import re as _re

    # まず先頭ヘッダ (# pdf_name, 総ページ数...) を削る
    body_start = md_text.find('\n[ページ ')
    if body_start < 0:
        return []
    body = md_text[body_start:]

    # ページ単位に分割
    parts = PAGE_PAT.split(body)
    pages: list[tuple[int, str]] = []
    for i in range(1, len(parts), 2):
        pn = int(parts[i]); ptxt = parts[i+1] if i+1 < len(parts) else ''
        pages.append((pn, ptxt.strip()))

    # PyMuPDF fallback: markdown 内容が薄いページは PyMuPDF text と合成
    if pdf_path and pdf_path.exists():
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            # markdown 内容から「実質テキスト」を測る (画像マーカー除外)
            img_marker_pat = _re.compile(r'\[画像[^\]]*\]|（AI画像解析対象）')
            new_pages = []
            for pn, ptxt in pages:
                substantive = img_marker_pat.sub('', ptxt).strip()
                substantive_len = len(_re.sub(r'\s+', '', substantive))
                if substantive_len < 100 and pn - 1 < len(doc):
                    # PyMuPDF で text 抽出
                    pymupdf_text = doc[pn-1].get_text('text', sort=True).strip()
                    if pymupdf_text and len(_re.sub(r'\s+', '', pymupdf_text)) > substantive_len:
                        # markdown のマーカー残しつつ PyMuPDF text を本文として採用
                        combined = ptxt + '\n\n' + pymupdf_text if ptxt else pymupdf_text
                        new_pages.append((pn, combined))
                        continue
                new_pages.append((pn, ptxt))
            pages = new_pages
            doc.close()
        except Exception as e:
            print(f'  ⚠️ PyMuPDF fallback failed for {pdf_name}: {e}')

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ('#', 'h1'), ('##', 'h2'), ('###', 'h3'),
        ],
        strip_headers=False,
    )
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars, chunk_overlap=overlap,
        separators=['\n\n', '\n', '。', '、', ' ', ''],
    )

    chunks: list[dict] = []
    for page_num, page_text in pages:
        if not page_text.strip():
            continue

        # ページ内を markdown 見出しで切る
        try:
            md_chunks = header_splitter.split_text(page_text)
        except Exception:
            md_chunks = [type('X', (), {'page_content': page_text, 'metadata': {}})()]

        if not md_chunks:
            md_chunks = [type('X', (), {'page_content': page_text, 'metadata': {}})()]

        for mc in md_chunks:
            content = mc.page_content
            meta_h = mc.metadata if hasattr(mc, 'metadata') else {}
            section = ' / '.join([str(v) for v in meta_h.values()]) if meta_h else ''

            # 1000字超えるなら更に分割
            if len(content) <= max_chars:
                pieces = [content]
            else:
                pieces = sub_splitter.split_text(content)

            for j, piece in enumerate(pieces):
                piece = piece.strip()
                if not piece:
                    continue
                # 短い chunk (画像マーカーのみ等) も保持: ページ番号メタが大事なので
                # 検索ヒットさせるためsection情報を必ず本文先頭に prepend
                text_for_embed = f'{section}\n{piece}' if section and section not in piece else piece
                # 画像のみページは page番号 + ファイル名を本文に含めてマッチ可能にする
                if len(piece) < 80:
                    text_for_embed = f'{pdf_name} ページ{page_num}\n{text_for_embed}'
                cid = hashlib.sha256(
                    f'{pdf_name}|p{page_num}|s{j}|{text_for_embed[:200]}'.encode('utf-8')
                ).hexdigest()
                chunks.append({
                    'id': cid,
                    'text': text_for_embed,
                    'source': pdf_name,
                    'page': page_num,
                    'sub': j,
                    'section': section,
                })
    return chunks


def load_all_chunks(pp_dir: Path, max_chars: int, overlap: int,
                    pdf_dir: Path = None) -> list[dict]:
    out = []
    for sub in sorted(pp_dir.iterdir()):
        if not sub.is_dir(): continue
        txt = sub / 'extracted_text.txt'
        if not txt.exists():
            print(f'  ⚠️ {sub.name}: no extracted_text.txt'); continue
        md = txt.read_text(encoding='utf-8')
        # PyMuPDF fallback 用の元PDFパス
        pdf_path = (pdf_dir / f'{sub.name}.pdf') if pdf_dir else None
        cs = parse_md_to_page_chunks(md, f'{sub.name}.pdf', pdf_path, max_chars, overlap)
        out.extend(cs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pp-dir', default='_bench/_pp')
    ap.add_argument('--pdf-dir', default='../Fujitsu-RAG-Hard-Benchmark/dataset/PDFs',
                    help='PyMuPDF fallback 用の元PDFフォルダ')
    ap.add_argument('--collection', default='fjrag_hard_md')
    ap.add_argument('--max-chars', type=int, default=1000)
    ap.add_argument('--overlap', type=int, default=120)
    ap.add_argument('--batch-size', type=int, default=100)
    args = ap.parse_args()

    pp_dir = (_proj / args.pp_dir).resolve()
    if not pp_dir.exists():
        print(f'preprocessed dir not found: {pp_dir}'); sys.exit(1)

    print(f'PP dir: {pp_dir}')
    print(f'Collection: {args.collection}')

    pdf_dir = (_proj / args.pdf_dir).resolve()
    if not pdf_dir.exists():
        print(f'PDF dir not found (fallback無効): {pdf_dir}'); pdf_dir = None

    t0 = time.time()
    chunks = load_all_chunks(pp_dir, args.max_chars, args.overlap, pdf_dir)
    print(f'\n{len(chunks)} chunks built ({time.time()-t0:.1f}s)')

    # 統計
    if chunks:
        lens = [len(c['text']) for c in chunks]
        print(f'len min/med/max: {min(lens)} / {sorted(lens)[len(lens)//2]} / {max(lens)}')
        from collections import Counter
        by_src = Counter(c['source'] for c in chunks)
        print(f'\n上位source:')
        for src, n in by_src.most_common(10):
            print(f'  {n:5d}  {src}')

    # PGVector投入
    from langchain_core.documents import Document
    from graphrag_core.config import reset_settings, get_settings
    from graphrag_core.llm.factory import create_embeddings
    from graphrag_core.db.utils import (
        ensure_embedding_id_unique, ensure_schema_compatibility, ensure_hnsw_index,
        add_connection_timeout, batch_pgvector_from_documents,
        ensure_tokenized_schema, batch_update_tokenized,
    )
    from graphrag_core.text.japanese import get_japanese_processor

    reset_settings(); s = get_settings()
    embeddings = create_embeddings()
    docs = [
        Document(page_content=c['text'],
                 metadata={'id': c['id'], 'source': c['source'],
                           'page': c['page'], 'sub': c['sub'],
                           'section': c.get('section', '')})
        for c in chunks
    ]

    ensure_embedding_id_unique(s.pg_conn)
    ensure_schema_compatibility(s.pg_conn)
    ensure_hnsw_index(s.pg_conn)
    pg = add_connection_timeout(s.pg_conn, timeout=30)

    print(f'\nPGVector投入中 (batch={args.batch_size})...')
    t1 = time.time()
    batch_pgvector_from_documents(
        docs, embeddings,
        connection=pg, collection_name=args.collection,
        pre_delete_collection=True, batch_size=args.batch_size,
        progress_callback=lambda i, total, n:
            (i % 500 != (i+n) % 500 or i+n == total) and print(f'  {i+n}/{total}'),
    )
    print(f'PGVector OK ({time.time()-t1:.1f}s)')

    # 日本語トークン化 (ハイブリッド検索用)
    jp = get_japanese_processor()
    if jp and s.enable_japanese_search:
        print('日本語トークン化中...')
        t2 = time.time()
        ensure_tokenized_schema(s.pg_conn)
        for d in docs:
            try:
                d.metadata['tokenized_content'] = jp.tokenize(d.page_content)
            except Exception:
                d.metadata['tokenized_content'] = None
        n = batch_update_tokenized(s.pg_conn, docs)
        print(f'  {n}件 ({time.time()-t2:.1f}s)')

    print(f'\n完了: {len(docs)} chunks → {args.collection}  total={time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
