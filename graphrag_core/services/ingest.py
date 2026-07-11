"""取り込みサービス: bytes 入力からの Document / CSVエッジ生成（st非依存）。

ui/system.py の load_documents / load_csv_edges から UploadedFile 依存を除去した版。
Streamlit 側は UploadedFile→(name, bytes) 変換だけ行いここへ委譲、
FastAPI 側は multipart の (filename, bytes) をそのまま渡す。
"""
from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

from graphrag_core.document.pdf import load_pdf_text


def load_document_from_bytes(filename: str, data: bytes) -> Document:
    """1ファイル（pdf/txt/md/その他テキスト）を Document に変換する。"""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            text_content = load_pdf_text(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        text_content = data.decode("utf-8", errors="replace")

    return Document(page_content=text_content, metadata={"source": filename})


def load_documents_from_bytes(files: List[tuple[str, bytes]]) -> List[Document]:
    return [load_document_from_bytes(name, data) for name, data in files]


def load_csv_edges_from_bytes(data: bytes) -> List[Dict[str, str]]:
    """edges.csv (source,target,label 系の別名吸収) → エッジリスト。"""
    text = data.decode("utf-8-sig", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    edges: List[Dict[str, str]] = []
    for row in reader:
        if not row:
            continue
        normalized = {k.strip().lower() if k else k: v for k, v in row.items()}
        src = (normalized.get("source") or normalized.get("from") or normalized.get("src") or "").strip()
        tgt = (normalized.get("target") or normalized.get("to") or normalized.get("dst") or "").strip()
        rel = (normalized.get("label") or normalized.get("relation") or normalized.get("rel") or "RELATED_TO").strip()
        if not src or not tgt:
            continue
        edges.append({"source": src, "target": tgt, "label": rel})
    return edges
