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

from graphrag_core.document.pdf import load_pdf_with_figures


def archive_original(filename: str, data: bytes) -> None:
    """取り込んだ原本を保管する（/docs 配信 → 根拠カードから原本へ飛ぶ用）。

    source名で保存し、再取り込みは上書き（=同一文書の改訂と同義）。
    保管失敗は取り込み自体を止めない。
    """
    from graphrag_core.config import get_settings
    try:
        safe = Path(filename).name  # パス区切りを除去
        base = Path(get_settings().doc_archive_dir)
        if not base.is_absolute():
            base = Path(__file__).resolve().parents[2] / base
        base.mkdir(parents=True, exist_ok=True)
        (base / safe).write_bytes(data)
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning("原本アーカイブ失敗（取り込みは継続）: %s", e)


def load_document_from_bytes(filename: str, data: bytes) -> Document:
    """1ファイル（pdf/txt/md/その他テキスト）を Document に変換する。

    PDFで図が切り出せた場合は metadata["figures"] に図レコードを載せる
    （chunking.expand_figure_chunks が図チャンクへ展開して pop する）。
    """
    archive_original(filename, data)
    suffix = Path(filename).suffix.lower()
    figures: list = []
    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            text_content, figures = load_pdf_with_figures(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        text_content = data.decode("utf-8", errors="replace")

    metadata: Dict = {"source": filename}
    if figures:
        metadata["figures"] = figures
    return Document(page_content=text_content, metadata=metadata)


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
