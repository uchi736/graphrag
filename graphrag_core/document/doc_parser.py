"""doc-parser（DGX :8770）連携 — 構造保持Markdown抽出。

doc-parser は MinerU(pipeline, -l japan) と Docling を同梱したHTTPサービス。
PDF/画像を「表・見出し構造を保った Markdown」に変換して返すため、
従来の OCR行抽出（PaddleX）で潰れていた表（様式↔条文 等）がチャンクに残る。

前処理ツール比較（2026-06-27, NCR帳票）の結論:
- MinerU pipeline: 表構造◎ × 日本語◎（和文はvlmでなくpipelineを使うこと）
- Docling: 代替エンジン（engine="docling" で切替可能）

環境変数:
    PDF_PROCESSOR=doc_parser
    DOC_PARSER_ENDPOINT=http://192.168.0.250:8770
    DOC_PARSER_ENGINE=mineru | docling
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_pdf_doc_parser(path: str, engine: Optional[str] = None,
                           endpoint: Optional[str] = None,
                           timeout: float = 900.0) -> str:
    """doc-parser で PDF を構造保持 Markdown に変換して返す。"""
    import requests
    from graphrag_core.config import get_settings

    s = get_settings()
    endpoint = (endpoint or s.doc_parser_endpoint).rstrip("/")
    engine = (engine or s.doc_parser_engine or "mineru").lower()
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    filename = Path(path).name

    if engine == "docling":
        resp = requests.post(
            f"{endpoint}/docling/parse",
            json={"file_b64": b64, "filename": filename, "ocr": False,
                  "to_html": False},
            timeout=timeout)
    else:
        resp = requests.post(
            f"{endpoint}/mineru/parse",
            json={"file_b64": b64, "filename": filename, "lang": "japan"},
            timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    md = data.get("markdown") or ""
    logger.info("doc-parser(%s): %s -> %d chars (%.1fs)",
                data.get("engine", engine), filename, len(md),
                data.get("elapsed_seconds", -1))
    return md
