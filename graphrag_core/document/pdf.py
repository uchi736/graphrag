"""PDFテキスト抽出ユーティリティ

`load_pdf_text()` が唯一のエントリポイント。`PDF_PROCESSOR` 環境変数で
処理方式を切り替え、失敗時はより軽量な方式へ自動フォールバックする。
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _extract_pymupdf(path: str) -> str:
    """PyMuPDFで全ページのテキストを抽出して結合（オフライン・軽量）"""
    import fitz

    pdf_doc = fitz.open(str(path))
    try:
        text_parts = []
        for page_num in range(len(pdf_doc)):
            text = pdf_doc[page_num].get_text("text", sort=True)
            if text.strip():
                text_parts.append(text)
    finally:
        pdf_doc.close()
    return "\n\n".join(text_parts)


# 後方互換
def extract_pdf_text(path: str) -> str:
    """PyMuPDF直接呼び出し（レガシー、load_pdf_text 推奨）"""
    return _extract_pymupdf(path)


def load_pdf_text(path: str, processor: Optional[str] = None) -> str:
    """設定に応じたパイプラインでPDFを処理しテキストを返す

    Args:
        path: PDFファイルパス
        processor: "onprem" | "azure_di" | "pymupdf" | None (Settings を参照)

    Returns:
        抽出テキスト。全方式失敗時は空文字列
    """
    from graphrag_core.config import get_settings

    s = get_settings()
    processor = processor or s.pdf_processor or "onprem"

    if processor == "doc_parser":
        return load_pdf_with_figures(path, processor)[0]

    if processor == "onprem":
        try:
            from graphrag_core.document.onprem_pdf import extract_pdf_onprem
            text = extract_pdf_onprem(path)
            if text:
                return text
            logger.warning("onprem PDF処理が空結果 → pymupdf にフォールバック")
        except Exception as e:
            logger.warning("onprem PDF処理失敗 → pymupdf にフォールバック: %s", e)
        return _extract_pymupdf(path)

    if processor == "azure_di":
        try:
            from graphrag_core.document.azure_di import AzureDocumentIntelligenceProcessor
            proc = AzureDocumentIntelligenceProcessor(s)
            docs = proc.process(str(path))
            if docs:
                return docs[0].page_content
            logger.warning("Azure DI が空結果 → pymupdf にフォールバック")
        except Exception as e:
            logger.warning("Azure DI失敗 → pymupdf にフォールバック: %s", e)
        return _extract_pymupdf(path)

    # default: pymupdf
    return _extract_pymupdf(path)


def load_pdf_with_figures(path: str, processor: Optional[str] = None) -> tuple[str, list]:
    """PDFを処理し (テキスト, 図レコード) を返す。

    図レコードが得られるのは doc_parser 経路のみ（他方式は常に空リスト）。
    doc_parser 失敗時は onprem → pymupdf にフォールバック（その場合も図なし）。
    """
    from graphrag_core.config import get_settings

    s = get_settings()
    processor = processor or s.pdf_processor or "onprem"

    if processor == "doc_parser":
        # 構造保持Markdown（Docling/MinerU @ doc-parser:8770）。
        # 表が Markdown表として残るため、様式↔条文のような表情報が
        # チャンク・KG抽出に乗る（従来OCRの平文化問題への対策）
        try:
            from graphrag_core.document.doc_parser import extract_pdf_with_figures
            text, figures = extract_pdf_with_figures(path)
            if text:
                return text, figures
            logger.warning("doc-parser が空結果 → onprem にフォールバック")
        except Exception as e:
            logger.warning("doc-parser 失敗 → onprem にフォールバック: %s", e)
        try:
            from graphrag_core.document.onprem_pdf import extract_pdf_onprem
            text = extract_pdf_onprem(path)
            if text:
                return text, []
        except Exception as e:
            logger.warning("onprem も失敗 → pymupdf にフォールバック: %s", e)
        return _extract_pymupdf(path), []

    return load_pdf_text(path, processor), []
