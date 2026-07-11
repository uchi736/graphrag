"""オンプレ PDF 処理プロセッサ（完全ローカル）

兄弟プロジェクト graphrag の paddleocr_remote バックエンドと HTTP レベルで対称な
オンプレ PDF 抽出を提供する。DGX Spark 上の PaddleX PP-OCRv5 serving (port 8005) を
HTTP 経由で呼び、PDF をページ毎テキストに変換する。

PDF_PROCESSOR / PDF_BACKEND 環境変数でバックエンドを選択:
  PDF_PROCESSOR=onprem (既定) + PDF_BACKEND=paddleocr_remote (既定)  -> PaddleX
  PDF_PROCESSOR=azure_di                                            -> Azure DI (ロールバック用)

戻り値はページ毎テキストの list[str]（既存 run.py / app.py の契約と同一）。

PaddleX API: POST {PADDLEX_ENDPOINT}/ocr
  Request:  {"file": "<base64>", "fileType": 0}   (0=pdf, 1=image)
  Response: {"errorCode": 0,
             "result": {"ocrResults": [{"prunedResult": {"rec_texts": [...]}}, ...]}}
"""

import base64
import logging
import os
import re
import time
from http.client import RemoteDisconnected

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

logger = logging.getLogger(__name__)

# WSAECONNRESET (10054) や mid-stream な切断は requests.ConnectionError として
# 上がってくる。ConnectionResetError/Aborted は念のため明示。
_RETRYABLE_EXC = (
    RequestsConnectionError,
    RequestsTimeout,
    ConnectionResetError,
    ConnectionAbortedError,
    RemoteDisconnected,
)
# 即時 / 1秒 / 3秒
_RETRY_BACKOFF = (0, 1, 3)


def _clean_japanese_spaces(text: str) -> str:
    """日本語文字に隣接する不要なスペースを除去（既存 EDC と同一ロジック）"""
    text = re.sub(r'[ ]+([ぁ-んァ-ヴー一-龠々〆〤])', r'\1', text)
    text = re.sub(r'([ぁ-んァ-ヴー一-龠々〆〤])[ ]+', r'\1', text)
    return text.strip()


def _post_with_retry(url: str, payload: dict, timeout: float, name: str) -> dict:
    """OCR POST を最大3回試行。base64サイズと試行毎のエラーをログ。"""
    b64 = payload.get("file", "")
    size_mb = len(b64) / 1024 / 1024
    logger.info("OCR POST %s base64=%.2f MB -> %s", name, size_mb, url)

    last_exc = None
    for attempt in range(len(_RETRY_BACKOFF)):
        wait = _RETRY_BACKOFF[attempt]
        if wait:
            time.sleep(wait)
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            if attempt > 0:
                logger.info("OCR succeeded on retry #%d for %s (size=%.2f MB)", attempt, name, size_mb)
            return r.json()
        except _RETRYABLE_EXC as e:
            last_exc = e
            logger.warning(
                "OCR attempt %d/%d failed for %s (size=%.2f MB): %s: %s",
                attempt + 1, len(_RETRY_BACKOFF), name, size_mb, type(e).__name__, e,
            )
    assert last_exc is not None
    raise last_exc


def _extract_paddlex_remote(file_path: str) -> list:
    """PaddleX PP-OCRv5 serving に PDF を送り、ページ毎テキストのリストを返す"""
    endpoint = os.environ.get("PADDLEX_ENDPOINT", "http://localhost:8005").rstrip("/")
    timeout = float(os.environ.get("PADDLEX_TIMEOUT", "180"))
    name = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    payload = {"file": b64, "fileType": 0}  # 0 = pdf
    data = _post_with_retry(f"{endpoint}/ocr", payload, timeout, name)

    if data.get("errorCode") not in (0, None):
        msg = data.get("errorMsg", "unknown")
        raise RuntimeError(f"PaddleX error on {name}: {msg}")

    ocr_results = (data.get("result") or {}).get("ocrResults") or []

    # ocrResults は PDF のページ毎に1要素
    texts = []
    for ocr in ocr_results:
        pruned = ocr.get("prunedResult") or {}
        rec_texts = pruned.get("rec_texts") or []
        page_text = "\n".join(t for t in rec_texts if t)
        page_text = _clean_japanese_spaces(page_text)
        if page_text:
            texts.append(page_text)

    return texts


def _extract_azure_di(file_path: str) -> list:
    """Azure Document Intelligence を使用してPDFからテキストをページごとに抽出

    緊急ロールバック用（PDF_PROCESSOR=azure_di で選択）。
    AZURE_DI_ENDPOINT / AZURE_DI_API_KEY が必要。
    """
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

    endpoint = os.environ.get("AZURE_DI_ENDPOINT")
    api_key = os.environ.get("AZURE_DI_API_KEY")
    model = os.environ.get("AZURE_DI_MODEL", "prebuilt-layout")

    if not endpoint or not api_key:
        raise ValueError(
            "PDF処理(azure_di)にはAZURE_DI_ENDPOINTとAZURE_DI_API_KEYの環境変数が必要です（.envファイルで設定）"
        )

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key)
    )

    with open(file_path, "rb") as f:
        file_content = f.read()

    poller = client.begin_analyze_document(
        model,
        AnalyzeDocumentRequest(bytes_source=file_content),
    )
    result = poller.result()

    texts = []
    if result.pages:
        for page in result.pages:
            page_num = page.page_number
            page_content = []
            if result.paragraphs:
                for para in result.paragraphs:
                    if hasattr(para, 'bounding_regions') and para.bounding_regions:
                        for region in para.bounding_regions:
                            if region.page_number == page_num:
                                page_content.append(para.content)
                                break

            if page_content:
                text = "\n".join(page_content)
                text = _clean_japanese_spaces(text)
                texts.append(text)

    # ページごとの抽出が空の場合、全体コンテンツを使用
    if not texts and result.content:
        texts = [result.content]

    return texts


def extract_text_from_pdf(file_path: str, processor: str = None, backend: str = None) -> list:
    """PDFからテキストをページごとに抽出（オンプレ既定）

    Args:
        file_path: PDFファイルのパス
        processor: "onprem"(既定) / "azure_di"。未指定時は PDF_PROCESSOR env。
        backend:   onprem時のバックエンド。"paddleocr_remote"(既定)。未指定時は PDF_BACKEND env。

    Returns:
        ページごとのテキストリスト
    """
    processor = processor or os.environ.get("PDF_PROCESSOR", "onprem")

    if processor == "onprem":
        backend = backend or os.environ.get("PDF_BACKEND", "paddleocr_remote")
        if backend == "paddleocr_remote":
            return _extract_paddlex_remote(file_path)
        raise ValueError(f"Unsupported PDF_BACKEND for onprem: {backend}")

    if processor == "azure_di":
        return _extract_azure_di(file_path)

    raise ValueError(f"Unsupported PDF_PROCESSOR: {processor}")
