"""PaddleX PP-OCRv5 リモートバックエンド

DGX Spark (Grace CPU) 上の PaddleX serving (port 8005) を HTTP 経由で呼ぶ。
preprocessing_optimizer の ImageAnalysisBackend ABC に準拠し、UnifiedProcessor に
注入可能。ローカルに paddlepaddle をインストール不要（aarch64 GPU対応欠如の回避）。

API: POST {endpoint}/ocr
  Request:  {"file": "<base64-png>", "fileType": 1}  (1=image, 0=pdf)
  Response: {"errorCode": 0, "result": {"ocrResults": [{"prunedResult": {"rec_texts": [...]}}]}}
"""

from __future__ import annotations

import base64
import logging
import time
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Any, Dict

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

from graphrag_core.config import get_settings

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


class PaddleXRemoteBackend:
    """preprocessing_optimizer.ImageAnalysisBackend 互換のリモート backend"""

    def __init__(self, endpoint: str = None, timeout: float = None):
        s = get_settings()
        self.endpoint = (endpoint or s.paddlex_endpoint).rstrip("/")
        self.timeout = timeout if timeout is not None else s.paddlex_timeout

    @property
    def name(self) -> str:
        return "paddleocr_remote"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.endpoint}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _post_with_retry(self, url: str, payload: Dict[str, Any], image_name: str) -> Dict[str, Any]:
        """OCR POSTを最大3回試行。base64サイズと試行毎のエラーをログ。"""
        b64 = payload.get("file", "")
        size_mb = len(b64) / 1024 / 1024
        logger.info("OCR POST %s base64=%.2f MB -> %s", image_name, size_mb, url)

        last_exc: Exception | None = None
        for attempt in range(len(_RETRY_BACKOFF)):
            wait = _RETRY_BACKOFF[attempt]
            if wait:
                time.sleep(wait)
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                if attempt > 0:
                    logger.info(
                        "OCR succeeded on retry #%d for %s (size=%.2f MB)",
                        attempt, image_name, size_mb,
                    )
                return r.json()
            except _RETRYABLE_EXC as e:
                last_exc = e
                logger.warning(
                    "OCR attempt %d/%d failed for %s (size=%.2f MB): %s: %s",
                    attempt + 1, len(_RETRY_BACKOFF), image_name, size_mb,
                    type(e).__name__, e,
                )
        # 全試行失敗
        assert last_exc is not None
        raise last_exc

    def analyze(self, image_path: Path, page_type: str, page_info: Dict[str, Any]) -> str:
        """画像1枚をPaddleXに送り、検出テキストを結合して返す"""
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            payload = {"file": b64, "fileType": 1}  # 1 = image
            data = self._post_with_retry(
                f"{self.endpoint}/ocr", payload, image_path.name,
            )

            if data.get("errorCode") != 0:
                msg = data.get("errorMsg", "unknown")
                logger.warning("PaddleX error on %s: %s", image_path.name, msg)
                return f"[OCRエラー: {msg}]"

            ocr_results = (data.get("result") or {}).get("ocrResults") or []
            if not ocr_results:
                return ""

            # 画像1枚 → ocrResults[0]、複数ページPDFなら複数要素
            lines = []
            for ocr in ocr_results:
                pruned = ocr.get("prunedResult") or {}
                texts = pruned.get("rec_texts") or []
                lines.extend(t for t in texts if t)
            return "\n".join(lines)

        except RequestsTimeout:
            logger.warning("PaddleX timeout (%.0fs) on %s after retries", self.timeout, image_path.name)
            return "[OCRタイムアウト]"
        except Exception as e:
            logger.warning("PaddleX failed on %s after retries: %s", image_path.name, e)
            return f"[OCR失敗: {e}]"
