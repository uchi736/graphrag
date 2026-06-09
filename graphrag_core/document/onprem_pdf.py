"""オンプレPDF前処理アダプタ

`preprocessing_optimizer` の `UnifiedProcessor` を呼び、結果の
`extracted_text.txt` (Markdown) を1つの文字列として返す。

- 画像解析は DGX Spark 上のVision vLLM を使用（OpenAI互換）
- 環境変数:
    PDF_BACKEND=vllm|paddleocr|hybrid|none
    VLLM_VISION_ENDPOINT, VLLM_VISION_MODEL, VLLM_VISION_API_KEY
    PREPROCESSING_OPTIMIZER_PATH (別プロジェクトのパス)
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from graphrag_core.config import get_settings

logger = logging.getLogger(__name__)

_IMPORT_READY = False


def _ensure_importable() -> None:
    """preprocessing_optimizer を sys.path に追加し、core パッケージを解決可能にする"""
    global _IMPORT_READY
    if _IMPORT_READY:
        return

    s = get_settings()
    raw_path = s.preprocessing_optimizer_path
    base = Path(raw_path)
    if not base.is_absolute():
        # graphrag リポジトリルート基準で解決
        base = (Path(__file__).resolve().parents[2] / raw_path).resolve()

    if not base.exists():
        raise FileNotFoundError(
            f"preprocessing_optimizer が見つかりません: {base} "
            f"(PREPROCESSING_OPTIMIZER_PATH で指定してください)"
        )

    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    _IMPORT_READY = True


def extract_pdf_onprem(pdf_path: str, backend: Optional[str] = None) -> str:
    """オンプレ前処理パイプラインでPDFを処理し、Markdownテキストを返す

    Args:
        pdf_path: PDFファイルのパス
        backend: 画像解析バックエンド (vllm/paddleocr/hybrid/none)。
                 未指定時は PDF_BACKEND 環境変数 or "vllm"

    Returns:
        抽出されたMarkdownテキスト。失敗時は空文字列
    """
    _ensure_importable()
    s = get_settings()
    backend = backend or s.pdf_backend or "vllm"

    # preprocessing_optimizer 側の VLLMBackend は VLLM_BASE_URL/VLLM_MODEL/VLLM_API_KEY
    # を参照するので、Vision専用envを一時的に流し込む。
    prev = {
        "VLLM_BASE_URL": os.environ.get("VLLM_BASE_URL"),
        "VLLM_MODEL": os.environ.get("VLLM_MODEL"),
        "VLLM_API_KEY": os.environ.get("VLLM_API_KEY"),
    }
    try:
        if backend == "vllm":
            os.environ["VLLM_BASE_URL"] = s.vllm_vision_endpoint
            os.environ["VLLM_MODEL"] = s.vllm_vision_model
            os.environ["VLLM_API_KEY"] = s.vllm_vision_api_key

        from core.processor import UnifiedProcessor  # type: ignore

        with tempfile.TemporaryDirectory(prefix="onprem_pdf_") as tmp:
            out_dir = Path(tmp) / "out"

            # paddleocr_remote は preprocessing_optimizer の get_backend() に無いので
            # backend="none" で起動して self.backend を差し替える
            if backend == "paddleocr_remote":
                from graphrag_core.document.paddleocr_remote import PaddleXRemoteBackend
                processor = UnifiedProcessor(output_format="text", backend="none")
                processor.backend = PaddleXRemoteBackend()
                processor.backend_name = "paddleocr_remote"
            else:
                processor = UnifiedProcessor(output_format="text", backend=backend)

            processor.process(str(pdf_path), output_dir=str(out_dir), parallel=True)

            text_file = out_dir / "extracted_text.txt"
            if text_file.exists():
                return text_file.read_text(encoding="utf-8")
            logger.warning("extracted_text.txt が生成されませんでした: %s", text_file)
            return ""
    except Exception as e:
        logger.error("オンプレPDF処理失敗: %s", e, exc_info=True)
        return ""
    finally:
        # env を元に戻す
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
