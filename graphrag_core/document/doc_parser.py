"""doc-parser（DGX :8770）連携 — 構造保持Markdown抽出。

doc-parser は MinerU(pipeline, -l japan) と Docling を同梱したHTTPサービス。
PDF/画像を「表・見出し構造を保った Markdown」に変換して返すため、
従来の OCR行抽出（PaddleX）で潰れていた表（様式↔条文 等）がチャンクに残る。

エンジン選択（前処理ツール比較 2026-06-27/07-16 の実測に基づく）:
- docling（既定）: IBM製＝**非中国系スタックで完結**。図をbase64で返すので
  図パイプライン（gemma4-visionキャプション）が成立。弱点2つは本モジュールで補正:
  ①和文の行折返しスペース → 決定的正規化で除去
  ②図base64の肥大 → 抽出してキャプション文へ置換
- mineru: 複雑帳票の結合セルは最も正確だが中国製（上海AI Lab）。
  非中国要件が無い案件用に engine="mineru" で切替可能

環境変数:
    PDF_PROCESSOR=doc_parser
    DOC_PARSER_ENDPOINT=http://192.168.0.250:8770
    DOC_PARSER_ENGINE=docling | mineru
    DOC_PARSER_FIGURE_CAPTIONS=true   # 図をgemma4-visionで説明文化して本文に差し込む
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# 日本語文字（漢字・かな・カナ・長音・々）に挟まれた単独半角スペース。
# PDFの行折り返し位置でdoclingが挿入する分かち書きアーティファクト
# （「事 業場」「はじ め」）を除去する。日本語は語間スペースを使わないため
# 単独スペースの除去は安全（連続スペース＝意図的な字下げは温存）。
_JA = r"[ぁ-んァ-ヶ一-龯々ー]"
_JA_SPACE_RX = re.compile(f"({_JA}) (?={_JA})")

# Markdown中のbase64埋め込み画像 ![alt](data:image/png;base64,....)
_B64_IMG_RX = re.compile(r"!\[[^\]]*\]\(data:image/([a-zA-Z]+);base64,([A-Za-z0-9+/=\s]+)\)")


def normalize_japanese_spacing(text: str) -> str:
    """行折り返し由来の日本語文字間スペースを除去する（決定的・LLM不要）。"""
    return _JA_SPACE_RX.sub(r"\1", text)


def extract_embedded_figures(md: str) -> Tuple[str, List[bytes]]:
    """base64埋め込み図を取り出し、本文を [図N] プレースホルダに置換する。

    3MB級のbase64をそのままチャンクに流すと破綻するため、キャプション有無に
    かかわらず必ず実行する。
    Returns: (置換済みmarkdown, 図バイト列リスト)
    """
    figures: List[bytes] = []

    def _repl(m: re.Match) -> str:
        try:
            figures.append(base64.b64decode(re.sub(r"\s+", "", m.group(2))))
        except Exception:
            return ""
        return f"[図{len(figures)}]"

    return _B64_IMG_RX.sub(_repl, md), figures


def caption_figures(figures: List[bytes], *, context: str = "",
                    timeout: float = 120.0) -> List[str]:
    """gemma4-vision（チャットLLMの画像入力）で図の説明文を生成する。

    失敗した図は空文字（プレースホルダのまま残す）。表の読み取りには使わない
    （列対応を誤る実測あり）— 図・グラフ・写真の意味説明専用。
    """
    from graphrag_core.llm.factory import create_chat_llm
    llm = create_chat_llm(temperature=0, timeout=timeout, max_retries=1)
    captions: List[str] = []
    for i, img in enumerate(figures, 1):
        try:
            b64 = base64.b64encode(img).decode()
            msg = [{"role": "user", "content": [
                {"type": "text", "text":
                 "この図（文書内の図表・グラフ・写真）の内容を、検索で見つけられるよう"
                 "2〜3文の日本語で説明してください。数値・傾向・対象物を具体的に。"
                 "前置き・見出し・記号は不要で、説明文だけを出力すること。"
                 + (f"\n文書の文脈: {context[:200]}" if context else "")},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}]
            out = llm.invoke(msg).content.strip()
            captions.append(out)
        except Exception as e:
            logger.warning("figure caption %d failed: %s", i, e)
            captions.append("")
    return captions


def extract_pdf_with_figures(path: str, engine: Optional[str] = None,
                             endpoint: Optional[str] = None,
                             timeout: float = 900.0) -> Tuple[str, List[dict]]:
    """doc-parser で PDF を構造保持 Markdown に変換し、図も切り出して返す。

    Returns:
        (markdown, figures) — figures は
        [{"index": 1始まり, "caption": 説明文, "image_path": 保存ファイル名}]。
        画像は settings.doc_parser_figures_dir に sha256先頭16桁.png で保存（冪等）。
        呼び出し側はこれを図チャンク（キャプション検索→画像表示）に使う。
    """
    import hashlib

    import requests

    from graphrag_core.config import get_settings

    s = get_settings()
    endpoint = (endpoint or s.doc_parser_endpoint).rstrip("/")
    engine = (engine or s.doc_parser_engine or "docling").lower()
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
    raw_len = len(md)

    # docling後処理: 図の抽出＋プレースホルダ化（base64肥大対策・必須）と
    # 和文スペース正規化。mineru出力には base64 が無いので実質no-op
    md, figures = extract_embedded_figures(md)
    md = normalize_japanese_spacing(md)

    records: List[dict] = []
    if figures:
        fig_dir = Path(s.doc_parser_figures_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        captions = [""] * len(figures)
        if s.doc_parser_figure_captions:
            try:
                captions = caption_figures(figures, context=md[:400])
                logger.info("doc-parser: %d/%d figures captioned",
                            sum(1 for c in captions if c), len(figures))
            except Exception as e:
                logger.warning("figure captioning skipped: %s", e)

        for i, (img, cap) in enumerate(zip(figures, captions), 1):
            name = hashlib.sha256(img).hexdigest()[:16] + ".png"
            fp = fig_dir / name
            if not fp.exists():
                fp.write_bytes(img)
            if cap:
                md = md.replace(f"[図{i}]", f"[図{i}: {cap}]")
            records.append({"index": i, "caption": cap, "image_path": name})

    logger.info("doc-parser(%s): %s -> %d chars (raw %d, figures %d, %.1fs)",
                data.get("engine", engine), filename, len(md), raw_len,
                len(figures), data.get("elapsed_seconds", -1))
    return md, records


def extract_pdf_doc_parser(path: str, engine: Optional[str] = None,
                           endpoint: Optional[str] = None,
                           timeout: float = 900.0) -> str:
    """後方互換ラッパ: Markdown本文のみ返す（図レコードは捨てる）。"""
    md, _ = extract_pdf_with_figures(path, engine=engine, endpoint=endpoint,
                                     timeout=timeout)
    return md
