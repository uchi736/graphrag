"""
chunk_utils.py - Markdown対応2段階チャンキング

Stage 1: MarkdownHeaderTextSplitter (##, ### で分割)
Stage 2: RecursiveCharacterTextSplitter (500文字で再分割)
"""
import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def _normalize_markdown_headers(text: str) -> str:
    """Markdownヘッダーを正規化（##の後にスペースがない場合を修正）

    例: "##ガス軸受" → "## ガス軸受"
    """
    # ### の後にスペースがない場合（###の後が日本語や英字で始まる）
    text = re.sub(r'^(###)([^\s#])', r'\1 \2', text, flags=re.MULTILINE)
    # ## の後にスペースがない場合（##の後が日本語や英字で始まる、###は除外）
    text = re.sub(r'^(##)([^#\s])', r'\1 \2', text, flags=re.MULTILINE)
    # # の後にスペースがない場合（#の後が日本語や英字で始まる、##は除外）
    text = re.sub(r'^(#)([^#\s])', r'\1 \2', text, flags=re.MULTILINE)
    return text


def create_markdown_chunks(
    docs: List[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    2段階Markdown対応チャンキング

    Args:
        docs: 入力ドキュメントリスト
        chunk_size: 最大チャンクサイズ（デフォルト: 1024文字）
        chunk_overlap: オーバーラップ（デフォルト: 100文字）

    Returns:
        チャンク化されたドキュメントリスト（メタデータにh1, h2, h3を含む）
    """
    # Stage 1: Markdown構造で分割（## と ### のみ）
    # # (H1) はメタデータのみ、分割境界にしない
    headers_to_split_on = [
        ("##", "h2"),
        ("###", "h3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # ヘッダーを本文に残す
    )

    # Stage 2: 文字数で再分割
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "、", " ", ""],
        length_function=len
    )

    all_chunks = []
    for doc in docs:
        # Markdownヘッダーを正規化（##の後にスペースがない場合を修正）
        normalized_content = _normalize_markdown_headers(doc.page_content)

        # # (H1) を抽出してメタデータに追加
        h1_title = _extract_h1_title(normalized_content)

        # Stage 1: Markdown分割
        md_chunks = md_splitter.split_text(normalized_content)

        for md_chunk in md_chunks:
            # md_chunkはDocumentオブジェクト（metadataにh2, h3が入る）

            # Stage 2: 文字数で再分割（長い場合のみ）
            if len(md_chunk.page_content) > chunk_size:
                sub_chunks = char_splitter.create_documents([md_chunk.page_content])
                for sub in sub_chunks:
                    # メタデータをマージ
                    sub.metadata.update(doc.metadata)  # ソースファイル名など
                    sub.metadata.update(md_chunk.metadata)  # h2, h3
                    if h1_title:
                        sub.metadata["h1"] = h1_title
                    all_chunks.append(sub)
            else:
                # そのまま使用
                md_chunk.metadata.update(doc.metadata)
                if h1_title:
                    md_chunk.metadata["h1"] = h1_title
                all_chunks.append(md_chunk)

    return all_chunks


def _extract_h1_title(text: str) -> str:
    """テキストから # (H1) タイトルを抽出（正規化済みテキストを想定）"""
    for line in text.split("\n"):
        line = line.strip()
        # 正規化済みなので "# " で始まるはず
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip()
    return ""


def format_chunk_source(chunk: Document) -> str:
    """
    チャンクのメタデータからパンくずリスト形式の出典を生成

    Args:
        chunk: Documentオブジェクト

    Returns:
        パンくずリスト形式の出典文字列
        例: "guide.pdf > 申請手続きガイド > 1. 必要書類 > 1.1 本人確認書類"
    """
    parts = []

    # ファイル名
    source = chunk.metadata.get("source", "")
    if source:
        parts.append(source)

    # ヘッダー階層
    h1 = chunk.metadata.get("h1", "")
    h2 = chunk.metadata.get("h2", "")
    h3 = chunk.metadata.get("h3", "")

    if h1:
        parts.append(h1)
    if h2:
        parts.append(h2)
    if h3:
        parts.append(h3)

    return " > ".join(parts) if parts else "Unknown"
