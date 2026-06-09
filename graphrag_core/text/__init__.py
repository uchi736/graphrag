"""graphrag_core.text - テキスト処理（チャンキング・日本語トークン化）"""

from graphrag_core.text.chunking import create_markdown_chunks, format_chunk_source
from graphrag_core.text.japanese import (
    JapaneseTextProcessor,
    get_japanese_processor,
    SUDACHI_AVAILABLE,
)

__all__ = [
    "create_markdown_chunks",
    "format_chunk_source",
    "JapaneseTextProcessor",
    "get_japanese_processor",
    "SUDACHI_AVAILABLE",
]
