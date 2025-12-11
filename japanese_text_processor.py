"""
日本語テキスト処理モジュール
Sudachiによるトークン化と正規化
"""
from typing import Optional
import unicodedata

try:
    from sudachipy import tokenizer, dictionary
    SUDACHI_AVAILABLE = True
except ImportError:
    SUDACHI_AVAILABLE = False


class JapaneseTextProcessor:
    """日本語テキストのトークン化と正規化"""

    def __init__(self, min_token_length: int = 2):
        if not SUDACHI_AVAILABLE:
            raise ImportError(
                "sudachipy が見つかりません。インストールしてください:\n"
                "pip install sudachipy sudachidict_core"
            )

        self.tokenizer_obj = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.C  # 長単位（推奨）
        self.min_token_length = min_token_length

    def tokenize(self, text: str) -> str:
        """
        日本語テキストを正規化＋トークン化

        Args:
            text: 入力テキスト

        Returns:
            スペース区切りのトークン文字列（例: "桃太郎 が 鬼 を 倒す"）
        """
        # 正規化
        text = self._normalize(text)

        # Sudachiでトークン化
        tokens = [
            m.surface()
            for m in self.tokenizer_obj.tokenize(text, self.mode)
            if len(m.surface()) >= self.min_token_length
        ]

        return " ".join(tokens)

    def _normalize(self, text: str) -> str:
        """テキスト正規化（NFKC + 小文字化）"""
        # NFKC正規化（全角英数→半角、濁点統合など）
        text = unicodedata.normalize('NFKC', text)
        # 小文字化
        text = text.lower()
        return text


# グローバルインスタンス（遅延初期化）
_processor_instance: Optional[JapaneseTextProcessor] = None

def get_japanese_processor(min_token_length: int = 2) -> Optional[JapaneseTextProcessor]:
    """シングルトンパターンでプロセッサを取得"""
    global _processor_instance

    if not SUDACHI_AVAILABLE:
        return None

    if _processor_instance is None:
        _processor_instance = JapaneseTextProcessor(min_token_length)

    return _processor_instance
