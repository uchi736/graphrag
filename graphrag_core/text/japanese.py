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

    # 検索に不要な助詞・助動詞・機能語（surface + normalized_form 両方）
    _STOPWORDS = frozenset({
        # 助詞
        "の", "は", "が", "を", "に", "で", "と", "も", "か",
        "な", "へ", "や", "て", "だ", "た", "れ", "ば", "ず", "ぬ", "ん",
        # 助動詞・補助動詞（surface形）
        "する", "いる", "ある", "なる",
        # 助動詞・補助動詞（normalized_form: Sudachi正規化後）
        "為る", "居る", "有る", "成る", "れる", "られる", "下さる",
        # 形式名詞
        "こと", "ため", "もの", "よう", "ところ",
        # 指示詞
        "これ", "それ", "あれ", "この", "その", "あの",
        # その他の機能語
        "さ", "し", "せ",
    })

    def __init__(self, min_token_length: int = 1):
        if not SUDACHI_AVAILABLE:
            raise ImportError(
                "sudachipy が見つかりません。インストールしてください:\n"
                "pip install sudachipy sudachidict_core"
            )

        self.tokenizer_obj = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.B  # 中単位（検索向き）
        self.min_token_length = min_token_length

    def tokenize(self, text: str) -> str:
        """
        日本語テキストを正規化＋トークン化

        Args:
            text: 入力テキスト

        Returns:
            スペース区切りのトークン文字列（例: "桃太郎 鬼 倒す"）
        """
        # 正規化
        text = self._normalize(text)

        # Sudachiでトークン化（正規化形 + ストップワード除去）
        tokens = []
        for m in self.tokenizer_obj.tokenize(text, self.mode):
            norm = m.normalized_form().strip()
            if norm and len(norm) >= self.min_token_length and norm not in self._STOPWORDS:
                tokens.append(norm)

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

def get_japanese_processor(min_token_length: int = 1) -> Optional[JapaneseTextProcessor]:
    """シングルトンパターンでプロセッサを取得"""
    global _processor_instance

    if not SUDACHI_AVAILABLE:
        return None

    if _processor_instance is None:
        _processor_instance = JapaneseTextProcessor(min_token_length)

    return _processor_instance
