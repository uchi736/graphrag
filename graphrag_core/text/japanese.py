"""
日本語テキスト処理モジュール
Sudachiによるトークン化と正規化
"""
from typing import Optional
import threading
import unicodedata

try:
    from sudachipy import tokenizer, dictionary
    SUDACHI_AVAILABLE = True
except ImportError:
    SUDACHI_AVAILABLE = False


def normalize_entity_text(text: str) -> str:
    """エンティティ名照合用の正規化（NFKC + 小文字化 + 空白圧縮）。

    KGノードの search_keys 構築（ビルド側）と質問エンティティの照合（検索側）の
    両方でこの関数を使い、表記揺れ（全角/半角・大文字/小文字）による
    ミスマッチを防ぐ。Sudachi非依存。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    return " ".join(text.split())


import re as _re

_HIRAGANA_RX = _re.compile(r"[ぁ-ん]")
_KANJI_KATA_RX = _re.compile(r"[一-龯ァ-ヶ]")
_SPACE_DOT_RX = _re.compile(r"[\s・･]")


def kana_variant_key(text: str) -> str:
    """かな揺れ照合キー（送り仮名・助詞・末尾長音を除いた骨格）。

    「ガス軸受/ガス軸受け」「データの連携/データ連携」「サーバ/サーバー」を
    同一キーに落とす。誤統合を防ぐガード:
    - 漢字/カタカナを含まない語（型番・英数字）は対象外（U7314/U7414等の保護）
    - 骨格が元の文字数の過半を保持しない場合は無効
      （「わかりやすい情報」→「情報」のような内容語の脱落を弾く）
    - 長音「ー」は末尾のみ除去（「サーバー」→「サーバ」。語中のーは音価があるため保持）

    Returns:
        骨格キー。ガードに引っかかる場合は空文字
    """
    t = _SPACE_DOT_RX.sub("", normalize_entity_text(text))
    if not t or not _KANJI_KATA_RX.search(t):
        return ""
    skel = _HIRAGANA_RX.sub("", t.rstrip("ー"))
    if len(skel) < 3 or len(skel) * 2 <= len(t):
        return ""
    return skel


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
        # SudachiPy の Tokenizer オブジェクトはスレッドセーフではなく、
        # 複数スレッドが同時に tokenize() を呼ぶと Rust 側で
        # "Already borrowed" panic が出る（バッチ評価の並列実行で露呈）。
        # シングルトンを共有しつつ tokenize をロックで直列化する。
        # （1クエリのトークン化は軽量で、LLM呼び出しがボトルネックのため影響軽微）
        self._lock = threading.Lock()

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
        # tokenize() 呼び出しはスレッドセーフでないためロックで保護
        tokens = []
        with self._lock:
            morphemes = list(self.tokenizer_obj.tokenize(text, self.mode))
        for m in morphemes:
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
