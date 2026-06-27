"""UI フィードバック共通ヘルパ

エラー表示を統一する。人間向けの一行説明を st.error で出し、
トレースバックは折りたたみ（既定で隠す）。アプリ全体で同じ見た目にする。
"""

from __future__ import annotations

import traceback

import streamlit as st


def show_error(message: str, exc: Exception | None = None) -> None:
    """人間向けエラー + 折りたたみトレースバック。

    Parameters
    ----------
    message : str
        ユーザー向けの説明（例: "グラフ構築に失敗しました"）。
    exc : Exception, optional
        捕捉した例外。あれば説明に要約を付し、詳細をexpanderに格納する。
    """
    if exc is not None:
        st.error(f"{message}: {exc}")
        with st.expander("詳細 (トレースバック)", expanded=False):
            st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    else:
        st.error(message)
        with st.expander("詳細 (トレースバック)", expanded=False):
            st.code(traceback.format_exc())
