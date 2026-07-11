"""進捗通知の抽象化（st.progress / st.spinner の置換）。

サービス層の長時間処理は ProgressFn を受け取り、節目ごとに ProgressEvent を
publish する。配送層がそれを UI に変換する:
- Streamlit: st.progress / st.info / st.warning へマッピング
- FastAPI:   ジョブレジストリ経由で SSE イベントへ変換
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ProgressEvent:
    """長時間処理の進捗1件。

    stage:   処理フェーズ識別子（load/chunk/kg_extract/csv_edges/entity_vector/
             tokenize/pgvector/consolidate/enrich/provenance など）
    current/total: カウンタ（不明なら None。spinner 相当は message のみ）
    message: 人間向けメッセージ（日本語）
    level:   info | warning | error
    ok/err:  累積成功・失敗数（KG抽出ループ用。未使用なら 0）
    """
    stage: str
    message: str = ""
    current: Optional[int] = None
    total: Optional[int] = None
    level: str = "info"
    ok: int = 0
    err: int = 0

    @property
    def percent(self) -> Optional[float]:
        if self.current is None or not self.total:
            return None
        return round(100.0 * self.current / self.total, 1)


ProgressFn = Callable[[ProgressEvent], None]


def noop_progress(event: ProgressEvent) -> None:
    """進捗通知が不要な呼び出し（CLI等）用の既定コールバック。"""


class JobCancelled(Exception):
    """協調キャンセル要求により処理を中断したことを示す。"""
