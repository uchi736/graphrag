"""プロセス内ジョブキュー（画像生成・編集の非同期実行）。

計画書 §7 第2フェーズ「ジョブキューと状態管理（同時実行は当面1件でよい）」。
単一ワーカースレッドで直列実行し、状態を queued -> running -> done/failed で管理する。
uvicorn --workers 1 前提（レジストリがプロセス内のため）。
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# runner(job, should_cancel) -> 保存済み画像パスのリスト
Runner = Callable[["ImageJob", Callable[[], bool]], List[Path]]


@dataclass
class ImageJob:
    id: str
    kind: str                         # generate | edit
    params: Dict[str, Any]
    runner: Runner
    state: str = "queued"             # queued | running | done | failed | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    prompt_id: Optional[str] = None   # ComfyUI 側 prompt_id
    error: Optional[str] = None
    images: List[Path] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "job_id": self.id,
            "kind": self.kind,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "prompt_id": self.prompt_id,
            "error": self.error,
            "images": [str(p) for p in self.images],
            "params": _public_params(self.params),
        }


def _public_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """レスポンスに載せてよいパラメータだけ抜き出す（内部フィールドを除く）。"""
    return {k: v for k, v in params.items() if not k.startswith("_")}


class JobQueue:
    """単一ワーカーで直列実行するジョブキュー。"""

    def __init__(self, max_jobs: int = 500):
        self._jobs: Dict[str, ImageJob] = {}
        self._order: List[str] = []
        self._q: "Queue[str]" = Queue()
        self._lock = threading.Lock()
        self._max_jobs = max_jobs
        self._worker = threading.Thread(target=self._run, name="imagegen-worker", daemon=True)
        self._worker.start()

    def submit(self, kind: str, params: Dict[str, Any], runner: Runner) -> ImageJob:
        job = ImageJob(id=uuid.uuid4().hex[:12], kind=kind, params=params, runner=runner)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._evict_locked()
        self._q.put(job.id)
        return job

    def get(self, job_id: str) -> Optional[ImageJob]:
        return self._jobs.get(job_id)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            ids = list(self._order)
        return [self._jobs[i].snapshot() for i in reversed(ids) if i in self._jobs]

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.state in ("queued", "running"):
            job.cancel_event.set()
            return True
        return False

    # ---- 内部 -----------------------------------------------------------

    def _evict_locked(self) -> None:
        # 古い終了済みジョブから間引く（メモリ上限）
        while len(self._order) > self._max_jobs:
            old_id = self._order.pop(0)
            self._jobs.pop(old_id, None)

    def _run(self) -> None:
        while True:
            job_id = self._q.get()
            job = self._jobs.get(job_id)
            if job is None:
                continue
            if job.cancel_event.is_set():
                job.state = "cancelled"
                job.finished_at = time.time()
                continue
            job.state = "running"
            job.started_at = time.time()
            logger.info("job %s (%s) start", job.id, job.kind)
            try:
                images = job.runner(job, job.cancel_event.is_set)
                job.images = images
                job.state = "cancelled" if job.cancel_event.is_set() else "done"
            except Exception as e:  # noqa: BLE001 — 全例外を捕捉して failed に落とす
                job.state = "failed"
                job.error = f"{type(e).__name__}: {e}"
                logger.exception("job %s failed", job.id)
            finally:
                job.finished_at = time.time()
                logger.info("job %s -> %s (%.1fs)", job.id, job.state,
                            (job.finished_at or 0) - (job.started_at or job.finished_at or 0))
