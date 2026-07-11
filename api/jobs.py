"""プロセス内ジョブ管理（KG構築・増分更新などの長時間処理）。

- 重量ジョブは同時1本（実行中に投入すると 409 相当の RuntimeError）
- 進捗は ProgressFn（thread側）→ リングバッファ + 購読者キュー（SSE）
- 協調キャンセル: threading.Event → services 側の should_cancel()
- uvicorn workers=1 前提（プロセス内レジストリのため）
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional

from graphrag_core.services.progress import JobCancelled, ProgressEvent, ProgressFn


@dataclass
class Job:
    id: str
    kind: str
    state: str = "queued"        # queued | running | succeeded | failed | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    events: deque = field(default_factory=lambda: deque(maxlen=1000))
    subscribers: List[Queue] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[Dict] = None
    error: Optional[str] = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
        }

    def publish(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        if event.get("type") == "progress":
            self.progress = event["data"]
        for q in list(self.subscribers):
            q.put(event)


class JobBusy(RuntimeError):
    def __init__(self, running_job_id: str):
        super().__init__(f"別のジョブが実行中です: {running_job_id}")
        self.running_job_id = running_job_id


class JobRegistry:
    """重量ジョブの直列実行レジストリ。"""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._running_id: Optional[str] = None

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> List[Dict]:
        return [j.snapshot() for j in sorted(self._jobs.values(),
                                             key=lambda x: x.created_at, reverse=True)]

    def submit(self, kind: str, fn: Callable[[ProgressFn, Callable[[], bool]], Dict]) -> Job:
        """fn(progress, should_cancel) を別スレッドで実行するジョブを登録する。"""
        with self._lock:
            if self._running_id is not None:
                running = self._jobs.get(self._running_id)
                if running and running.state == "running":
                    raise JobBusy(self._running_id)
            job = Job(id=uuid.uuid4().hex[:12], kind=kind)
            self._jobs[job.id] = job
            self._running_id = job.id

        def progress_fn(ev: ProgressEvent) -> None:
            job.publish({"type": "progress", "data": {
                "stage": ev.stage, "message": ev.message,
                "current": ev.current, "total": ev.total,
                "percent": ev.percent, "ok": ev.ok, "err": ev.err,
                "level": ev.level,
            }})

        def runner():
            job.state = "running"
            job.started_at = time.time()
            job.publish({"type": "state", "data": {"state": "running"}})
            try:
                result = fn(progress_fn, job.cancel_event.is_set)
                job.result = result
                job.state = "succeeded"
            except JobCancelled:
                job.state = "cancelled"
            except Exception as e:
                job.state = "failed"
                job.error = f"{type(e).__name__}: {e}"
            finally:
                job.finished_at = time.time()
                with self._lock:
                    if self._running_id == job.id:
                        self._running_id = None
                job.publish({"type": "state", "data": {
                    "state": job.state, "result": job.result, "error": job.error}})

        threading.Thread(target=runner, name=f"job-{kind}-{job.id}", daemon=True).start()
        return job

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.state in ("queued", "running"):
            job.cancel_event.set()
            return True
        return False

    def subscribe(self, job_id: str):
        """SSE用: 過去イベントreplay + ライブイベントのジェネレータ。"""
        job = self._jobs.get(job_id)
        if job is None:
            return
        q: Queue = Queue()
        # 過去分をreplay（接続前の進捗を落とさない）
        for ev in list(job.events):
            yield ev
        if job.state in ("succeeded", "failed", "cancelled"):
            return
        job.subscribers.append(q)
        try:
            while True:
                try:
                    ev = q.get(timeout=15)
                except Empty:
                    yield {"type": "ping", "data": {}}
                    continue
                yield ev
                if ev.get("type") == "state" and ev["data"].get("state") in (
                        "succeeded", "failed", "cancelled"):
                    return
        finally:
            try:
                job.subscribers.remove(q)
            except ValueError:
                pass
