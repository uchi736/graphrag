"""アプリ共有状態（設定・ComfyUI クライアント・ジョブキュー）。"""
from __future__ import annotations

from dataclasses import dataclass

from .comfy_client import ComfyClient
from .config import Settings
from .jobs import JobQueue


@dataclass
class AppState:
    settings: Settings
    comfy: ComfyClient
    queue: JobQueue

    @classmethod
    def create(cls, settings: Settings) -> "AppState":
        settings.ensure_dirs()
        return cls(
            settings=settings,
            comfy=ComfyClient(settings.comfyui_url, http_timeout=settings.http_timeout),
            queue=JobQueue(),
        )
