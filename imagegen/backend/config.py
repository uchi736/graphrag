"""アプリ設定（環境変数から読み込み）。

すべて環境変数で上書き可能。`.env.sample` を参照。
DGX Spark 上では ComfyUI が別プロセス（既定 127.0.0.1:8188）で動く前提。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
_APP_DIR = _BACKEND_DIR.parent  # imagegen/


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # ComfyUI 接続先
    comfyui_url: str = "http://127.0.0.1:8188"
    # ポーリング間隔（秒）と 1 ジョブあたりの上限待ち時間（秒）
    poll_interval: float = 1.5
    job_timeout: float = 900.0
    # ComfyUI への HTTP タイムアウト（接続・読み取り）
    http_timeout: float = 30.0

    # 保存先
    output_dir: Path = _APP_DIR / "outputs"
    upload_dir: Path = _APP_DIR / "uploads"

    # フロントエンド静的配信ディレクトリ
    frontend_dir: Path = _APP_DIR / "frontend"

    # 開発時 CORS（Vite 等の別オリジンから叩く場合のみ）
    cors_origins: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            comfyui_url=os.getenv("COMFYUI_URL", cls.comfyui_url).rstrip("/"),
            poll_interval=float(os.getenv("COMFYUI_POLL_INTERVAL", cls.poll_interval)),
            job_timeout=float(os.getenv("IMAGEGEN_JOB_TIMEOUT", cls.job_timeout)),
            http_timeout=float(os.getenv("COMFYUI_HTTP_TIMEOUT", cls.http_timeout)),
            output_dir=Path(os.getenv("IMAGEGEN_OUTPUT_DIR", str(cls.output_dir))),
            upload_dir=Path(os.getenv("IMAGEGEN_UPLOAD_DIR", str(cls.upload_dir))),
            frontend_dir=Path(os.getenv("IMAGEGEN_FRONTEND_DIR", str(cls.frontend_dir))),
            cors_origins=os.getenv("IMAGEGEN_CORS_ORIGINS", cls.cors_origins),
        )

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
