"""画像生成・編集アプリ FastAPI エントリポイント。

起動:
    python -m uvicorn imagegen.backend.main:app --host 0.0.0.0 --port 8100
    （ジョブキューがプロセス内のため --workers 1 前提）

フロントエンド（imagegen/frontend）を同一オリジンで静的配信する。
別オリジン（Vite 等）から叩く場合は IMAGEGEN_CORS_ORIGINS を設定。
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .routes import router
from .state import AppState

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
for noisy in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.ctx = AppState.create(settings)
    logger.info(
        "imagegen 起動 (comfyui=%s, output=%s)",
        settings.comfyui_url,
        settings.output_dir,
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Image Generation & Editing API",
        description="DGX Spark 上の ComfyUI をバックエンドにした画像生成・編集アプリ",
        version="0.1.0",
        lifespan=lifespan,
    )

    cors = os.getenv("IMAGEGEN_CORS_ORIGINS", "")
    if cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in cors.split(",") if o.strip()],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(router)

    # フロントエンド静的配信（explicit ルートの後にマウントするので API が優先）
    settings = get_settings()
    if settings.frontend_dir.exists():
        app.mount(
            "/",
            StaticFiles(directory=str(settings.frontend_dir), html=True),
            name="frontend",
        )

    return app


app = create_app()
