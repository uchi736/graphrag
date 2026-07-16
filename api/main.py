"""GraphRAG API エントリポイント。

起動:
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
    （ジョブ管理がプロセス内のため workers=1 必須）

開発時は Vite dev server(5173) からのCORSを API_CORS_ORIGINS で許可:
    API_CORS_ORIGINS=http://localhost:5173
本番は frontend/dist を同一オリジンで配信するため CORS 不要。
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.state import AppState

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
for noisy in ("httpx", "httpcore", "neo4j"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from graphrag_core.config import get_settings
    from graphrag_core.services.admin import required_env_check

    st = AppState()
    st.settings = get_settings()
    # UIで切り替えたコレクションを再起動後も維持（優先度: runtimeファイル > .env）
    from api.state import load_runtime_overrides
    _rt = load_runtime_overrides()
    if _rt.get("pg_collection"):
        logger.info("PG_COLLECTION runtime override: %s -> %s (.graphrag_runtime.json)",
                    st.settings.pg_collection, _rt["pg_collection"])
        st.settings.pg_collection = _rt["pg_collection"]
    st.env_report = required_env_check(st.settings)
    # ui/state.py:236 と同じ副作用（ParentDocumentRetriever 等が参照）
    if st.settings.pg_conn:
        os.environ.setdefault("PGVECTOR_CONNECTION_STRING", st.settings.pg_conn)

    if not st.env_report["ok"]:
        # 起動は継続し /api/health が degraded を返す。他ルートは 503。
        logger.error("必須環境変数が不足: %s", st.env_report["missing"])
    else:
        try:
            from langchain_neo4j import Neo4jGraph
            from langchain_postgres import PGVector
            from graphrag_core.db.utils import add_connection_timeout, retry_on_timeout
            from graphrag_core.llm.factory import create_chat_llm, create_embeddings

            st.embeddings = create_embeddings()
            st.llm = create_chat_llm(temperature=0)
            st.graph = Neo4jGraph(
                url=st.settings.neo4j_uri,
                username=st.settings.neo4j_user,
                password=st.settings.neo4j_pw,
                enhanced_schema=False,
            )
            pg = add_connection_timeout(st.settings.pg_conn, timeout=30)
            st.vector_store = retry_on_timeout(lambda: PGVector(
                connection=pg,
                embeddings=st.embeddings,
                collection_name=st.settings.pg_collection,
            ), max_retries=3, delay=2.0)
            from api.jobs import JobRegistry
            st.jobs = JobRegistry()
            st.warm_hybrid_retriever()
            logger.info("AppState 初期化完了 (collection=%s)", st.settings.pg_collection)
        except Exception as e:
            st.startup_error = f"{type(e).__name__}: {e}"
            logger.exception("AppState 初期化失敗（/api/health で degraded 応答）")

    app.state.ctx = st
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="GraphRAG API",
        description="GraphRAG（ハイブリッド検索 + ナレッジグラフ）の HTTP API。React SPA と外部連携用。",
        version="0.1.0",
        lifespan=lifespan,
    )

    # 開発時 CORS（Vite dev server 用）
    cors = os.getenv("API_CORS_ORIGINS", "")
    if cors:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in cors.split(",") if o.strip()],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    from api.routers import admin as admin_router
    from api.routers import build as build_router
    from api.routers import dictionary as dictionary_router
    from api.routers import documents as documents_router
    from api.routers import graph as graph_router
    from api.routers import qa as qa_router
    app.include_router(admin_router.router)
    app.include_router(qa_router.router)
    app.include_router(graph_router.router)
    app.include_router(documents_router.router)
    app.include_router(build_router.router)
    app.include_router(dictionary_router.router)

    # 図画像の静的配信（QA根拠カードの <img src="/figures/..."> 用）。
    # SPAキャッチオールより先にマウントすること（後だとfallbackに飲まれる）
    from graphrag_core.config import get_settings as _gs
    _figures_dir = Path(_gs().doc_parser_figures_dir)
    if not _figures_dir.is_absolute():
        _figures_dir = Path(__file__).resolve().parent.parent / _figures_dir
    _figures_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/figures", StaticFiles(directory=_figures_dir), name="figures")

    # React SPA 静的配信（frontend/dist がある場合のみ）。/api/* が優先。
    if _FRONTEND_DIST.exists():
        app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

        # index.html はキャッシュさせない（ハッシュ付きassetsは長期キャッシュ可）。
        # これが無いとブラウザが旧 index.html を掴み、再ビルド後も画面が変わらない。
        _NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}

        @app.get("/{path:path}", include_in_schema=False)
        def spa_fallback(path: str):
            candidate = _FRONTEND_DIST / path
            if path and candidate.is_file():
                headers = _NO_CACHE if candidate.name == "index.html" else None
                return FileResponse(candidate, headers=headers)
            # 存在しない assets はSPAフォールバックしない（旧キャッシュの
            # index.html が古いハッシュのJSを要求した際、HTMLをJSとして
            # 返してしまう混乱を防ぐ。404ならブラウザが再取得側に倒れる）
            if path.startswith("assets/"):
                from fastapi.responses import Response
                return Response(status_code=404)
            return FileResponse(_FRONTEND_DIST / "index.html", headers=_NO_CACHE)

    return app


app = create_app()
