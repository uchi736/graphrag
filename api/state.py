"""アプリ全体で共有するシングルトン群（Streamlit の session_state 相当）。

lifespan で1回だけ構築し app.state.ctx に載せる。
高コストな初期化（HybridRetriever の BM25 構築）は非同期ウォームアップ。

コレクションの実行時切替:
    UIで切り替えた PG_COLLECTION は .graphrag_runtime.json に永続化し、
    次回起動時も維持する（優先度: runtimeファイル > .env/環境変数）。
    .env を共有する他プログラムに影響を与えずに graphrag だけ切替できる。
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

RUNTIME_FILE = Path(__file__).resolve().parent.parent / ".graphrag_runtime.json"


def load_runtime_overrides() -> Dict[str, Any]:
    try:
        if RUNTIME_FILE.exists():
            return json.loads(RUNTIME_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("runtime override 読込失敗（無視して.env値を使用）: %s", e)
    return {}


def save_runtime_overrides(**kwargs: Any) -> None:
    data = load_runtime_overrides()
    data.update(kwargs)
    RUNTIME_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                            encoding="utf-8")


@dataclass
class AppState:
    settings: Any = None
    env_report: Dict[str, Any] = field(default_factory=dict)
    llm: Any = None                  # QA用 Chat LLM（ジョブは都度生成）
    embeddings: Any = None
    graph: Any = None                # Neo4jGraph（driverはスレッド安全）
    vector_store: Any = None         # PGVector
    jobs: Any = None                 # JobRegistry（Phase 4 で実装）
    startup_error: Optional[str] = None

    @property
    def ready(self) -> bool:
        """QA/グラフ操作を受け付けられる状態か。"""
        return (
            bool(self.env_report.get("ok"))
            and self.graph is not None
            and self.vector_store is not None
        )

    def warm_hybrid_retriever(self) -> None:
        """HybridRetriever（BM25構築＝最重量）を別スレッドでウォームアップ。

        失敗しても致命ではない（初回QAで再試行される）。
        """
        s = self.settings

        def _warm():
            try:
                from graphrag_core.retrieval.hybrid import HybridRetriever
                HybridRetriever.get_instance(s.pg_conn, s.pg_collection)
                logger.info("HybridRetriever warmed up (collection=%s)", s.pg_collection)
            except Exception as e:
                logger.warning("HybridRetriever warmup failed (retried on first QA): %s", e)

        threading.Thread(target=_warm, name="hybrid-warmup", daemon=True).start()

    def switch_collection(self, name: str) -> Dict[str, Any]:
        """検索対象コレクションを実行時に切り替える（再起動不要・永続化）。

        vector_store を作り直し、新コレクションの BM25 をウォームアップ。
        グラフは共有（出自の一致は呼び出し側/UIが provenance で確認する）。
        """
        from langchain_postgres import PGVector
        from graphrag_core.db.utils import add_connection_timeout, retry_on_timeout

        s = self.settings
        old = s.pg_collection
        pg = add_connection_timeout(s.pg_conn, timeout=30)
        self.vector_store = retry_on_timeout(lambda: PGVector(
            connection=pg,
            embeddings=self.embeddings,
            collection_name=name,
        ), max_retries=3, delay=2.0)
        s.pg_collection = name
        save_runtime_overrides(pg_collection=name)
        self.warm_hybrid_retriever()
        logger.info("collection switched: %s -> %s (persisted)", old, name)
        return {"old": old, "new": name}

    def invalidate_retrieval(self) -> None:
        """コーパス変更後（build/クリア/増分更新）の検索キャッシュ無効化＋再ウォーム。"""
        s = self.settings
        try:
            from graphrag_core.retrieval.hybrid import HybridRetriever
            if hasattr(HybridRetriever, "clear_instance"):
                HybridRetriever.clear_instance(s.pg_conn, s.pg_collection)
            else:  # 後方互換（clear_instance 未導入時は全消し）
                HybridRetriever.clear_cache()
        except Exception as e:
            logger.warning("invalidate_retrieval failed: %s", e)
        self.warm_hybrid_retriever()
