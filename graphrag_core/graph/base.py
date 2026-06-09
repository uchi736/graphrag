"""
GraphBackend Protocol
=====================
グラフバックエンドの統一インターフェースを定義する Protocol クラス。
Neo4j ラッパー (langchain Neo4jGraph) が準拠すべき共通 API。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from langchain_neo4j.graphs.graph_document import GraphDocument


@runtime_checkable
class GraphBackend(Protocol):
    """グラフバックエンドの共通インターフェース (Protocol)

    Neo4j (langchain Neo4jGraph) が実装すべきメソッドを定義。
    runtime_checkable なので isinstance() チェックが可能。
    """

    # ------------------------------------------------------------------
    # GraphDocument 一括取り込み
    # ------------------------------------------------------------------
    def add_graph_documents(
        self,
        graph_docs: List[GraphDocument],
        include_source: bool = True,
    ) -> None:
        """GraphDocument をグラフに追加"""
        ...

    # ------------------------------------------------------------------
    # クエリ / データ取得
    # ------------------------------------------------------------------
    def query(
        self,
        query_str: str | None = None,
        params: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Cypher クエリを実行"""
        ...

    def refresh_schema(self) -> None:
        """スキーマ更新"""
        ...
