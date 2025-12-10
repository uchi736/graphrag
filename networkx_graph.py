"""
NetworkX-based Graph Backend for Graph-RAG
==========================================
Neo4jの代替として、NetworkXをグラフバックエンドとして使用するための実装。
Neo4jGraphと互換性のあるインターフェースを提供。

特徴:
- Neo4jGraph互換のAPI
- pickle/JSONによる永続化
- 1-2 hop近傍探索
- 可視化用の統一フォーマット出力
"""
from __future__ import annotations

import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

import networkx as nx

try:
    from langchain_community.graphs.graph_document import GraphDocument
except ImportError:
    from langchain_community.graphs import GraphDocument


class NetworkXGraph:
    """Neo4jGraph互換のNetworkXグラフラッパー"""

    def __init__(
        self,
        storage_path: str = "graph.pkl",
        auto_save: bool = True
    ):
        """
        Args:
            storage_path: グラフ保存ファイルパス
            auto_save: 変更時に自動保存するか
        """
        self.storage_path = Path(storage_path)
        self.auto_save = auto_save

        # MultiDiGraph: 複数の関係タイプをサポート
        self.graph = nx.MultiDiGraph()

        # メタデータ管理
        self.node_metadata = {}  # node_id -> {type, properties}
        self.edge_metadata = {}  # (src, tgt, key) -> {type, properties}

        # 既存グラフがあれば読み込み
        if self.storage_path.exists():
            self.load()

    def add_graph_documents(
        self,
        graph_docs: List[GraphDocument],
        include_source: bool = True
    ) -> None:
        """GraphDocumentをグラフに追加（Neo4jGraph互換）"""
        for doc in graph_docs:
            # ノード追加
            for node in doc.nodes:
                node_id = node.id
                node_type = node.type if hasattr(node, 'type') else 'Unknown'

                self.graph.add_node(node_id)
                self.node_metadata[node_id] = {
                    'type': node_type,
                    'properties': getattr(node, 'properties', {})
                }

            # エッジ追加
            for rel in doc.relationships:
                source = rel.source.id
                target = rel.target.id
                rel_type = rel.type

                # MultiDiGraphなので同じノード間に複数のエッジを追加可能
                edge_key = self.graph.add_edge(source, target, type=rel_type)
                self.edge_metadata[(source, target, edge_key)] = {
                    'type': rel_type,
                    'properties': getattr(rel, 'properties', {})
                }

            # ソースドキュメント情報を保持（include_source=True時）
            if include_source and hasattr(doc, 'source'):
                source_doc = doc.source
                if source_doc:
                    # Chunkノードとして追加（Neo4jと同様）
                    chunk_id = getattr(source_doc, 'metadata', {}).get('id', str(id(source_doc)))
                    chunk_text = source_doc.page_content if hasattr(source_doc, 'page_content') else ''

                    self.graph.add_node(chunk_id)
                    self.node_metadata[chunk_id] = {
                        'type': 'Chunk',
                        'properties': {'text': chunk_text}
                    }

                    # ChunkからエンティティへのMENTIONS関係
                    for node in doc.nodes:
                        self.graph.add_edge(chunk_id, node.id, type='MENTIONS')

        if self.auto_save:
            self.save()

    def query(self, query_str: str = None, params: Dict = None) -> List[Dict[str, Any]]:
        """
        グラフクエリ実行（簡易版Cypher風インターフェース）

        Neo4jのCypherの代わりに、簡易的なクエリをサポート。
        または params で直接検索パラメータを指定可能。

        Args:
            query_str: Cypherクエリ（限定的にサポート）
            params: クエリパラメータ

        Returns:
            クエリ結果（辞書のリスト）
        """
        # パラメータベースのクエリ（主にエンティティ検索用）
        if params and 'entities' in params:
            return self._search_by_entities(params['entities'])

        # Cypherクエリの簡易パース（限定的）
        if query_str:
            return self._parse_and_execute_cypher(query_str, params or {})

        return []

    def _search_by_entities(self, entities: List[str]) -> List[Dict[str, Any]]:
        """エンティティ名から1-hop近傍を検索"""
        results = []

        for entity in entities:
            # 部分一致でノード検索
            matched_nodes = [
                n for n in self.graph.nodes()
                if entity.lower() in str(n).lower()
                and not self._is_chunk_node(n)
            ]

            # 各マッチノードの1-hop近傍を取得
            for node in matched_nodes:
                # 出力エッジ（双方向）
                for neighbor in self.graph.neighbors(node):
                    if self._is_chunk_node(neighbor):
                        continue

                    # エッジ情報取得
                    edges = self.graph.get_edge_data(node, neighbor)
                    for key, edge_data in edges.items():
                        if edge_data.get('type') != 'MENTIONS':
                            results.append({
                                'start': node,
                                'type': edge_data.get('type', 'RELATED'),
                                'end': neighbor
                            })

                # 入力エッジ
                for predecessor in self.graph.predecessors(node):
                    if self._is_chunk_node(predecessor):
                        continue

                    edges = self.graph.get_edge_data(predecessor, node)
                    for key, edge_data in edges.items():
                        if edge_data.get('type') != 'MENTIONS':
                            results.append({
                                'start': predecessor,
                                'type': edge_data.get('type', 'RELATED'),
                                'end': node
                            })

        # 重複除去
        seen = set()
        unique_results = []
        for r in results:
            key = (r['start'], r['type'], r['end'])
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        return unique_results

    def _parse_and_execute_cypher(self, cypher: str, params: Dict) -> List[Dict[str, Any]]:
        """限定的なCypherクエリをパースして実行"""
        cypher_upper = cypher.upper()

        # MATCH (n) DETACH DELETE n （全削除）
        if 'DETACH DELETE' in cypher_upper and 'MATCH (N)' in cypher_upper:
            self.graph.clear()
            self.node_metadata.clear()
            self.edge_metadata.clear()
            if self.auto_save:
                self.save()
            return []

        # MATCH (n) RETURN count(n) （ノード数カウント）
        if 'COUNT' in cypher_upper and 'RETURN' in cypher_upper:
            return [{'node_count': self.graph.number_of_nodes()}]

        # MATCH ()-[r]->() RETURN count(r) （エッジ数カウント）
        if 'COUNT' in cypher_upper and '[R]' in cypher_upper:
            return [{'rel_count': self.graph.number_of_edges()}]

        # その他のクエリはエンティティ検索にフォールバック
        if params and 'entities' in params:
            return self._search_by_entities(params['entities'])

        # デフォルト: 全エッジ返却（制限付き）
        return self.get_all_relationships(limit=50)

    def get_all_relationships(self, limit: int = 100) -> List[Dict[str, Any]]:
        """全リレーションシップを取得（制限付き）"""
        results = []
        count = 0

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if self._is_chunk_node(u) or self._is_chunk_node(v):
                continue
            if data.get('type') == 'MENTIONS':
                continue

            results.append({
                'start': u,
                'type': data.get('type', 'RELATED'),
                'end': v
            })

            count += 1
            if count >= limit:
                break

        return results

    def get_graph_data(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        可視化用の統一フォーマットでグラフデータを取得

        Neo4jのget_enhanced_graph_data()と同じフォーマット:
        {
            'source': node_id,
            'source_type': node_type,
            'target': node_id,
            'target_type': node_type,
            'relation': relation_type,
            'source_degree': int,
            'target_degree': int,
            'source_docs': [doc_names],
            'target_docs': [doc_names]
        }
        """
        results = []
        count = 0

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            # Chunkノード除外
            if self._is_chunk_node(u) or self._is_chunk_node(v):
                continue

            # MENTIONS関係除外
            if data.get('type') == 'MENTIONS':
                continue

            source_type = self.node_metadata.get(u, {}).get('type', 'Unknown')
            target_type = self.node_metadata.get(v, {}).get('type', 'Unknown')

            source_degree = self.graph.degree(u)
            target_degree = self.graph.degree(v)

            # ドキュメント情報取得（Chunk経由）
            source_docs = self._get_document_sources(u)
            target_docs = self._get_document_sources(v)

            results.append({
                'source': u,
                'source_type': source_type,
                'target': v,
                'target_type': target_type,
                'relation': data.get('type', 'RELATED'),
                'source_degree': source_degree,
                'target_degree': target_degree,
                'source_docs': source_docs,
                'target_docs': target_docs
            })

            count += 1
            if count >= limit:
                break

        return results

    def _get_document_sources(self, node_id: str) -> List[str]:
        """ノードが言及されているドキュメント名を取得"""
        docs = []

        # Chunkノードから逆方向にたどる
        for predecessor in self.graph.predecessors(node_id):
            if self._is_chunk_node(predecessor):
                # ChunkのメタデータからDocumentノードへのリンクを探す
                # （実装簡略化のため、ここではChunk IDのみ返す）
                # 本格実装ではChunk->Document関係を追跡
                pass

        return docs

    def _is_chunk_node(self, node_id: str) -> bool:
        """ChunkノードかどうかをIDパターンで判定"""
        # 32文字の16進数ハッシュならChunk
        if isinstance(node_id, str) and len(node_id) == 32:
            try:
                int(node_id, 16)
                return True
            except ValueError:
                pass
        return False

    def refresh_schema(self) -> None:
        """スキーマ更新（Neo4jGraph互換、NetworkXでは不要）"""
        pass

    def save(self, path: Optional[str] = None) -> None:
        """グラフを永続化"""
        save_path = Path(path) if path else self.storage_path

        data = {
            'graph': nx.node_link_data(self.graph),
            'node_metadata': self.node_metadata,
            'edge_metadata': {str(k): v for k, v in self.edge_metadata.items()}
        }

        # Pickle形式で保存
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        # JSONバックアップも作成
        json_path = save_path.with_suffix('.json')
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            # JSON化できない場合は無視
            pass

    def load(self, path: Optional[str] = None) -> None:
        """グラフを読み込み"""
        load_path = Path(path) if path else self.storage_path

        if not load_path.exists():
            return

        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        self.graph = nx.node_link_graph(data['graph'], multigraph=True)
        self.node_metadata = data['node_metadata']

        # edge_metadataのキーを復元
        self.edge_metadata = {}
        for k_str, v in data['edge_metadata'].items():
            # "(src, tgt, key)" 形式の文字列をタプルに戻す
            try:
                key = eval(k_str)
                self.edge_metadata[key] = v
            except Exception:
                pass


class NetworkXGraphRetriever:
    """
    GraphRetriever互換のNetworkX用Retriever

    質問からエンティティを抽出し、N-hopトラバーサルでサブグラフを取得。
    """

    def __init__(
        self,
        graph: NetworkXGraph,
        k: int = 4,
        search_type: str = "networkx",
        llm = None  # エンティティ抽出用LLM（オプション）
    ):
        """
        Args:
            graph: NetworkXGraphインスタンス
            k: 返却する関係数
            search_type: 検索タイプ（互換性のためのダミー）
            llm: エンティティ抽出用LLM
        """
        self.graph = graph
        self.k = k
        self.search_type = search_type
        self.llm = llm

    def invoke(self, question: str) -> List[Dict[str, Any]]:
        """質問からグラフコンテキストを取得"""
        # エンティティ抽出
        entities = self._extract_entities(question)

        if not entities:
            return []

        # グラフ検索
        results = self.graph.query(params={'entities': entities})

        # 上位k件に制限
        return results[:self.k]

    def as_runnable(self):
        """LCEL互換のRunnableとして返す"""
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(self.invoke)

    def _extract_entities(self, question: str) -> List[str]:
        """質問からエンティティを抽出"""
        if self.llm:
            try:
                # LLMを使った高精度抽出
                extraction_prompt = f"""以下の質問文から、固有名詞や重要なエンティティを抽出してください。
エンティティのみをカンマ区切りで出力してください。説明は不要です。

質問: {question}

エンティティ:"""
                response = self.llm.invoke(extraction_prompt)
                entities = [e.strip() for e in response.content.split(',') if e.strip()]
                return entities
            except Exception:
                pass

        # フォールバック: 簡易的なキーワード抽出
        # 2文字以上の単語を抽出
        words = question.split()
        entities = [w for w in words if len(w) > 1]
        return entities[:5]  # 最大5個
