"""
entity_vectorizer.py
====================
エンティティのベクトル化と検索を管理するモジュール
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import hashlib

logger = logging.getLogger(__name__)
load_dotenv()


class EntityVectorizer:
    """エンティティのベクトル化と検索を管理"""

    def __init__(self, connection_string: str, embeddings: Optional[Any] = None):
        """
        初期化

        Args:
            connection_string: PostgreSQL接続文字列
            embeddings: 埋め込みモデル（省略時はAzureOpenAIEmbeddings使用）
        """
        self.connection_string = connection_string
        self.collection_name = os.getenv("PG_COLLECTION", "graphrag") + "_entities"

        if embeddings is None:
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
        else:
            self.embeddings = embeddings

        self.vector_store = None
        self._initialize_store()

    def _initialize_store(self):
        """ベクトルストアを初期化（既存コレクションがあれば接続、なければ後で作成）"""
        try:
            # 既存コレクションへの接続を試みる（削除しない）
            self.vector_store = PGVector(
                embeddings=self.embeddings,
                connection=self.connection_string,
                collection_name=self.collection_name,
                use_jsonb=True
            )
            logger.info(f"Connected to existing entity vector store: {self.collection_name}")
        except Exception as e:
            # コレクションが存在しない場合は初回追加時に作成
            logger.warning(f"Entity vector store not found, will create on first add: {e}")
            self.vector_store = None

    def add_entities(self, entities: List[Dict[str, Any]], graph_docs: Optional[List[Any]] = None) -> int:
        """
        エンティティをベクトル化して保存

        Args:
            entities: エンティティのリスト [{id, type, properties, related_text}]
            graph_docs: GraphDocumentリスト（コンテキスト抽出用）

        Returns:
            保存されたエンティティ数
        """
        if not entities:
            logger.warning("No entities to add")
            return 0

        logger.info(f"Adding {len(entities)} entities to vector store")
        if len(entities) > 0:
            logger.info(f"Sample entities: {[e.get('id', '') for e in entities[:5]]}")

        documents = []
        ids = []

        for entity in entities:
            # エンティティIDとテキストを生成
            entity_id = entity.get('id', '')
            entity_type = entity.get('type', 'Term')
            properties = entity.get('properties', {})

            # コンテキストテキストの構築
            # エンティティ名、タイプ、プロパティ、関連テキストを結合
            text_parts = [
                f"Entity: {entity_id}",
                f"Type: {entity_type}"
            ]

            # プロパティを追加
            for key, value in properties.items():
                if value and key != 'id':  # idは既に含まれているのでスキップ
                    text_parts.append(f"{key}: {value}")

            # 関連テキストがあれば追加
            related_text = entity.get('related_text', '')
            if related_text:
                text_parts.append(f"Context: {related_text}")

            # ドキュメント作成
            content = "\n".join(text_parts)
            doc = Document(
                page_content=content,
                metadata={
                    'entity_id': entity_id,
                    'entity_type': entity_type,
                    'properties': properties
                }
            )
            documents.append(doc)

            # 一意なIDを生成（entity_id + entity_typeのハッシュ）
            unique_id = hashlib.md5(f"{entity_id}_{entity_type}".encode()).hexdigest()
            ids.append(unique_id)

        logger.info(f"Created {len(documents)} documents for vectorization")

        # エンティティコレクションのみを再作成
        # pre_delete_collection=True は collection_name で指定したコレクションのみを削除
        # （graphragコレクションには影響しない）
        logger.info(f"Creating/updating entity vector collection: {self.collection_name}")

        # バッチ分割（PostgreSQLのbindパラメータ上限対策）
        BATCH_SIZE = 500
        total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
        saved_count = 0
        error_count = 0

        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            logger.info(f"Batch {batch_num}/{total_batches}: {len(batch)} entities")

            try:
                self.vector_store = PGVector.from_documents(
                    batch,
                    self.embeddings,
                    connection=self.connection_string,
                    collection_name=self.collection_name,  # "graphrag_entities"
                    pre_delete_collection=(i == 0),  # 初回バッチのみ削除・再作成
                    use_jsonb=True
                )
                saved_count += len(batch)
            except Exception as e:
                error_count += len(batch)
                logger.error(f"Batch {batch_num}/{total_batches} failed: {e}")
                if 'column "id" of relation "langchain_pg_embedding" does not exist' in str(e):
                    logger.error(
                        "The existing langchain_pg_embedding table uses an old schema (missing id column). "
                        "Drop the langchain_pg_embedding and langchain_pg_collection tables so "
                        "langchain-postgres can recreate them with the current schema."
                    )

        if saved_count > 0:
            logger.info(f"Successfully added {saved_count} entities to vector store '{self.collection_name}'"
                        + (f" ({error_count} failed)" if error_count else ""))
        else:
            logger.error(f"All {error_count} entities failed to save")
        return saved_count

    def search_similar_entities(
        self,
        query: str,
        k: int = 10,
        score_threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        クエリに類似したエンティティを検索

        Args:
            query: 検索クエリ
            k: 取得する結果数
            score_threshold: 類似度の閾値

        Returns:
            [(entity_id, score)] のリスト
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []

        try:
            logger.info(f"Searching entities with query: {query[:100]}...")

            # ベクトル検索
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k * 2  # より多く取得してからフィルタリング
            )

            logger.info(f"Raw search returned {len(results)} results")
            if results:
                logger.info(f"Sample scores: {[score for _, score in results[:3]]}")

            # 閾値以上のスコアのみフィルタリング
            filtered_results = []
            for doc, score in results:
                # PGVectorはコサイン距離を返す（0に近いほど類似）
                # 距離が小さいほど類似度が高い
                # 閾値判定を距離ベースで行う
                distance_threshold = 1.0 - score_threshold  # 類似度閾値を距離に変換

                if score <= distance_threshold:  # 距離が閾値以下なら採用
                    entity_id = doc.metadata.get('entity_id', '')
                    if entity_id:
                        # 表示用の類似度スコア（0-1、1に近いほど類似）
                        similarity = max(0.0, 1.0 - score)
                        filtered_results.append((entity_id, similarity))
                        logger.debug(f"Found entity: {entity_id} (distance: {score:.3f}, similarity: {similarity:.3f})")

            logger.info(f"Filtered to {len(filtered_results)} results with threshold {score_threshold}")

            # 結果が少ない場合、閾値を緩めて再検索
            if len(filtered_results) < 3 and score_threshold > 0.5:
                logger.info("Too few results, trying with relaxed threshold")
                return self.search_similar_entities(query, k, score_threshold * 0.8)

            return filtered_results

        except Exception as e:
            logger.error(f"Failed to search entities: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def search_by_embedding(
        self,
        embedding: List[float],
        k: int = 10,
        score_threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        埋め込みベクトルで類似エンティティを検索

        Args:
            embedding: 検索用埋め込みベクトル
            k: 取得する結果数
            score_threshold: 類似度の閾値

        Returns:
            [(entity_id, score)] のリスト
        """
        if not self.vector_store:
            return []

        try:
            # ベクトル検索
            results = self.vector_store.similarity_search_by_vector_with_score(
                embedding,
                k=k
            )

            # 閾値以上のスコアのみフィルタリング
            filtered_results = []
            for doc, score in results:
                # PGVectorのスコアは距離なので、類似度に変換
                similarity = 1.0 - score
                if similarity >= score_threshold:
                    entity_id = doc.metadata.get('entity_id', '')
                    if entity_id:
                        filtered_results.append((entity_id, similarity))

            return filtered_results

        except Exception as e:
            logger.error(f"Failed to search entities by embedding: {e}")
            return []

    def search_keyword_entities(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        キーワード検索でエンティティを検索（部分一致）

        Args:
            query: 検索クエリ
            k: 取得する結果数

        Returns:
            [(entity_id, score)] のリスト
        """
        try:
            import psycopg
            from db_utils import normalize_pg_connection_string

            conn_string = normalize_pg_connection_string(self.connection_string)

            # 日本語トークナイザーがあれば使用
            try:
                from japanese_text_processor import get_japanese_processor
                processor = get_japanese_processor()
                if processor:
                    tokens = processor.tokenize(query).split()
                else:
                    tokens = query.split()
            except ImportError:
                tokens = query.split()

            if not tokens:
                return []

            results = []
            with psycopg.connect(conn_string) as conn:
                with conn.cursor() as cur:
                    # エンティティコレクションのUUIDを取得
                    cur.execute("""
                        SELECT uuid FROM langchain_pg_collection
                        WHERE name = %s
                    """, (self.collection_name,))
                    row = cur.fetchone()
                    if not row:
                        logger.warning(f"Collection {self.collection_name} not found")
                        return []
                    collection_uuid = row[0]

                    # 各トークンでエンティティ名を部分一致検索
                    # cmetadata->>'entity_id' にエンティティ名が格納されている
                    for token in tokens:
                        if len(token) < 2:  # 1文字は除外
                            continue

                        cur.execute("""
                            SELECT DISTINCT cmetadata->>'entity_id' as entity_id
                            FROM langchain_pg_embedding
                            WHERE collection_id = %s
                              AND (
                                  cmetadata->>'entity_id' ILIKE %s
                                  OR document ILIKE %s
                              )
                            LIMIT %s
                        """, (collection_uuid, f'%{token}%', f'%{token}%', k))

                        for row in cur.fetchall():
                            entity_id = row[0]
                            if entity_id:
                                # キーワード一致は高スコア（1.0）
                                results.append((entity_id, 1.0))

            # 重複除去して返す
            seen = set()
            unique_results = []
            for entity_id, score in results:
                if entity_id not in seen:
                    seen.add(entity_id)
                    unique_results.append((entity_id, score))
                    if len(unique_results) >= k:
                        break

            logger.info(f"Keyword search found {len(unique_results)} entities")
            return unique_results

        except Exception as e:
            logger.error(f"Failed to search entities by keyword: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def search_hybrid_entities(
        self,
        query: str,
        k: int = 10,
        score_threshold: float = 0.7,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        search_type: str = "hybrid"
    ) -> List[Tuple[str, float]]:
        """
        ハイブリッド検索（ベクトル + キーワード）でエンティティを検索

        Args:
            query: 検索クエリ
            k: 取得する結果数
            score_threshold: ベクトル検索の類似度閾値
            vector_weight: ベクトル検索の重み（RRF用）
            keyword_weight: キーワード検索の重み（RRF用）
            search_type: "hybrid", "vector", "keyword"

        Returns:
            [(entity_id, score)] のリスト
        """
        logger.info(f"Hybrid entity search: type={search_type}, query={query[:50]}...")

        vector_results = []
        keyword_results = []

        # ベクトル検索
        if search_type in ("hybrid", "vector"):
            vector_results = self.search_similar_entities(
                query,
                k=k * 2,  # 多めに取得
                score_threshold=score_threshold
            )
            logger.info(f"Vector search found {len(vector_results)} entities")

        # キーワード検索
        if search_type in ("hybrid", "keyword"):
            keyword_results = self.search_keyword_entities(query, k=k * 2)
            logger.info(f"Keyword search found {len(keyword_results)} entities")

        # 単一モードの場合はそのまま返す
        if search_type == "vector":
            return vector_results[:k]
        if search_type == "keyword":
            return keyword_results[:k]

        # RRF（Reciprocal Rank Fusion）でスコアを統合
        rrf_scores = {}
        rrf_k = 60  # RRFの定数

        # ベクトル検索結果のRRFスコア
        for rank, (entity_id, score) in enumerate(vector_results):
            rrf_score = vector_weight * (1.0 / (rrf_k + rank + 1))
            rrf_scores[entity_id] = rrf_scores.get(entity_id, 0) + rrf_score

        # キーワード検索結果のRRFスコア
        for rank, (entity_id, score) in enumerate(keyword_results):
            rrf_score = keyword_weight * (1.0 / (rrf_k + rank + 1))
            rrf_scores[entity_id] = rrf_scores.get(entity_id, 0) + rrf_score

        # スコアでソートして返す
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        final_results = [(entity_id, score) for entity_id, score in sorted_results[:k]]

        logger.info(f"Hybrid search final: {len(final_results)} entities")
        if final_results:
            logger.info(f"Top entities: {[e[0] for e in final_results[:3]]}")

        return final_results

    def extract_entities_from_graph(self, graph, graph_backend: str = "neo4j") -> List[Dict[str, Any]]:
        """
        グラフからエンティティを抽出してコンテキストを付加

        Args:
            graph: グラフオブジェクト
            graph_backend: グラフバックエンド ("neo4j" or "networkx")

        Returns:
            エンティティリスト
        """
        entities = []

        try:
            if graph_backend == "neo4j":
                # Neo4jからエンティティと関連チャンクを取得
                # Term または CSVNode ラベルを持つノードを対象とする
                query = """
                MATCH (n)
                WHERE (n:Term OR n:CSVNode) AND n.id IS NOT NULL
                OPTIONAL MATCH (n)<-[:MENTIONS]-(chunk:Chunk)
                WITH n, COLLECT(DISTINCT substring(chunk.text, 0, 500)) AS chunk_texts
                RETURN n.id AS entity_id,
                       labels(n) AS types,
                       properties(n) AS props,
                       chunk_texts
                LIMIT 1000
                """

                logger.info("Extracting entities from Neo4j graph")
                results = graph.query(query)
                logger.info(f"Neo4j query returned {len(results)} entities")

                for result in results:
                    entity_id = result.get('entity_id', '')
                    types = result.get('types', ['Term'])
                    props = result.get('props', {})
                    chunk_texts = result.get('chunk_texts', [])

                    # 関連テキストを結合（最大5つのチャンク）
                    related_text = ' '.join(chunk_texts[:5]) if chunk_texts else ''

                    entities.append({
                        'id': entity_id,
                        'type': types[0] if types else 'Term',
                        'properties': props,
                        'related_text': related_text[:1000]  # 最大1000文字
                    })

            else:  # networkx
                # NetworkXグラフからエンティティを抽出
                import networkx as nx

                if hasattr(graph, 'graph'):
                    nx_graph = graph.graph
                    node_metadata = getattr(graph, 'node_metadata', {})
                else:
                    nx_graph = graph
                    node_metadata = {}

                for node_id in nx_graph.nodes():
                    # Chunkノードは除外（32文字の16進数ハッシュ）
                    if not node_id:
                        continue
                    # SHA256ハッシュ（64文字）またはMD5ハッシュ（32文字）のChunk IDを除外
                    if len(node_id) == 32 or len(node_id) == 64:
                        try:
                            int(node_id, 16)  # 16進数かチェック
                            continue  # ハッシュIDなのでスキップ
                        except ValueError:
                            pass  # 16進数でなければ通常のノード

                    # node_metadataからタイプを取得（NetworkXGraph用）
                    meta = node_metadata.get(node_id, {})
                    entity_type = meta.get('type', 'Term')
                    properties = meta.get('properties', {})

                    # 関連するエッジからコンテキストを取得
                    successors = list(nx_graph.successors(node_id))
                    predecessors = list(nx_graph.predecessors(node_id))
                    related_nodes = list(dict.fromkeys(successors + predecessors))[:5]
                    # 関連ノードもハッシュIDを除外
                    related_nodes = [n for n in related_nodes if len(n) != 32 and len(n) != 64]
                    related_text = f"Connected to: {', '.join(related_nodes)}" if related_nodes else ''

                    entities.append({
                        'id': node_id,
                        'type': entity_type,
                        'properties': properties,
                        'related_text': related_text
                    })

            logger.info(f"Extracted {len(entities)} entities from graph")

        except Exception as e:
            logger.error(f"Failed to extract entities from graph: {e}")

        return entities

    def clear_entity_vectors(self) -> bool:
        """
        エンティティベクトルをクリア

        Returns:
            成功時True
        """
        try:
            # コレクションを削除して再作成
            self.vector_store = PGVector.from_documents(
                [],
                self.embeddings,
                connection=self.connection_string,
                collection_name=self.collection_name,
                pre_delete_collection=True,
                use_jsonb=True
            )
            logger.info("Entity vectors cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear entity vectors: {e}")
            return False
