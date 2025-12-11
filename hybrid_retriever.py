"""
ハイブリッド検索機能
ベクトル検索 + 日本語キーワード検索をRRFで統合
"""
from typing import List, Dict, Any, Optional
import psycopg
from japanese_text_processor import get_japanese_processor


class HybridRetriever:
    """PGVectorベースのハイブリッド検索"""

    def __init__(self, pg_conn_string: str, collection_name: str = "graphrag"):
        self.conn_string = pg_conn_string
        self.collection_name = collection_name
        self.japanese_processor = get_japanese_processor()

    def search(
        self,
        query_text: str,
        query_vector: List[float],
        k: int = 5,
        search_type: str = 'hybrid',
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        ハイブリッド検索実行

        Args:
            query_text: クエリテキスト
            query_vector: クエリのベクトル表現
            k: 返す結果数
            search_type: 'vector', 'keyword', 'hybrid'
            rrf_k: RRFパラメータ（通常60で固定）

        Returns:
            検索結果のリスト
        """
        if search_type == 'vector':
            return self._vector_search(query_vector, k)
        elif search_type == 'keyword':
            return self._keyword_search(query_text, k)
        else:  # hybrid
            return self._hybrid_search_rrf(query_text, query_vector, k, rrf_k)

    def _vector_search(self, query_vector: List[float], k: int) -> List[Dict[str, Any]]:
        """ベクトル検索のみ"""
        with psycopg.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        e.id,
                        e.document as text,
                        e.cmetadata as metadata,
                        1 - (e.embedding <=> %s::vector) as score
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                """, (query_vector, self.collection_name, query_vector, k))

                results = []
                for row in cur.fetchall():
                    results.append({
                        'id': row[0],
                        'text': row[1],
                        'metadata': row[2],
                        'score': row[3]
                    })
                return results

    def _keyword_search(self, query_text: str, k: int) -> List[Dict[str, Any]]:
        """日本語キーワード検索"""
        if not self.japanese_processor:
            return []

        # クエリをトークン化
        tokenized_query = self.japanese_processor.tokenize(query_text)
        if not tokenized_query.strip():
            return []

        tokens = tokenized_query.split()
        tsquery_str = " | ".join(tokens)  # OR検索

        with psycopg.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        e.id,
                        e.document as text,
                        e.cmetadata as metadata,
                        ts_rank(
                            to_tsvector('simple', COALESCE(e.tokenized_content, '')),
                            to_tsquery('simple', %s)
                        ) as score
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                      AND to_tsvector('simple', COALESCE(e.tokenized_content, ''))
                          @@ to_tsquery('simple', %s)
                    ORDER BY score DESC
                    LIMIT %s
                """, (tsquery_str, self.collection_name, tsquery_str, k))

                results = []
                for row in cur.fetchall():
                    results.append({
                        'id': row[0],
                        'text': row[1],
                        'metadata': row[2],
                        'score': row[3]
                    })
                return results

    def _hybrid_search_rrf(
        self,
        query_text: str,
        query_vector: List[float],
        k: int,
        rrf_k: int
    ) -> List[Dict[str, Any]]:
        """RRF方式のハイブリッド検索"""
        if not self.japanese_processor:
            # 日本語処理が使えない場合はベクトル検索のみ
            return self._vector_search(query_vector, k)

        tokenized_query = self.japanese_processor.tokenize(query_text)
        if not tokenized_query.strip():
            return self._vector_search(query_vector, k)

        tokens = tokenized_query.split()
        tsquery_str = " | ".join(tokens)

        with psycopg.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    WITH keyword_results AS (
                        SELECT
                            e.id,
                            e.document as text,
                            e.cmetadata as metadata,
                            ROW_NUMBER() OVER (ORDER BY ts_rank(
                                to_tsvector('simple', COALESCE(e.tokenized_content, '')),
                                to_tsquery('simple', %(tsquery)s)
                            ) DESC) as k_rank
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = %(collection)s
                          AND to_tsvector('simple', COALESCE(e.tokenized_content, ''))
                              @@ to_tsquery('simple', %(tsquery)s)
                        LIMIT 100
                    ),
                    vector_results AS (
                        SELECT
                            e.id,
                            e.document as text,
                            e.cmetadata as metadata,
                            ROW_NUMBER() OVER (ORDER BY e.embedding <=> %(vector)s::vector) as v_rank
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = %(collection)s
                        LIMIT 100
                    )
                    SELECT
                        COALESCE(k.id, v.id) as id,
                        COALESCE(k.text, v.text) as text,
                        COALESCE(k.metadata, v.metadata) as metadata,
                        (COALESCE(1.0 / (%(rrf_k)s + k.k_rank), 0) +
                         COALESCE(1.0 / (%(rrf_k)s + v.v_rank), 0)) as rrf_score
                    FROM keyword_results k
                    FULL OUTER JOIN vector_results v ON k.id = v.id
                    ORDER BY rrf_score DESC
                    LIMIT %(k)s
                """, {
                    'tsquery': tsquery_str,
                    'vector': query_vector,
                    'collection': self.collection_name,
                    'rrf_k': rrf_k,
                    'k': k
                })

                results = []
                for row in cur.fetchall():
                    results.append({
                        'id': row[0],
                        'text': row[1],
                        'metadata': row[2],
                        'score': row[3]
                    })
                return results
