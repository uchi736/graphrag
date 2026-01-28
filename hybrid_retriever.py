"""
ハイブリッド検索機能
ベクトル検索 + BM25キーワード検索をRRFで統合
"""
from typing import List, Dict, Any, Optional
import psycopg
from rank_bm25 import BM25Okapi
from japanese_text_processor import get_japanese_processor
from db_utils import normalize_pg_connection_string, ensure_tokenized_schema


class HybridRetriever:
    """PGVectorベースのハイブリッド検索（BM25 + ベクトル）"""

    def __init__(self, pg_conn_string: str, collection_name: str = "graphrag"):
        # SQLAlchemy形式 → psycopg形式に正規化
        self.conn_string = normalize_pg_connection_string(pg_conn_string)
        ensure_tokenized_schema(pg_conn_string)
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
            return self._bm25_search(query_text, k)
        else:  # hybrid
            return self._hybrid_search_rrf(query_text, query_vector, k, rrf_k)

    def _vector_search(self, query_vector: List[float], k: int) -> List[Dict[str, Any]]:
        """ベクトル検索のみ"""
        with psycopg.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        e.id as id,
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

    def _fetch_keyword_candidates(self, tsquery_str: str, limit: int = 200) -> List[Dict[str, Any]]:
        """PostgreSQL ts_rankでBM25候補を取得（プリフィルタ）"""
        with psycopg.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        e.id as id,
                        e.document as text,
                        e.cmetadata as metadata,
                        COALESCE(e.tokenized_content, '') as tokenized
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                      AND to_tsvector('simple', COALESCE(e.tokenized_content, ''))
                          @@ to_tsquery('simple', %s)
                    ORDER BY ts_rank(
                        to_tsvector('simple', COALESCE(e.tokenized_content, '')),
                        to_tsquery('simple', %s)
                    ) DESC
                    LIMIT %s
                """, (self.collection_name, tsquery_str, tsquery_str, limit))

                results = []
                for row in cur.fetchall():
                    results.append({
                        'id': row[0],
                        'text': row[1],
                        'metadata': row[2],
                        'tokenized': row[3]
                    })
                return results

    def _bm25_rerank(self, query_tokens: List[str], candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """BM25でリランキング"""
        if not candidates:
            return []

        # 各候補のトークン化済みテキストをコーパスとして使用
        corpus = [doc['tokenized'].split() for doc in candidates]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)

        # スコア付きでソート
        scored = []
        for i, doc in enumerate(candidates):
            scored.append({
                'id': doc['id'],
                'text': doc['text'],
                'metadata': doc['metadata'],
                'score': float(scores[i])
            })

        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:k]

    def _bm25_search(self, query_text: str, k: int) -> List[Dict[str, Any]]:
        """BM25キーワード検索"""
        if not self.japanese_processor:
            return []

        tokenized_query = self.japanese_processor.tokenize(query_text)
        if not tokenized_query.strip():
            return []

        tokens = tokenized_query.split()
        tsquery_str = " | ".join(tokens)

        # PostgreSQLで候補取得 → Python側でBM25リランク
        candidates = self._fetch_keyword_candidates(tsquery_str, limit=200)
        return self._bm25_rerank(tokens, candidates, k)

    def _hybrid_search_rrf(
        self,
        query_text: str,
        query_vector: List[float],
        k: int,
        rrf_k: int
    ) -> List[Dict[str, Any]]:
        """RRF方式のハイブリッド検索（BM25 + ベクトル）"""
        # ベクトル検索結果
        vector_results = self._vector_search(query_vector, k=100)

        # BM25検索結果
        bm25_results = self._bm25_search(query_text, k=100)

        if not bm25_results:
            return vector_results[:k]

        # RRFスコア計算
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(vector_results):
            doc_id = doc['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
            result_map[doc_id] = doc

        for rank, doc in enumerate(bm25_results):
            doc_id = doc['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
            if doc_id not in result_map:
                result_map[doc_id] = doc

        # スコアでソート
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for doc_id in sorted_ids[:k]:
            doc = result_map[doc_id]
            results.append({
                'id': doc['id'],
                'text': doc['text'],
                'metadata': doc['metadata'],
                'score': rrf_scores[doc_id]
            })

        return results
