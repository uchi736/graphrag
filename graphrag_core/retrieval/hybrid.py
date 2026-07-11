"""
ハイブリッド検索機能
ベクトル検索 + BM25キーワード検索をRRFで統合
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import psycopg
from rank_bm25 import BM25Okapi
from graphrag_core.text.japanese import get_japanese_processor
from graphrag_core.db.utils import normalize_pg_connection_string, ensure_tokenized_schema
from graphrag_core.prompts import RERANK_PROMPT
from graphrag_core.llm.langfuse_utils import observe, get_langfuse_callback, update_current_span

logger = logging.getLogger(__name__)


class HybridRetriever:
    """PGVectorベースのハイブリッド検索（BM25 + ベクトル）"""

    _instances: Dict[tuple, "HybridRetriever"] = {}

    @classmethod
    def get_instance(cls, pg_conn_string: str, collection_name: str = "graphrag") -> "HybridRetriever":
        """同一 (conn, collection) のインスタンスを再利用"""
        key = (pg_conn_string, collection_name)
        if key not in cls._instances:
            cls._instances[key] = cls(pg_conn_string, collection_name)
        return cls._instances[key]

    @classmethod
    def clear_cache(cls):
        """キャッシュをクリア（テスト・再構築時用）"""
        cls._instances.clear()

    @classmethod
    def clear_instance(cls, pg_conn_string: str, collection_name: str = "graphrag"):
        """対象コレクションのインスタンスだけ破棄（増分更新/再構築後のBM25再構築用）"""
        cls._instances.pop((pg_conn_string, collection_name), None)

    def __init__(self, pg_conn_string: str, collection_name: str = "graphrag"):
        # SQLAlchemy形式 → psycopg形式に正規化
        self.conn_string = normalize_pg_connection_string(pg_conn_string)
        ensure_tokenized_schema(pg_conn_string)
        self.collection_name = collection_name
        self.japanese_processor = get_japanese_processor()

        # フルコーパス BM25 インデックス
        self._corpus_ids: List[str] = []
        self._bm25: Optional[BM25Okapi] = None
        self._build_bm25_index()

        # PGVector列の次元をキャッシュ（埋め込みプロバイダ切替時のミスマッチ早期検出用）
        self._column_dim: Optional[int] = self._fetch_column_dim()

    def _fetch_column_dim(self) -> Optional[int]:
        """langchain_pg_embedding.embedding カラムの次元を PG から取得"""
        try:
            with psycopg.connect(self.conn_string) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT atttypmod FROM pg_attribute
                        WHERE attrelid = 'langchain_pg_embedding'::regclass
                          AND attname = 'embedding'
                    """)
                    row = cur.fetchone()
                    return int(row[0]) if row and row[0] and row[0] > 0 else None
        except Exception:
            return None

    def _build_bm25_index(self):
        """コレクション全体の tokenized_content から BM25 インデックスを構築"""
        if not self.japanese_processor:
            return
        try:
            with psycopg.connect(self.conn_string) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT e.id, e.tokenized_content
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = %s
                          AND e.tokenized_content IS NOT NULL
                          AND e.tokenized_content != ''
                    """, (self.collection_name,))
                    rows = cur.fetchall()

            if not rows:
                logger.info("BM25: tokenized_content が空のためインデックスをスキップ")
                return

            self._corpus_ids = [row[0] for row in rows]
            corpus = [row[1].split() for row in rows]
            self._bm25 = BM25Okapi(corpus)
            logger.info(f"BM25: フルコーパスインデックス構築完了 ({len(rows)}件)")
        except Exception as e:
            logger.warning(f"BM25 インデックス構築エラー: {e}")

    @observe(name="hybrid_search", capture_input=False)
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
        update_current_span(input={"query_text": query_text, "k": k, "search_type": search_type})
        if search_type == 'vector':
            return self._vector_search(query_vector, k)
        elif search_type == 'keyword':
            return self._bm25_search(query_text, k)
        else:  # hybrid
            return self._hybrid_search_rrf(query_text, query_vector, k, rrf_k)

    @observe(name="vector_search", capture_input=False)
    def _vector_search(self, query_vector: List[float], k: int) -> List[Dict[str, Any]]:
        """ベクトル検索のみ"""
        if self._column_dim and len(query_vector) != self._column_dim:
            logger.error(
                "埋め込み次元不一致: query=%d, PG列=%d。"
                "EMBEDDING_PROVIDER を切り替えた場合は scripts/reset_pgvector_tables.py → scripts/build_kg.py で再構築してください。",
                len(query_vector), self._column_dim,
            )
            return []
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

    @observe(name="bm25_search")
    def _bm25_search(self, query_text: str, k: int) -> List[Dict[str, Any]]:
        """フルコーパス BM25 キーワード検索"""
        if not self.japanese_processor or not self._bm25:
            return []

        tokenized_query = self.japanese_processor.tokenize(query_text)
        if not tokenized_query.strip():
            return []

        tokens = tokenized_query.split()

        # フルコーパスでスコアリング
        scores = self._bm25.get_scores(tokens)

        # スコア > 0 の上位 k 件を取得
        top_indices = np.argsort(scores)[::-1]
        top_entries = [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0][:k]

        if not top_entries:
            return []

        # top-k の ID で DB から本文・メタデータを取得
        top_ids = [self._corpus_ids[i] for i, _ in top_entries]
        score_map = {self._corpus_ids[i]: s for i, s in top_entries}

        with psycopg.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                placeholders = ", ".join(["%s"] * len(top_ids))
                cur.execute(f"""
                    SELECT e.id, e.document, e.cmetadata
                    FROM langchain_pg_embedding e
                    WHERE e.id IN ({placeholders})
                """, top_ids)
                db_map = {row[0]: (row[1], row[2]) for row in cur.fetchall()}

        # スコア順で結果を構築
        results = []
        for doc_id in top_ids:
            if doc_id in db_map:
                text, metadata = db_map[doc_id]
                results.append({
                    'id': doc_id,
                    'text': text,
                    'metadata': metadata,
                    'score': score_map[doc_id]
                })

        return results

    @observe(name="hybrid_search_rrf", capture_input=False)
    def _hybrid_search_rrf(
        self,
        query_text: str,
        query_vector: List[float],
        k: int,
        rrf_k: int
    ) -> List[Dict[str, Any]]:
        """RRF方式のハイブリッド検索（BM25 + ベクトル）"""
        # 内部取得件数: top_k * 10（最大100件）
        internal_k = min(k * 10, 100)

        # ベクトル検索結果
        vector_results = self._vector_search(query_vector, k=internal_k)

        # BM25検索結果
        bm25_results = self._bm25_search(query_text, k=internal_k)

        # トレースinput: 両検索のチャンク概要（ベクトルではなく結果を表示）
        update_current_span(input={
            "query_text": query_text,
            "k": k,
            "vector_results_count": len(vector_results),
            "vector_results": [{"text": r["text"][:200], "score": r["score"]} for r in vector_results[:5]],
            "bm25_results_count": len(bm25_results),
            "bm25_results": [{"text": r["text"][:200], "score": r["score"]} for r in bm25_results[:5]],
        })

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


@observe(name="doc_reranking")
def rerank_with_llm(
    query: str,
    docs: List[Dict[str, Any]],
    llm,
    k: int,
    langfuse_config: dict = None,
) -> List[Dict[str, Any]]:
    """
    LLMでリランキング

    Args:
        query: 検索クエリ
        docs: 検索結果のリスト
        llm: LangChainのChatModel
        k: 返す件数
        langfuse_config: 非推奨（後方互換用、@observeで自動トレース）

    Returns:
        リランキング後の検索結果
    """
    if not docs or len(docs) <= 1:
        return docs[:k]

    # cross-encoder リランカーが使えるなら優先（LLM呼び出しより10-100倍速い）
    from graphrag_core.retrieval.reranker import is_reranker_enabled, rerank_by_score
    if is_reranker_enabled():
        # チャンクは最大1024字。500字で切ると後半に根拠がある場合に
        # リランカーから見えなくなるため、ほぼ全文の1000字まで渡す
        # （bge-reranker-v2-m3 は長文入力に対応）
        return rerank_by_score(
            query, docs,
            text_fn=lambda d: d.get('text', '')[:1000],
            top_k=k,
        )

    # フォールバック: LLM rerank
    doc_texts = []
    for i, doc in enumerate(docs):
        text = doc.get('text', '')[:500]
        doc_texts.append(f"{i+1}. {text}")

    prompt = RERANK_PROMPT.format(
        question=query,
        documents="\n\n".join(doc_texts)
    )

    try:
        response = llm.invoke(prompt, config=get_langfuse_callback())
        content = response.content if hasattr(response, 'content') else str(response)

        # "3,1,5,2,4" → [3,1,5,2,4]
        order = [int(x.strip()) for x in content.split(',') if x.strip().isdigit()]

        # 並び替え
        reranked = []
        seen = set()
        for idx in order:
            if 1 <= idx <= len(docs) and idx not in seen:
                reranked.append(docs[idx - 1])
                seen.add(idx)

        # 漏れたドキュメントを追加
        for i, doc in enumerate(docs):
            if (i + 1) not in seen:
                reranked.append(doc)

        return reranked[:k]
    except Exception as e:
        print(f"[Rerank] Error: {e}")
        return docs[:k]  # エラー時は元の順序
