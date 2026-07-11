"""登録ドキュメント関連サービス。

- list_registered_documents: ソース別チャンク集計（ui/documents_tab.py の SQL を関数化）
- build_add_chunk_fn: 1チャンクのLLM抽出→グラフ書込クロージャ
  （scripts/update_doc.py から移設。増分更新ジョブ/CLI 共用）
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Dict, List

from graphrag_core.config import get_settings
from graphrag_core.db.utils import normalize_pg_connection_string


def list_document_chunks(pg_conn: str, pg_collection: str, source: str,
                         limit: int = 50, offset: int = 0,
                         focus_id: str | None = None) -> Dict[str, Any]:
    """指定ソース文書のチャンク本文を取得する（登録ドキュメント/QA根拠のチャンク閲覧用）。

    focus_id を渡すと、そのチャンクが含まれるページに offset を自動調整して返す
    （QAの参照ドキュメントから「文書内で見る」を開いたとき用）。
    """
    import psycopg
    raw_conn = normalize_pg_connection_string(pg_conn)
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    with psycopg.connect(raw_conn) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*)
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s AND COALESCE(e.cmetadata->>'source','(unknown)') = %s
            """, (pg_collection, source))
            total = cur.fetchone()[0]

            # focus_id 指定時: そのチャンクの順位からページ先頭 offset を計算
            if focus_id:
                cur.execute("""
                    SELECT rn - 1 FROM (
                        SELECT COALESCE(e.cmetadata->>'id', e.id) AS chunk_id,
                               ROW_NUMBER() OVER (ORDER BY COALESCE(e.cmetadata->>'id', e.id)) AS rn
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = %s AND COALESCE(e.cmetadata->>'source','(unknown)') = %s
                    ) t WHERE chunk_id = %s
                """, (pg_collection, source, focus_id))
                row = cur.fetchone()
                if row is not None:
                    offset = (row[0] // limit) * limit
            cur.execute("""
                SELECT COALESCE(e.cmetadata->>'id', e.id) AS chunk_id,
                       e.cmetadata->>'page' AS page,
                       e.document
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s AND COALESCE(e.cmetadata->>'source','(unknown)') = %s
                ORDER BY chunk_id
                LIMIT %s OFFSET %s
            """, (pg_collection, source, limit, offset))
            rows = cur.fetchall()
    return {
        "source": source,
        "total": total,
        "offset": offset,
        "chunks": [{"id": r[0], "page": r[1], "text": r[2]} for r in rows],
    }


def list_registered_documents(pg_conn: str, pg_collection: str) -> Dict[str, Any]:
    """コレクション内のソース別チャンク数を集計する。"""
    import psycopg
    raw_conn = normalize_pg_connection_string(pg_conn)
    with psycopg.connect(raw_conn) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COALESCE(e.cmetadata->>'source', '(unknown)') as source,
                    COUNT(*) as chunk_count
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                GROUP BY e.cmetadata->>'source'
                ORDER BY chunk_count DESC
            """, (pg_collection,))
            rows = cur.fetchall()
    documents = [{"source": r[0], "chunk_count": r[1]} for r in rows]
    return {
        "collection": pg_collection,
        "total_chunks": sum(d["chunk_count"] for d in documents),
        "documents": documents,
    }


# ── 増分更新用の抽出クロージャ（update_doc.py:89 から移設） ────────────
def build_add_chunk_fn(graph, llm) -> Callable:
    """1チャンクを LLM 抽出してグラフに追加する副作用関数を返す。

    incremental.update_document に DI する（build系ビルダーと同じ流儀:
    エンティティID正規化 + attach_source_chunks + ProcessedChunk + Document.source）。
    """
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from graphrag_core.graph.schema import (
        entity_naming_instructions, get_allowed_node_types, get_allowed_relations)
    from graphrag_core.graph.enrichment import attach_source_chunks

    try:
        import neologdn
        _has_neo = True
    except Exception:
        neologdn = None
        _has_neo = False

    def _norm(name):
        if not name:
            return name
        s = unicodedata.normalize("NFKC", str(name))
        if _has_neo:
            s = neologdn.normalize(s)
        s = re.sub(r"\s+", "", s).strip()
        return s or str(name)

    s = get_settings()
    is_vllm = s.llm_provider.lower() == "vllm"
    additional = (
        "抽出する: 技術用語、概念、固有名詞、プロセス名、規格名、組織、製品。"
        "抽出しない: 一般的な名詞（『こと』『もの』『方法』）、代名詞、動詞。"
        "抽出しない: 数値・日付・単位のみの値。値はノードにしない。"
        "RELATED_TOは他に適切な関係がない場合の最終手段として使用。"
        + entity_naming_instructions()
    )
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=get_allowed_node_types(),
        allowed_relationships=get_allowed_relations(),
        strict_mode=False,
        ignore_tool_usage=is_vllm,
        additional_instructions=additional,
    )

    def add_chunk(chunk):
        chunk_docs = transformer.convert_to_graph_documents([chunk])
        for gd in chunk_docs:
            for node in gd.nodes:
                node.id = _norm(node.id)
            for rel in gd.relationships:
                rel.source.id = _norm(rel.source.id)
                rel.target.id = _norm(rel.target.id)
        graph.add_graph_documents(chunk_docs, include_source=True)
        cid = chunk.metadata["id"]
        attach_source_chunks(graph, chunk_docs, cid)
        graph.query(
            "MERGE (c:ProcessedChunk {hash: $h}) SET c.processed_at = datetime()", {"h": cid})
        from graphrag_core.graph.schema import chunk_label
        graph.query(
            "MATCH (d:" + chunk_label() + " {id: $id}) SET d.source = $src, d.page = $page",
            {"id": cid, "src": chunk.metadata.get("source"),
             "page": chunk.metadata.get("page")})
        return cid

    return add_chunk
