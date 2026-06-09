"""KGプロパティ後付けユーティリティ

Neo4jはプロパティグラフだが、LangChainの `add_graph_documents` は
ノード/エッジに `id`/type 以外のプロパティを書かない。
本モジュールはビルド中・ビルド後にプロパティを後付けする:

- attach_source_chunks(): edgeに source_chunks (抽出根拠チャンクID) を貯める
- compute_mention_count(): Term.mention_count を集計
- compute_pagerank(): Term.pagerank を NetworkX で計算
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def attach_source_chunks(graph, chunk_docs: Iterable[Any], chunk_hash: str) -> int:
    """1チャンクから抽出された GraphDocument 群について、その relationships
    の各エッジに `source_chunks` (チャンクhash の配列) を後付けする。

    並列ワーカーで複数チャンクが同一エッジに到達する可能性があるため、
    既存配列に同hash が無ければ append する冪等更新で書く。

    Args:
        graph: langchain Neo4jGraph
        chunk_docs: transformer.convert_to_graph_documents([chunk]) の戻り値
        chunk_hash: チャンクのSHA256 (metadata['id'])

    Returns:
        書き込んだ relationship 件数
    """
    if not chunk_hash:
        return 0

    count = 0
    for gd in chunk_docs:
        for rel in getattr(gd, "relationships", []) or []:
            rel_type = getattr(rel, "type", None)
            src_id = getattr(getattr(rel, "source", None), "id", None)
            tgt_id = getattr(getattr(rel, "target", None), "id", None)
            if not (rel_type and src_id and tgt_id):
                continue
            # rel_type を Cypher に埋め込むので識別子安全化（英大文字+数字+_のみ許可）
            safe_type = "".join(c for c in str(rel_type) if c.isalnum() or c == "_")
            if not safe_type:
                continue
            try:
                graph.query(
                    f"""
                    MATCH (a:Term {{id: $start}})
                    MATCH (b:Term {{id: $end}})
                    MERGE (a)-[r:`{safe_type}`]->(b)
                    ON CREATE SET r.source_chunks = [$chunk_id], r.extraction_count = 1
                    ON MATCH SET r.source_chunks =
                        CASE
                            WHEN $chunk_id IN COALESCE(r.source_chunks, [])
                                THEN r.source_chunks
                            ELSE COALESCE(r.source_chunks, []) + [$chunk_id]
                        END,
                        r.extraction_count = COALESCE(r.extraction_count, 0) + 1
                    """,
                    {"start": src_id, "end": tgt_id, "chunk_id": chunk_hash},
                )
                count += 1
            except Exception as e:
                logger.warning("attach_source_chunks failed for (%s)-[%s]->(%s): %s",
                               src_id, safe_type, tgt_id, e)
    return count


def compute_mention_count(graph) -> int:
    """全Termに `mention_count` (どれだけのDocumentから MENTIONS されているか) を書き込む。

    Returns:
        更新した Term 件数
    """
    try:
        graph.query(
            """
            MATCH (t:Term)
            OPTIONAL MATCH (t)<-[:MENTIONS]-(d:Document)
            WITH t, COUNT(DISTINCT d) AS n
            SET t.mention_count = n
            """
        )
        r = graph.query("MATCH (t:Term) WHERE t.mention_count IS NOT NULL RETURN COUNT(t) AS n")
        n = r[0]["n"] if r else 0
        logger.info("compute_mention_count: updated %d Term nodes", n)
        return n
    except Exception as e:
        logger.error("compute_mention_count failed: %s", e)
        return 0


def compute_pagerank(graph, alpha: float = 0.85, max_iter: int = 100) -> int:
    """Termノード間の関係グラフでPageRankを計算しTerm.pagerankに書き込む。

    APOC非依存・NetworkX実装。MENTIONS は除外し Term-Term の意味的エッジのみ使用。
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx 未導入のため pagerank をスキップ")
        return 0

    try:
        edges = graph.query(
            """
            MATCH (a:Term)-[r]->(b:Term)
            WHERE type(r) <> 'MENTIONS'
            RETURN a.id AS s, b.id AS t
            """
        )
        if not edges:
            logger.info("compute_pagerank: Term-Term edges が0件、スキップ")
            return 0

        nx_g = nx.DiGraph()
        for row in edges:
            nx_g.add_edge(row["s"], row["t"])

        ranks = nx.pagerank(nx_g, alpha=alpha, max_iter=max_iter)
        logger.info("compute_pagerank: nodes=%d edges=%d", nx_g.number_of_nodes(), nx_g.number_of_edges())

        # Neo4j に書き戻し（バッチ）
        rows = [{"id": k, "score": float(v)} for k, v in ranks.items()]
        batch_size = 500
        updated = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            graph.query(
                """
                UNWIND $rows AS row
                MATCH (t:Term {id: row.id})
                SET t.pagerank = row.score
                """,
                {"rows": batch},
            )
            updated += len(batch)
        logger.info("compute_pagerank: updated %d Term nodes", updated)
        return updated
    except Exception as e:
        logger.error("compute_pagerank failed: %s", e)
        return 0


def enrich_post_build(graph) -> dict:
    """ビルド後の集計プロパティをまとめて書き込む。

    Returns:
        各更新件数の dict
    """
    return {
        "mention_count": compute_mention_count(graph),
        "pagerank": compute_pagerank(graph),
    }
