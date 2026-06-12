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

from graphrag_core.graph.schema import entity_node_predicate

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
                # ノードタイプは外部スキーマで可変（Termとは限らない）ため、
                # add_graph_documents が作成済みの既存エッジを id ベースで MATCH して
                # プロパティを後付けする（MERGEだと別ラベル同名ノード間に誤エッジを作り得る）
                graph.query(
                    f"""
                    MATCH (a {{id: $start}})-[r:`{safe_type}`]->(b {{id: $end}})
                    SET r.source_chunks =
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
    """全エンティティノードに `mention_count` (どれだけのDocumentから MENTIONS されているか) を書き込む。

    Returns:
        更新したノード件数
    """
    _pred = entity_node_predicate("t")
    try:
        graph.query(
            f"""
            MATCH (t)
            WHERE {_pred}
            OPTIONAL MATCH (t)<-[:MENTIONS]-(d:Document)
            WITH t, COUNT(DISTINCT d) AS n
            SET t.mention_count = n
            """
        )
        r = graph.query(f"MATCH (t) WHERE {_pred} AND t.mention_count IS NOT NULL RETURN COUNT(t) AS n")
        n = r[0]["n"] if r else 0
        logger.info("compute_mention_count: updated %d entity nodes", n)
        return n
    except Exception as e:
        logger.error("compute_mention_count failed: %s", e)
        return 0


def compute_pagerank(graph, alpha: float = 0.85, max_iter: int = 100) -> int:
    """エンティティノード間の関係グラフでPageRankを計算し pagerank プロパティに書き込む。

    APOC非依存・NetworkX実装。MENTIONS は除外しエンティティ間の意味的エッジのみ使用。
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx 未導入のため pagerank をスキップ")
        return 0

    try:
        edges = graph.query(
            f"""
            MATCH (a)-[r]->(b)
            WHERE type(r) <> 'MENTIONS'
              AND {entity_node_predicate("a")}
              AND {entity_node_predicate("b")}
            RETURN a.id AS s, b.id AS t
            """
        )
        if not edges:
            logger.info("compute_pagerank: エンティティ間edgeが0件、スキップ")
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
                f"""
                UNWIND $rows AS row
                MATCH (t {{id: row.id}})
                WHERE {entity_node_predicate("t")}
                SET t.pagerank = row.score
                """,
                {"rows": batch},
            )
            updated += len(batch)
        logger.info("compute_pagerank: updated %d entity nodes", updated)
        return updated
    except Exception as e:
        logger.error("compute_pagerank failed: %s", e)
        return 0


def compute_search_keys(graph, batch_size: int = 2000) -> int:
    """全エンティティノードに `norm_id` と `search_keys` を書き込む。

    - norm_id: id の正規化形（NFKC + 小文字化 + 空白圧縮）
    - search_keys: norm_id + 正規化済み aliases + canonical_form の重複除去リスト

    検索側（pipeline.get_graph_context）はこのリストに対して CONTAINS 照合する。
    辞書適用（dictionary.apply_dictionary）の後に実行すること。

    Returns:
        更新したノード件数
    """
    from graphrag_core.text.japanese import normalize_entity_text, kana_variant_key

    _pred = entity_node_predicate("n")
    updated = 0
    skip = 0
    try:
        while True:
            rows = graph.query(
                f"""
                MATCH (n)
                WHERE {_pred}
                WITH n ORDER BY n.id
                SKIP $skip LIMIT $batch
                RETURN n.id AS id, n.aliases AS aliases, n.canonical_form AS canonical
                """,
                {"skip": skip, "batch": batch_size},
            )
            if not rows:
                break

            payload = []
            for r in rows:
                keys = [normalize_entity_text(r["id"])]
                for alias in (r.get("aliases") or []):
                    keys.append(normalize_entity_text(alias))
                if r.get("canonical"):
                    keys.append(normalize_entity_text(r["canonical"]))
                # かな揺れ骨格キー（送り仮名・助詞・末尾長音の揺れを照合で吸収）
                for base in list(keys):
                    kv = kana_variant_key(base)
                    if kv:
                        keys.append(kv)
                # 順序保持の重複除去 + 空文字除外
                seen = set()
                keys = [k for k in keys if k and not (k in seen or seen.add(k))]
                payload.append({"id": r["id"], "norm_id": keys[0] if keys else "", "keys": keys})

            graph.query(
                f"""
                UNWIND $rows AS row
                MATCH (n {{id: row.id}})
                WHERE {_pred}
                SET n.norm_id = row.norm_id, n.search_keys = row.keys
                """,
                {"rows": payload},
            )
            updated += len(payload)
            if len(rows) < batch_size:
                break
            skip += batch_size

        logger.info("compute_search_keys: updated %d entity nodes", updated)
        return updated
    except Exception as e:
        logger.error("compute_search_keys failed: %s", e)
        return updated


def enrich_post_build(graph) -> dict:
    """ビルド後の集計プロパティをまとめて書き込む。

    Returns:
        各更新件数の dict
    """
    return {
        "mention_count": compute_mention_count(graph),
        "pagerank": compute_pagerank(graph),
        "search_keys": compute_search_keys(graph),
    }
