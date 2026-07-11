"""ドキュメント改訂時の部分グラフ更新（文書スコープ置換 / approach D）。

方針（軽微な陳腐化を許容する版）:
- チャンクIDは内容ハッシュ `sha256(doc_id + 本文)`。改訂しても不変チャンクは同一ID。
- 差分 = 旧新IDの集合差。removed を剪定し、added だけ再抽出、unchanged は素通り。
- 剪定は既存の来歴（edge.source_chunks / MENTIONS）を使う:
    * エッジ: source_chunks から removed を外し、空になった抽出エッジを削除
    * Document(チャンク)ノードを削除 → MENTIONS / REFERS_TO も消える
    * 孤立エンティティ（MENTIONS 0本・意味エッジ0本）を削除（idを返す→entityベクトル同期用）
    * ProcessedChunk を削除
- PGVector 同期: 文書スコープで「新チャンク集合に一致」させる
  （unchanged 行は残す / 消えた行は削除 / 新規行を挿入）。LLM抽出の成否とは独立。
- エンティティベクトル同期: 削除された孤立エンティティを除去、added チャンクが
  言及するエンティティを upsert。
- consolidate/pagerank 等の名寄せ・グローバル集計は「破壊的マージ据え置き＋定期
  フル再構築でGC」。増分更新時は軽量後処理（mention_count/search_keys）のみ推奨。

【ID移行方針 (C1)】既存グラフのID体系（位置ID `doc__c0001` や doc_id無しhash）と
make_chunk_id は一致しないため、**その文書に最初に update_document を実行した際は
全チャンクが added/removed 扱い＝文書フル置換**になる（LLM再抽出コスト=文書1つ分）。
以降の更新は差分だけになる。一括マイグレーションは行わない（フル置換で自然移行）。

add_chunk_fn は LLM 抽出＋MERGE＋attach_source_chunks＋Document作成＋ProcessedChunk記録を
1チャンク分行う呼び出し可能オブジェクト（build/driver 側から注入）。
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from graphrag_core.graph.schema import entity_node_predicate, chunk_edge, chunk_label

logger = logging.getLogger(__name__)


# ── チャンクID（内容ハッシュ, doc_id込み） ─────────────────────────────
def make_chunk_id(doc_id: str, content: str) -> str:
    """doc_id と本文からチャンクIDを決定する。

    doc_id を混ぜることで、別文書の同一テキストが衝突して同一視されるのを防ぐ
    （文書スコープ削除を安全にするため）。
    """
    return hashlib.sha256(f"{doc_id}\n{content}".encode("utf-8")).hexdigest()


# ── 差分検出 ───────────────────────────────────────────────────────
def get_doc_chunk_ids(graph, doc_id: str) -> Set[str]:
    """Neo4j 上でこの文書に属する既存チャンク(Document)ID集合。"""
    rows = graph.query(
        "MATCH (d:" + chunk_label() + " {source: $s}) RETURN d.id AS id", {"s": doc_id}
    )
    return {r["id"] for r in (rows or []) if r.get("id")}


def compute_delta(graph, doc_id: str, new_chunks: Iterable[Any]) -> Dict[str, List]:
    """新チャンク群と既存グラフを突き合わせ、added/removed/unchanged を返す。

    new_chunks: page_content を持つ Document 群。metadata['id'] は make_chunk_id で
                上書きされる（呼び出し側は本文だけ用意すればよい）。
    Returns: {"added": [chunk,...], "removed": [id,...], "unchanged": [id,...]}
    """
    new_by_id = {}
    for c in new_chunks:
        cid = make_chunk_id(doc_id, c.page_content)
        c.metadata["id"] = cid
        c.metadata.setdefault("source", doc_id)
        new_by_id[cid] = c
    new_ids = set(new_by_id)
    old_ids = get_doc_chunk_ids(graph, doc_id)
    added = [new_by_id[i] for i in sorted(new_ids - old_ids)]
    removed = sorted(old_ids - new_ids)
    unchanged = sorted(new_ids & old_ids)
    return {"added": added, "removed": removed, "unchanged": unchanged,
            "new_ids": sorted(new_ids)}


# ── 剪定（削除の肝） ───────────────────────────────────────────────
def prune_chunks(graph, chunk_ids: List[str], dry_run: bool = False) -> Dict[str, Any]:
    """指定チャンクIDの寄与をグラフから剪定する。

    dry_run=True なら削除せず、影響件数の見積りだけ返す。
    Returns には orphan_entity_ids（削除した孤立エンティティのid）を含む
    （エンティティベクトル同期用）。
    """
    ids = list({c for c in chunk_ids if c})
    if not ids:
        return {"edges_deleted": 0, "documents_deleted": 0,
                "orphan_entities_deleted": 0, "orphan_entity_ids": [],
                "processed_deleted": 0}
    _ent = entity_node_predicate("n")

    if dry_run:
        edges_touched = graph.query(
            "MATCH ()-[r]->() WHERE r.source_chunks IS NOT NULL "
            "AND any(x IN r.source_chunks WHERE x IN $ids) "
            "RETURN count(r) AS c", {"ids": ids})[0]["c"]
        edges_emptied = graph.query(
            "MATCH ()-[r]->() WHERE r.source_chunks IS NOT NULL "
            "AND size([x IN r.source_chunks WHERE NOT x IN $ids]) = 0 "
            "AND any(x IN r.source_chunks WHERE x IN $ids) "
            "RETURN count(r) AS c", {"ids": ids})[0]["c"]
        docs = graph.query(
            "MATCH (d:" + chunk_label() + ") WHERE d.id IN $ids RETURN count(d) AS c", {"ids": ids})[0]["c"]
        return {"edges_pruned": edges_touched, "edges_deleted_est": edges_emptied,
                "documents_deleted": docs, "orphan_entities_deleted": "n/a(dry)",
                "orphan_entity_ids": [], "processed_deleted": len(ids)}

    stats: Dict[str, Any] = {}
    # 1. エッジの source_chunks から removed を外す
    graph.query(
        "MATCH ()-[r]->() WHERE r.source_chunks IS NOT NULL "
        "AND any(x IN r.source_chunks WHERE x IN $ids) "
        "SET r.source_chunks = [x IN r.source_chunks WHERE NOT x IN $ids]",
        {"ids": ids})
    # 2. 空になった抽出エッジを削除（MENTIONS は source_chunks を持たないので無傷）
    r = graph.query(
        "MATCH ()-[r]->() WHERE r.source_chunks IS NOT NULL AND size(r.source_chunks) = 0 "
        "WITH r LIMIT 500000 DELETE r RETURN count(r) AS c")
    stats["edges_deleted"] = r[0]["c"] if r else 0
    # 3. Document(チャンク)ノード削除 → MENTIONS / REFERS_TO も消える
    r = graph.query(
        "UNWIND $ids AS cid MATCH (d:" + chunk_label() + " {id: cid}) "
        "WITH d LIMIT 500000 DETACH DELETE d RETURN count(d) AS c", {"ids": ids})
    stats["documents_deleted"] = r[0]["c"] if r else 0
    # 4. 孤立エンティティ削除（MENTIONS 0・意味エッジ0）。Document は除外。
    #    削除前に id を収集し、エンティティベクトル同期に使えるよう返す。
    orphan_rows = graph.query(
        f"MATCH (n) WHERE {_ent} AND NOT n:{chunk_label()} "
        "AND NOT (n)<-[:" + chunk_edge() + "]-() "
        "AND NOT EXISTS { MATCH (n)-[e]-() WHERE type(e) <> '" + chunk_edge() + "' } "
        "RETURN elementId(n) AS eid, n.id AS id LIMIT 500000") or []
    stats["orphan_entity_ids"] = sorted({r["id"] for r in orphan_rows if r.get("id")})
    if orphan_rows:
        graph.query(
            "UNWIND $eids AS eid MATCH (n) WHERE elementId(n) = eid "
            "DETACH DELETE n", {"eids": [r["eid"] for r in orphan_rows]})
    stats["orphan_entities_deleted"] = len(orphan_rows)
    # 5. ProcessedChunk 削除
    r = graph.query(
        "UNWIND $ids AS cid MATCH (p:ProcessedChunk {hash: cid}) DELETE p RETURN count(p) AS c",
        {"ids": ids})
    stats["processed_deleted"] = r[0]["c"] if r else 0
    logger.info("prune_chunks: %s", {k: v for k, v in stats.items() if k != "orphan_entity_ids"})
    return stats


# ── PGVector 同期（文書スコープで新チャンク集合に一致させる） ─────────
def sync_pgvector_document(
    pg_conn: str,
    collection: str,
    doc_id: str,
    new_chunks: List[Any],
    embeddings,
) -> Dict[str, int]:
    """PGVector のこの文書の行を new_chunks に一致させる。

    - metadata['id'] が new に無い行（旧版・位置ID行を含む）を削除
    - まだ無い id の行だけ埋め込み計算して挿入（unchanged は再埋め込みしない）
    - metadata['tokenized_content'] があれば BM25 用に反映
    LLM抽出の成否とは独立に実行できる（検索は常に新版テキストを見る）。
    """
    from graphrag_core.db.utils import (
        batch_pgvector_from_documents, batch_update_tokenized,
        delete_doc_embeddings_except, get_doc_embedding_ids,
    )
    new_ids = [c.metadata["id"] for c in new_chunks]
    deleted = delete_doc_embeddings_except(pg_conn, collection, doc_id, keep_ids=new_ids)
    existing = get_doc_embedding_ids(pg_conn, collection, doc_id)
    to_insert = [c for c in new_chunks if c.metadata["id"] not in existing]
    if to_insert:
        batch_pgvector_from_documents(
            to_insert, embeddings, connection=pg_conn,
            collection_name=collection, pre_delete_collection=False)
        batch_update_tokenized(pg_conn, to_insert)
    logger.info("sync_pgvector_document(%s): deleted=%d inserted=%d kept=%d",
                doc_id, deleted, len(to_insert), len(existing))
    return {"pg_deleted": deleted, "pg_inserted": len(to_insert),
            "pg_kept": len(existing)}


# ── エンティティベクトル同期 ───────────────────────────────────────
def sync_entity_vectors(
    graph,
    pg_conn: str,
    embeddings,
    added_chunk_ids: List[str],
    orphan_entity_ids: List[str],
) -> Dict[str, int]:
    """エンティティベクトルコレクションを差分同期する。

    - 剪定で消えた孤立エンティティの埋め込みを削除
    - added チャンクが MENTIONS するエンティティを upsert
    """
    from graphrag_core.retrieval.entity_vector import EntityVectorizer
    ev = EntityVectorizer(pg_conn, embeddings)
    n_del = ev.delete_entities(orphan_entity_ids) if orphan_entity_ids else 0
    entities = []
    if added_chunk_ids:
        rows = graph.query(
            "MATCH (d:" + chunk_label() + ")-[:" + chunk_edge() + "]->(e) WHERE d.id IN $ids "
            "RETURN DISTINCT e.id AS id, labels(e) AS labels",
            {"ids": added_chunk_ids}) or []
        for r in rows:
            labels = [l for l in (r.get("labels") or []) if not l.startswith("__")]
            entities.append({"id": r["id"], "type": labels[0] if labels else "Term",
                             "properties": {}})
    n_up = ev.upsert_entities(entities) if entities else 0
    logger.info("sync_entity_vectors: deleted=%d upserted=%d", n_del, n_up)
    return {"entity_vec_deleted": n_del, "entity_vec_upserted": n_up}


# ── オーケストレータ ───────────────────────────────────────────────
def update_document(
    graph,
    doc_id: str,
    new_chunks: List[Any],
    add_chunk_fn: Callable[[Any], Any],
    pg_conn: Optional[str] = None,
    pg_collection: Optional[str] = None,
    embeddings=None,
    run_post: Callable[[], None] | None = None,
    workers: int = 4,
) -> Dict[str, Any]:
    """1文書ぶんの改訂を反映する（グラフ + PGVector + エンティティベクトル）。

    add_chunk_fn(chunk): 1チャンクを抽出してグラフに追加する副作用関数
        （metadata['id'] は compute_delta が設定済み）。
    pg_conn/pg_collection/embeddings を渡すと PGVector とエンティティベクトルも同期。
    run_post(): 後処理（軽量なら enrichment.enrich_post_update を推奨）。
    workers: added チャンク抽出の並列数（build_kg 系と同じ ThreadPool パターン。
        vLLM の continuous batching を活かす）。
    """
    delta = compute_delta(graph, doc_id, new_chunks)
    n_add, n_rm, n_keep = len(delta["added"]), len(delta["removed"]), len(delta["unchanged"])
    logger.info("update_document(%s): +%d / -%d / =%d", doc_id, n_add, n_rm, n_keep)
    result: Dict[str, Any] = {"doc_id": doc_id, "added": n_add, "removed": n_rm,
                              "unchanged": n_keep, "add_errors": 0}

    # 1. 剪定
    prune_stats = prune_chunks(graph, delta["removed"]) if delta["removed"] else {}
    result["prune"] = {k: v for k, v in prune_stats.items() if k != "orphan_entity_ids"}

    # 2. added の再抽出（グラフ）— 並列（LLM待ちがボトルネックのため）
    added_ok_ids = []
    if delta["added"]:
        import concurrent.futures
        n_workers = max(1, min(workers, len(delta["added"])))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(add_chunk_fn, c): c for c in delta["added"]}
            for fut in concurrent.futures.as_completed(futs):
                chunk = futs[fut]
                try:
                    fut.result()
                    added_ok_ids.append(chunk.metadata["id"])
                except Exception as e:
                    result["add_errors"] += 1
                    logger.warning("add_chunk_fn failed for %s: %s",
                                   chunk.metadata.get("id"), e)
    result["add_failed_ids"] = [c.metadata["id"] for c in delta["added"]
                                if c.metadata["id"] not in added_ok_ids]

    # 3. PGVector 同期（LLM成否と独立: 全 new_chunks に一致させる）
    if pg_conn and pg_collection and embeddings is not None:
        try:
            result["pgvector"] = sync_pgvector_document(
                pg_conn, pg_collection, doc_id, new_chunks, embeddings)
        except Exception as e:
            logger.error("PGVector sync failed for %s: %s", doc_id, e)
            result["pgvector"] = {"error": str(e)}
        # 4. エンティティベクトル同期
        try:
            result["entity_vectors"] = sync_entity_vectors(
                graph, pg_conn, embeddings,
                added_chunk_ids=added_ok_ids,
                orphan_entity_ids=prune_stats.get("orphan_entity_ids", []))
        except Exception as e:
            logger.error("entity vector sync failed for %s: %s", doc_id, e)
            result["entity_vectors"] = {"error": str(e)}

    # 5. 後処理
    if run_post is not None and (n_add or n_rm):
        run_post()

    return result


def delete_document(
    graph,
    doc_id: str,
    pg_conn: Optional[str] = None,
    pg_collection: Optional[str] = None,
    embeddings=None,
) -> Dict[str, Any]:
    """文書をグラフ・PGVector・エンティティベクトルから完全に取り除く。"""
    old_ids = sorted(get_doc_chunk_ids(graph, doc_id))
    prune_stats = prune_chunks(graph, old_ids) if old_ids else {}
    result: Dict[str, Any] = {"doc_id": doc_id, "removed": len(old_ids),
                              "prune": {k: v for k, v in prune_stats.items()
                                        if k != "orphan_entity_ids"}}
    if pg_conn and pg_collection and embeddings is not None:
        from graphrag_core.db.utils import delete_doc_embeddings_except
        result["pg_deleted"] = delete_doc_embeddings_except(
            pg_conn, pg_collection, doc_id, keep_ids=[])
        result["entity_vectors"] = sync_entity_vectors(
            graph, pg_conn, embeddings, added_chunk_ids=[],
            orphan_entity_ids=prune_stats.get("orphan_entity_ids", []))
    return result
