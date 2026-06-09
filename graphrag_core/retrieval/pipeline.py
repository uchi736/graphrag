"""
qa_pipeline.py
==============
GraphRAG QAパイプラインの共通関数。
app.py（Streamlit UI）と batch_eval.py（バッチ評価）の両方から呼ばれる。

すべての関数は明示的な引数を受け取り、st.session_state やクロージャ変数には依存しない。
設定は config dict で渡す。
"""
from __future__ import annotations

import os
import logging
from typing import Dict, Any, List, Optional


from langchain_core.documents import Document

from graphrag_core.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    RELATION_RANKING_PROMPT,
    PATH_RANKING_PROMPT,
)
from graphrag_core.retrieval.hybrid import HybridRetriever, rerank_with_llm
from graphrag_core.text.japanese import SUDACHI_AVAILABLE
from graphrag_core.retrieval.entity_vector import EntityVectorizer
from graphrag_core.llm.langfuse_utils import observe, get_langfuse_callback
from graphrag_core.llm.factory import create_chat_llm

logger = logging.getLogger(__name__)

# ── デフォルト設定 ────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "graph_hop_count": 2,
    "retrieval_top_k": 5,
    "enable_japanese_search": True,
    "enable_rerank": True,
    "enable_entity_vector": False,
    "entity_similarity_threshold": 0.7,
    "search_mode": "hybrid",
    "include_kg_source_chunks": True,
    "path_max_candidates": 30,
    "include_graph_lines": True,
}


def _cfg(config: Optional[dict], key: str):
    """config dict からキーを取得。未指定時はデフォルト値を返す。"""
    if config and key in config:
        return config[key]
    return DEFAULT_CONFIG.get(key)


# ── エンティティ抽出 ──────────────────────────────────────────────────

@observe(name="entity_extraction")
def extract_entities_from_question(
    question: str,
    llm,
    embeddings=None,
    pg_conn: str = None,
    config: dict = None,
) -> Dict[str, Any]:
    """LLMとベクトル検索を使って質問からエンティティを抽出

    Returns:
        Dict with keys:
        - llm_entities: LLMで抽出したエンティティ
        - vector_entities: ベクトル検索で見つかったエンティティ [(id, score), ...]
        - merged_entities: 統合後のエンティティリスト
    """
    result = {
        "llm_entities": [],
        "vector_entities": [],
        "merged_entities": []
    }
    entities = []

    # 1. LLMによるエンティティ抽出
    extraction_prompt = ENTITY_EXTRACTION_PROMPT.format(question=question)
    try:
        response = llm.invoke(extraction_prompt, config=get_langfuse_callback())
        llm_entities = [e.strip() for e in response.content.split(',') if e.strip()]
        result["llm_entities"] = llm_entities
        entities.extend(llm_entities)
    except Exception:
        fallback_entities = [w for w in question.split() if len(w) > 1]
        result["llm_entities"] = fallback_entities
        entities.extend(fallback_entities)

    # 2. Entity vector で LLM抽出 entity の同義語/表記揺れを引く
    # 注: 質問全文を embed すると一般語 (はじめに/対策) や無関係概念 (PaaS/PDCA) が拾われるため、
    # 個別 entity ごとに高い閾値で同義語のみ補完する
    if _cfg(config, 'enable_entity_vector') and embeddings and pg_conn and entities:
        try:
            entity_vectorizer = EntityVectorizer(pg_conn, embeddings)
            similarity_threshold = max(_cfg(config, 'entity_similarity_threshold') or 0.7, 0.85)
            search_mode = _cfg(config, 'search_mode')
            all_similar = []
            for ent in entities[:8]:  # LLM抽出の上位8個までで引く
                try:
                    sims = entity_vectorizer.search_hybrid_entities(
                        ent,
                        k=3,
                        score_threshold=similarity_threshold,
                        search_type=search_mode
                    )
                    all_similar.extend(sims)
                except Exception:
                    continue
            # 重複除去
            seen = set(entities)
            for entity_id, score in all_similar:
                if entity_id and entity_id not in seen:
                    seen.add(entity_id)
                    entities.append(entity_id)
            result["vector_entities"] = all_similar

            if all_similar:
                logger.info(f"[Entity Vector Expand] Added {len(entities) - len(result['llm_entities'])} synonyms (threshold={similarity_threshold})")
        except Exception as e:
            logger.warning(f"[Entity Vector Expand Error] {e}")

    result["merged_entities"] = entities
    return result


# ── リレーションリランキング ──────────────────────────────────────────

@observe(name="relation_ranking")
def rank_relations_by_relevance(
    question: str,
    relations: list,
    llm,
    top_k: int = 15,
    doc_context: str = "",
) -> list:
    """関係性を質問関連度で上位top_k件に絞る（reranker優先、fallback LLM）"""
    if not relations:
        return []

    # cross-encoder リランカーが使えるならそちらを使う（高速）
    from graphrag_core.retrieval.reranker import is_reranker_enabled, rerank_by_score
    if is_reranker_enabled():
        return rerank_by_score(
            question, relations,
            text_fn=lambda r: f"{r['start']} -[{r['type']}]-> {r['end']}",
            top_k=top_k,
        )

    # フォールバック: LLM rerank（slow, gpt-oss-120b）
    relations_text = "\n".join([
        f"{i+1}. {r['start']} -[{r['type']}]-> {r['end']}"
        for i, r in enumerate(relations)
    ])

    ranking_prompt = RELATION_RANKING_PROMPT.format(
        question=question,
        relations_text=relations_text,
        top_k=top_k,
        document_context=doc_context if doc_context else "(なし)"
    )

    try:
        response = llm.invoke(ranking_prompt, config=get_langfuse_callback())
        output = response.content.strip()
        if not output:
            return relations[:top_k]

        selected_ids = []
        for x in output.split(','):
            x = x.strip()
            if x.isdigit():
                selected_ids.append(int(x))

        ranked = [relations[i-1] for i in selected_ids if 1 <= i <= len(relations)]
        if not ranked:
            return relations[:top_k]
        return ranked
    except Exception:
        return relations[:top_k]


# ── パス構築 ──────────────────────────────────────────────────────────


def parse_neo4j_paths(result: list, max_candidates: int = 30) -> list:
    """Neo4jからパス単位で返された結果を直接パースする。

    Cypherが RETURN [node IN nodes(path) | node.id] AS node_ids,
    [r IN relationships(path) | ...] AS rels 形式で返すことを前提とする。
    """
    paths = []
    seen: set = set()
    for row in result:
        node_ids = row.get("node_ids", [])
        rels = row.get("rels", [])
        if not node_ids or not rels:
            continue

        # 重複除去（ノード列で判定）
        path_key = tuple(node_ids)
        if path_key in seen:
            continue
        seen.add(path_key)

        # node_ids の順序で方向を決定（DB上の方向と異なる場合がある）
        path_triples = []
        chain_parts = []
        for k in range(len(node_ids) - 1):
            u, v = node_ids[k], node_ids[k + 1]
            rel_type = rels[k].get("type", "RELATED") if k < len(rels) else "RELATED"
            path_triples.append({"start": u, "type": rel_type, "end": v})
            if k == 0:
                chain_parts.append(u)
            chain_parts.append(f"-[{rel_type}]->")
            chain_parts.append(v)

        paths.append({
            "path_text": " ".join(chain_parts),
            "triples": path_triples,
            "nodes": list(node_ids),
            "length": len(node_ids) - 1,
        })

    # マルチホップ優先
    paths.sort(key=lambda p: -p["length"])
    return paths[:max_candidates]


# ── パスリランキング ──────────────────────────────────────────────────

@observe(name="path_ranking")
def rank_paths_by_relevance(
    question: str,
    paths: list,
    llm,
    top_k: int = 15,
    doc_context: str = "",
    log: list = None,
) -> list:
    """LLMを使ってパス（チェーン）をリランキング

    Args:
        log: ログ出力先リスト。指定されると処理過程を追記する。
    """
    if log is None:
        log = []
    if not paths:
        return []

    # cross-encoder リランカー優先（gpt-oss-120b rerank の数十倍速い）
    from graphrag_core.retrieval.reranker import is_reranker_enabled, score_candidates
    if is_reranker_enabled():
        log.append(f"**候補パス {len(paths)}件 → reranker(cross-encoder)でスコアリング (top_k={top_k})**")
        texts = [p["path_text"] for p in paths]
        doc_ctx = (doc_context or "").strip()
        query = f"{question}\n{doc_ctx}" if doc_ctx else question
        scores = score_candidates(query, texts)
        if scores is not None:
            ranked = sorted(zip(scores, paths), key=lambda t: -t[0])
            result = [p for _, p in ranked[:top_k]]
            log.append(f"**選択後: {len(result)}件 (reranker)**")
            for i, (sc, p) in enumerate(ranked[:top_k]):
                log.append(f"  選択{i+1} (score={sc:.3f}): `{p['path_text']}`")
            return result
        log.append("reranker失敗 → LLM rerankにフォールバック")

    # フォールバック: gpt-oss-120b での LLM rerank
    paths_text = "\n".join(
        f"{i+1}. {p['path_text']}"
        for i, p in enumerate(paths)
    )

    log.append(f"**候補パス {len(paths)}件 → LLMに送信 (top_k={top_k})**")
    for i, p in enumerate(paths):
        log.append(f"  候補{i+1}: `{p['path_text']}`")

    ranking_prompt = PATH_RANKING_PROMPT.format(
        question=question, paths_text=paths_text, top_k=top_k,
        document_context=doc_context if doc_context else "(なし)"
    )

    try:
        response = llm.invoke(ranking_prompt, config=get_langfuse_callback())
        output = response.content.strip()
        log.append(f"**LLM生出力:** `{output}`")

        if not output:
            log.append("LLM出力が空 → 先頭をそのまま使用")
            return paths[:top_k]

        selected_ids = []
        for x in output.split(","):
            x = x.strip()
            if x.isdigit():
                selected_ids.append(int(x))

        ranked = [paths[i - 1] for i in selected_ids if 1 <= i <= len(paths)]
        if not ranked:
            log.append("有効なIDなし → 先頭をそのまま使用")
            return paths[:top_k]

        result = ranked[:top_k]
        log.append(f"**選択後: {len(result)}件**")
        for i, p in enumerate(result):
            log.append(f"  選択{i+1}: `{p['path_text']}`")
        return result
    except Exception as e:
        log.append(f"エラー: {e} → 先頭をそのまま使用")
        return paths[:top_k]


# ── グラフ検索 ────────────────────────────────────────────────────────

@observe(name="graph_search")
def get_graph_context(
    question: str,
    graph,
    llm,
    embeddings=None,
    pg_conn: str = None,
    config: dict = None,
    doc_context: str = "",
    precomputed_entities: dict = None,
) -> Dict[str, Any]:
    """質問からエンティティを抽出し、N-hopトラバーサルでサブグラフを取得

    Args:
        precomputed_entities: 事前に抽出済みのentity_result（並列化時に使用）

    Returns:
        Dict with keys:
        - triples: 関係性トリプルのリスト
        - paths: パスのリスト
        - extracted_entities: 抽出エンティティ情報
    """
    # 1. エンティティ抽出（事前計算済みならスキップ）
    if precomputed_entities is not None:
        entity_result = precomputed_entities
    else:
        entity_result = extract_entities_from_question(
            question, llm, embeddings=embeddings, pg_conn=pg_conn, config=config
        )
    entities = entity_result.get("merged_entities", [])
    if not entities:
        return {"triples": [], "paths": [], "extracted_entities": entity_result}

    # 2. ホップ数を取得
    hop_count = _cfg(config, 'graph_hop_count')
    top_k_map = {1: 15, 2: 20, 3: 25}
    max_candidates = int(_cfg(config, 'path_max_candidates') or 30)

    # ログリスト（呼び出し元がconfig経由で渡す）
    log = config.get("_path_rerank_log", []) if config else []

    # 3. Neo4j: パス戦略で起点ノードからN-hopのパスをそのまま返す
    top_k = top_k_map.get(hop_count, 15)
    limit_map = {1: 30, 2: 50, 3: 80}
    limit = limit_map.get(hop_count, 50)
    _hex = "NOT n.id =~ '[0-9a-f]{32,}'"
    _hex_node = "NOT node.id =~ '[0-9a-f]{32,}'"

    if hop_count == 1:
        query = (
            "UNWIND $entities AS entity "
            "MATCH (n) WHERE n.id CONTAINS entity AND " + _hex + " "
            "WITH collect(DISTINCT n) AS matched_nodes "
            "UNWIND matched_nodes AS start_node "
            "MATCH path = (start_node)-[r]-(end_node) "
            "WHERE type(r) <> 'MENTIONS' AND NOT end_node.id =~ '[0-9a-f]{32,}' "
            "RETURN [node IN nodes(path) | node.id] AS node_ids, "
            "[rel IN relationships(path) | {start: startNode(rel).id, type: type(rel), end: endNode(rel).id}] AS rels "
            f"LIMIT {limit}"
        )
    else:
        query = (
            "UNWIND $entities AS entity "
            "MATCH (n) WHERE n.id CONTAINS entity AND " + _hex + " "
            "WITH collect(DISTINCT n) AS matched_nodes "
            "UNWIND matched_nodes AS start_node "
            f"MATCH path = (start_node)-[*1..{hop_count}]-(end_node) "
            "WHERE ALL(r IN relationships(path) WHERE type(r) <> 'MENTIONS') "
            "AND ALL(node IN nodes(path) WHERE " + _hex_node + ") "
            "AND start_node <> end_node "
            "RETURN [node IN nodes(path) | node.id] AS node_ids, "
            "[rel IN relationships(path) | {start: startNode(rel).id, type: type(rel), end: endNode(rel).id}] AS rels "
            f"LIMIT {limit}"
        )

    try:
        result = graph.query(query, params={"entities": entities})
        if not result:
            return {"triples": [], "paths": [], "extracted_entities": entity_result}

        log.clear()
        log.append(f"**取得数: {len(result)}件, hop={hop_count}, top_k={top_k}** (Neo4j)")

        paths = parse_neo4j_paths(result, max_candidates=max_candidates)
        log.append(f"パス数: {len(paths)}件 (max_candidates={max_candidates})")
        if paths:
            if len(paths) > top_k:
                paths = rank_paths_by_relevance(question, paths, llm, top_k=top_k, doc_context=doc_context, log=log)
            else:
                log.append(f"パス{len(paths)}件 ≤ top_k={top_k} → ランキングスキップ")
            seen = set()
            flat_triples = []
            for p in paths:
                for t in p.get("triples", []):
                    key = (t["start"], t["type"], t["end"])
                    if key not in seen:
                        seen.add(key)
                        flat_triples.append(t)
            log.append(f"**最終出力: パス{len(paths)}件, フラットトリプル{len(flat_triples)}件**")
            return {"triples": flat_triples, "paths": paths, "extracted_entities": entity_result}
        else:
            log.append("パスなし → 従来方式(個別トリプル)にフォールバック")
            fb_query = (
                "UNWIND $entities AS entity "
                "MATCH (n) WHERE n.id CONTAINS entity AND " + _hex + " "
                "WITH collect(DISTINCT n) AS matched_nodes "
                "UNWIND matched_nodes AS start_node "
                "MATCH (start_node)-[r]-(connected_node) "
                "WHERE type(r) <> 'MENTIONS' AND NOT connected_node.id =~ '[0-9a-f]{32,}' "
                "WITH r, startNode(r) AS actual_start, endNode(r) AS actual_end "
                "RETURN DISTINCT actual_start.id AS start, type(r) AS type, actual_end.id AS end "
                f"LIMIT {limit}"
            )
            result = graph.query(fb_query, params={"entities": entities})
            if result and len(result) > top_k:
                result = rank_relations_by_relevance(question, result, llm, top_k=top_k, doc_context=doc_context)
            return {"triples": result if result else [], "paths": [], "extracted_entities": entity_result}
    except Exception as e:
        logger.error("primary graph query failed: %s", e, exc_info=True)
        log.append(f"⚠️ グラフクエリ失敗: {type(e).__name__}: {e} → フォールバックへ")
        # フォールバック: 単純な1-hopマッチング
        fallback_query = """
        MATCH (n)-[r]->(m)
        WHERE (ANY(entity IN $entities WHERE n.id CONTAINS entity OR m.id CONTAINS entity))
        AND type(r) <> 'MENTIONS'
        AND NOT n.id =~ '[0-9a-f]{32,}' AND NOT m.id =~ '[0-9a-f]{32,}'
        RETURN DISTINCT n.id AS start, type(r) AS type, m.id AS end
        LIMIT 20
        """
        try:
            result = graph.query(fallback_query, params={"entities": entities})
            if result:
                result = rank_relations_by_relevance(question, result, llm, top_k=15, doc_context=doc_context)
            return {"triples": result if result else [], "paths": [], "extracted_entities": entity_result}
        except Exception as e2:
            logger.error("fallback graph query also failed: %s", e2, exc_info=True)
            log.append(f"⚠️ フォールバッククエリも失敗: {type(e2).__name__}: {e2}")
            return {"triples": [], "paths": [], "extracted_entities": entity_result}


# ── 検索・マージ ──────────────────────────────────────────────────────

@observe(name="retrieve_and_merge", capture_output=False)
def retriever_and_merge(
    question: str,
    graph,
    llm,
    embeddings,
    vector_retriever,
    pg_conn: str,
    pg_collection: str = "graphrag",
    config: dict = None,
) -> Dict[str, Any]:
    """エンティティ抽出とドキュメント検索を並列実行し、グラフリランキング"""
    from concurrent.futures import ThreadPoolExecutor

    # ── Phase 1: エンティティ抽出 + ドキュメント検索を並列実行 ──
    def _search_documents():
        """ドキュメント検索（+ オプションのLLMリランク）"""
        if _cfg(config, 'enable_japanese_search') and SUDACHI_AVAILABLE:
            try:
                hybrid_retriever = HybridRetriever.get_instance(pg_conn, collection_name=pg_collection)
                query_embedding = embeddings.embed_query(question)
                search_type = _cfg(config, 'search_mode')
                retrieval_top_k = _cfg(config, 'retrieval_top_k')

                hybrid_results = hybrid_retriever.search(
                    query_text=question,
                    query_vector=query_embedding,
                    k=retrieval_top_k,
                    search_type=search_type
                )

                if _cfg(config, 'enable_rerank') and len(hybrid_results) > retrieval_top_k:
                    hybrid_results = rerank_with_llm(
                        question, hybrid_results, llm, k=retrieval_top_k
                    )

                return [
                    Document(page_content=r['text'], metadata=r['metadata'])
                    for r in hybrid_results
                ]
            except Exception as e:
                logger.warning(f"ハイブリッド検索エラー（ベクトル検索にフォールバック）: {e}")
                return vector_retriever.invoke(question) if vector_retriever else []
        else:
            return vector_retriever.invoke(question) if vector_retriever else []

    entity_result = None
    docs = []

    if graph is not None:
        # グラフありの場合: エンティティ抽出とドキュメント検索を並列実行
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_entities = executor.submit(
                extract_entities_from_question, question, llm,
                embeddings=embeddings, pg_conn=pg_conn, config=config
            )
            future_docs = executor.submit(_search_documents)

            entity_result = future_entities.result()
            docs = future_docs.result()
    else:
        docs = _search_documents()

    # ── Phase 2: グラフ検索（事前抽出エンティティ + doc_contextでリランキング）──
    triples = []
    paths = []
    extracted_entities = {}
    if graph is not None:
        doc_context = "\n---\n".join(d.page_content[:200] for d in docs[:5])
        graph_result = get_graph_context(
            question, graph, llm,
            embeddings=embeddings, pg_conn=pg_conn,
            config=config, doc_context=doc_context,
            precomputed_entities=entity_result,
        )
        triples = graph_result.get("triples", [])
        paths = graph_result.get("paths", [])
        extracted_entities = graph_result.get("extracted_entities", {})

    # 3. グラフからソースチャンクを取得
    # ① edge.source_chunks (新方式: triple単位の正確なリンク) を優先
    # ② 無ければ MENTIONS 経由 (旧方式: 端点エンティティのいずれかが言及されてるチャンク)
    kg_chunks = []
    if triples and _cfg(config, 'include_kg_source_chunks'):
        existing_texts = {d.page_content for d in docs}

        # ── 試行① edge.source_chunks 直引き ──
        triple_keys = [
            {"s": t.get("start"), "p": t.get("type"), "o": t.get("end")}
            for t in triples
            if t.get("start") and t.get("type") and t.get("end")
        ]
        chunk_results: list = []
        if triple_keys:
            try:
                cypher_strict = """
                UNWIND $triples AS tk
                MATCH (a:Term {id: tk.s})-[r]-(b:Term {id: tk.o})
                WHERE type(r) = tk.p AND r.source_chunks IS NOT NULL
                UNWIND r.source_chunks AS cid
                MATCH (d:Document {id: cid})
                RETURN DISTINCT
                  d.id AS chunk_id,
                  substring(d.text, 0, 2000) AS text,
                  d.source AS source,
                  d.page AS page
                LIMIT 5
                """
                chunk_results = graph.query(cypher_strict, params={"triples": triple_keys}) or []
            except Exception as e:
                logger.warning("source_chunks 直引き失敗 → MENTIONS にフォールバック: %s", e)
                chunk_results = []

        # ── フォールバック② MENTIONS 経由 ──
        if not chunk_results:
            entity_names = list(set(
                [t.get('start') for t in triples if t.get('start')]
                + [t.get('end') for t in triples if t.get('end')]
            ))
            if entity_names:
                try:
                    if hasattr(graph, 'get_source_chunks_list'):
                        chunk_results = graph.get_source_chunks_list(entity_names, limit=5)
                    else:
                        chunk_query = """
                        UNWIND $entity_names AS entity_name
                        MATCH (e {id: entity_name})<-[:MENTIONS]-(doc:Document)
                        RETURN DISTINCT doc.id AS chunk_id,
                               substring(doc.text, 0, 2000) AS text,
                               doc.source AS source,
                               doc.page AS page
                        LIMIT 5
                        """
                        chunk_results = graph.query(chunk_query, params={"entity_names": entity_names}) or []
                except Exception:
                    chunk_results = []

        for r in chunk_results:
            if r.get('text') and r['text'] not in existing_texts:
                kg_chunks.append(Document(
                    page_content=r['text'],
                    metadata={
                        'id': r.get('chunk_id'),
                        'source': r.get('source', 'KG'),
                        'page': r.get('page'),
                    }
                ))
                existing_texts.add(r['text'])

    # 4. トリプル/パスに出典情報を付与
    graph_lines = []
    source_chunks = {}
    if triples and hasattr(graph, 'get_source_chunks_for_entities'):
        entity_ids = list(set(
            [t.get('start') for t in triples if t.get('start')]
            + [t.get('end') for t in triples if t.get('end')]
        ))
        source_chunks = graph.get_source_chunks_for_entities(entity_ids)

    if paths:
        for i, path_data in enumerate(paths, 1):
            path_text = path_data.get("path_text", "")
            if not path_text:
                continue
            path_triples = path_data.get("triples", [])
            # 1-hop はトリプル、2-hop以上は推論パスと表記を分ける
            is_multi = len(path_triples) >= 2
            label = f"推論パス{i}" if is_multi else f"トリプル{i}"
            if path_triples:
                first_ent = path_triples[0].get("start", "")
                last_ent = path_triples[-1].get("end", "")
                src = (
                    source_chunks.get(first_ent, {}).get('source')
                    or source_chunks.get(last_ent, {}).get('source')
                    or ''
                )
                header = f"{label}: {path_text}" + (f" [出典: {src}]" if src else "")
                graph_lines.append(header)
                if is_multi:
                    for j, t in enumerate(path_triples, 1):
                        s, r, e = t.get('start', ''), t.get('type', ''), t.get('end', '')
                        graph_lines.append(f"  [P{i}-E{j}] {s} -[{r}]-> {e}")
            else:
                graph_lines.append(f"{label}: {path_text}")
    elif triples:
        for t in triples:
            start, rel, end = t.get('start', ''), t.get('type', ''), t.get('end', '')
            src = source_chunks.get(start, {}).get('source') or source_chunks.get(end, {}).get('source') or ''
            if src:
                graph_lines.append(f"{start} -[{rel}]→ {end} [出典: {src}]")
            else:
                graph_lines.append(f"{start} -[{rel}]→ {end}")

    if not graph_lines:
        graph_lines = ["(グラフデータなし)"]

    # graph_lines を context から除外するモード (KG はエンティティ抽出経由で利用、テキストは出さない)
    if not _cfg(config, "include_graph_lines"):
        graph_lines = []

    # 5. コンテキスト組み立て
    all_docs = docs.copy()
    if _cfg(config, 'include_kg_source_chunks'):
        all_docs.extend(kg_chunks)

    doc_contexts = []
    for d in all_docs:
        source = d.metadata.get('source', 'Unknown')
        doc_contexts.append(f"[出典: {source}]\n{d.page_content}")

    if graph_lines:
        context = (
            "<GRAPH_CONTEXT>\n" + "\n".join(graph_lines) + "\n</GRAPH_CONTEXT>\n\n" +
            "<DOCUMENT_CONTEXT>\n" + "\n---\n".join(doc_contexts) + "\n</DOCUMENT_CONTEXT>"
        )
    else:
        context = "<DOCUMENT_CONTEXT>\n" + "\n---\n".join(doc_contexts) + "\n</DOCUMENT_CONTEXT>"
    return {
        "context": context,
        "question": question,
        "vector_sources": docs,
        "kg_source_chunks": kg_chunks,
        "graph_sources": triples,
        "graph_paths": paths,
        "extracted_entities": extracted_entities
    }
