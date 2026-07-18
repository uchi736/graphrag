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
from graphrag_core.graph.schema import chunk_edge, chunk_label
from graphrag_core.llm.langfuse_utils import observe, get_langfuse_callback
from graphrag_core.llm.factory import create_chat_llm

logger = logging.getLogger(__name__)

# ── デフォルト設定 ────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "graph_hop_count": 2,
    "retrieval_top_k": 5,
    "rerank_pool_size": 20,
    "enable_japanese_search": True,
    "enable_rerank": True,
    "enable_entity_vector": False,
    "entity_similarity_threshold": 0.85,
    "search_mode": "hybrid",
    "include_kg_source_chunks": True,
    "kg_chunk_top_k": 5,
    "path_max_candidates": 30,
    "include_graph_lines": False,
    # 参照追跡（「P.98参照」「『◯◯』をご覧ください」を1ホップ辿る）。
    # 機構はSIM質問等で実証済みだが、FJ-Hard 100問ベンチでは正味効果が
    # 中立〜微負（FJH-11参照）だったためデフォルトOFF。明示的な参照を
    # 辿る必要があるマニュアル・規程QAでは有効化を検討
    "enable_reference_follow": False,
    "reference_follow_top_k": 5,
    # 条件付き関係の検索ルーティング（既定OFF・規程系限定）
    "enable_condition_routing": False,
    "include_condition_lines": False,
    "condition_routing_top_k": 20,
}


def _cfg(config: Optional[dict], key: str):
    """config dict からキーを取得。未指定時はデフォルト値を返す。"""
    if config and key in config:
        return config[key]
    return DEFAULT_CONFIG.get(key)


# ── 条件付き関係(qualifier/reify)の検索ルーティング（既定OFF・規程系限定） ──

def classify_condition_intent(question: str) -> str:
    """質問が条件起点(列挙/閾値/条件)かをヒューリスティックで判定（LLM不使用）。

    単一チャンクの条件照会は 'none' を返してルーティングしない（A/B中立のため）。
    """
    import re
    if re.search(r"全て|すべて|全部|列挙|網羅|もれなく|漏れなく|どんな場合|どのような場合|いずれの", question):
        return "enumerate_all"
    if re.search(r"(以上|以下|未満|を超え|超える)", question) and re.search(r"[0-9０-９]|％|%|割|倍", question):
        return "rate_threshold"
    if re.search(r"条件|が絡む|を含む|該当する", question):
        return "condition_origin"
    return "none"


def get_condition_origin_context(question, graph, config, pg_collection) -> dict:
    """reify グラフ(:CondFact/:Cond/[:WHEN])を決定的Cypherで引き、<CONDITION_FACTS> を作る。

    collection スコープ(:CondFact.pg_collection)で絞り、コーパス横断の漏れを防ぐ。
    Returns: {"facts": [...], "block": str}
    """
    import re
    top_k = int(_cfg(config, "condition_routing_top_k") or 20)
    intent = classify_condition_intent(question)
    rows = []
    try:
        if intent == "rate_threshold":
            m = re.search(r"([0-9０-９]+)\s*(?:％|%|割|倍|日|円|時間|度)?\s*(?:以上|を超え|超える|より大き)", question)
            thr = int(m.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))) if m else 0
            rows = graph.query(
                "MATCH (st:CondFact) WHERE st.pg_collection=$c AND st.value_num >= $thr "
                "OPTIONAL MATCH (st)-[:WHEN]->(cd:Cond) "
                "RETURN st.label AS label, st.value_num AS value, st.unit AS unit, st.source AS source, "
                "collect(DISTINCT cd.value) AS conds ORDER BY st.value_num DESC LIMIT $k",
                {"c": pg_collection, "thr": thr, "k": top_k},
            )
        else:
            keys = re.findall(r"[一-龥ぁ-んァ-ヶー]{2,}", question)[:8]
            rows = graph.query(
                "UNWIND $keys AS k "
                "MATCH (st:CondFact)-[:WHEN]->(cd:Cond) "
                "WHERE st.pg_collection=$c AND (cd.value CONTAINS k OR COALESCE(cd.norm_value,'') CONTAINS k) "
                "WITH DISTINCT st "
                "MATCH (st)-[:WHEN]->(allc:Cond) "
                "RETURN st.label AS label, st.value_num AS value, st.unit AS unit, st.source AS source, "
                "collect(DISTINCT allc.value) AS conds ORDER BY st.value_num LIMIT $k",
                {"keys": keys, "c": pg_collection, "k": top_k},
            )
    except Exception as e:
        logger.warning("condition-origin query failed: %s", e)
        return {"facts": [], "block": ""}

    facts = [{"label": r.get("label"), "value": r.get("value"), "unit": r.get("unit"),
              "source": r.get("source"), "conditions": r.get("conds", [])} for r in (rows or [])]
    if not facts:
        return {"facts": [], "block": ""}
    lines = []
    for f in facts:
        val = (f"{f['value']}{f['unit'] or ''}"
               if f["value"] not in (None, -1) else (f["label"] or ""))
        conds = " かつ ".join([c for c in f["conditions"] if c])
        src = f" [出典: {f['source']}]" if f.get("source") else ""
        lines.append(f"- {f['label']}: {val} （条件: {conds}）{src}")
    block = "<CONDITION_FACTS>\n" + "\n".join(lines) + "\n</CONDITION_FACTS>"
    return {"facts": facts, "block": block}


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
        "merged_entities": [],
        "theme_keywords": [],
    }
    entities = []

    # 1. LLMによる高/低レベルキーワード同時抽出（LightRAG dual-level 流）。
    #    低レベル=固有名（search_keys照合用）、高レベル=テーマ語（関係キーワード索引用）。
    #    LLM呼び出し回数は従来と同じ1回。JSONパース失敗時は旧カンマ区切り解釈へ。
    from graphrag_core.prompts import DUAL_KEYWORD_EXTRACTION_PROMPT
    extraction_prompt = DUAL_KEYWORD_EXTRACTION_PROMPT.format(question=question)
    try:
        response = llm.invoke(extraction_prompt, config=get_langfuse_callback())
        content = response.content.strip()
        llm_entities = []
        try:
            import json as _json
            import re as _re
            t = _re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", content)
            m = _re.search(r"\{.*\}", t, _re.DOTALL)
            data = _json.loads(m.group(0) if m else t)
            llm_entities = [str(e).strip() for e in (data.get("entities") or []) if str(e).strip()]
            result["theme_keywords"] = [
                str(k).strip() for k in (data.get("themes") or []) if str(k).strip()][:6]
        except Exception:
            # JSONで返らなかった場合は旧挙動（カンマ区切り）として解釈
            llm_entities = [e.strip() for e in content.split(',') if e.strip()]
        llm_entities = [e for e in llm_entities if e not in ("なし", "None", "N/A", "n/a")]
        result["llm_entities"] = llm_entities
        entities.extend(llm_entities)
    except Exception:
        # 日本語質問はスペース区切りされないため、Sudachiトークン化でフォールバック
        fallback_entities = []
        try:
            from graphrag_core.text.japanese import get_japanese_processor
            processor = get_japanese_processor()
            if processor:
                fallback_entities = [
                    t for t in processor.tokenize(question).split() if len(t) > 1
                ]
        except Exception:
            pass
        if not fallback_entities:
            fallback_entities = [w for w in question.split() if len(w) > 1]
        result["llm_entities"] = fallback_entities
        entities.extend(fallback_entities)

    # 2. Entity vector で LLM抽出 entity の同義語/表記揺れを引く
    # 注: 質問全文を embed すると一般語 (はじめに/対策) や無関係概念 (PaaS/PDCA) が拾われるため、
    # 個別 entity ごとに高い閾値で同義語のみ補完する
    if _cfg(config, 'enable_entity_vector') and embeddings and pg_conn and entities:
        try:
            entity_vectorizer = EntityVectorizer(pg_conn, embeddings)
            # 同義語/表記揺れ補完は高類似度が必須。既定 0.85（DEFAULT_CONFIG/Settings）だが
            # 下限切り上げは撤廃し、設定値をそのまま使う（0.85未満も調整可能に）。
            similarity_threshold = _cfg(config, 'entity_similarity_threshold') or 0.85
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


# ── トリプル/パスの自然文化（cross-encoder入力用）──────────────────────
# cross-encoderリランカー（bge-reranker等）は自然文ペアで学習されているため、
# "A -[IS_A]-> B" のような記号表記より自然文の方がスコアリングが安定する。
# 表示用テキスト(path_text)は従来の記号表記のまま変えない。

_REL_TEMPLATES = {
    "IS_A": "{s}は{o}の一種である",
    "BELONGS_TO_CATEGORY": "{s}は{o}に属する",
    "PART_OF": "{s}は{o}の一部である",
    "HAS_PART": "{s}は{o}を含む",
    "HAS_STEP": "{s}には{o}という手順がある",
    "FOLLOWS": "{s}は{o}の後に行う",
    "PRECEDES": "{s}は{o}の前に行う",
    "REQUIRES_BEFORE": "{s}の前に{o}が必要である",
    "HAS_ATTRIBUTE": "{s}は{o}という属性を持つ",
    "HAS_VALUE": "{s}の値は{o}である",
    "MEASURED_IN": "{s}は{o}で測定される",
    "CAUSES": "{s}は{o}を引き起こす",
    "AFFECTS": "{s}は{o}に影響する",
    "PREVENTS": "{s}は{o}を防ぐ",
    "MITIGATES": "{s}は{o}を軽減する",
    "DEPENDS_ON": "{s}は{o}に依存する",
    "REQUIRES": "{s}は{o}を必要とする",
    "ENABLES": "{s}は{o}を可能にする",
    "APPLIES_TO": "{s}は{o}に適用される",
    "COVERS": "{s}は{o}を対象に含む",
    "EXCLUDES": "{s}は{o}を対象外とする",
    "TARGETS": "{s}は{o}を対象とする",
    "DEFINED_BY": "{s}は{o}で定義される",
    "REFERENCES": "{s}は{o}を参照する",
    "SAME_AS": "{s}は{o}と同義である",
    "ALIAS_OF": "{s}は{o}の別名である",
    "OWNED_BY": "{s}は{o}が所有する",
    "MANAGED_BY": "{s}は{o}が管理する",
    "ISSUED_BY": "{s}は{o}が発行する",
    "ENACTED_BY": "{s}は{o}が制定する",
    "OPERATED_BY": "{s}は{o}が運営する",
    "OCCURRED_IN": "{s}は{o}に行われた",
    "VALID_FROM": "{s}は{o}から有効である",
    "COMPARED_TO": "{s}は{o}と比較される",
    "DESCRIBED_IN": "{s}は{o}に記載されている",
    "USES": "{s}は{o}を使用する",
    "RELATED_TO": "{s}は{o}と関連する",
}


def triple_to_natural_text(t: dict) -> str:
    """トリプル1件を自然文に変換（未知の関係タイプは汎用表現）"""
    s, rel, o = t.get("start", ""), t.get("type", ""), t.get("end", "")
    template = _REL_TEMPLATES.get(rel)
    if template:
        return template.format(s=s, o=o)
    return f"{s}と{o}は{rel}の関係にある"


def path_to_natural_text(path: dict) -> str:
    """パス（トリプル列）を自然文に変換"""
    triples = path.get("triples", [])
    if not triples:
        return path.get("path_text", "")
    return "。".join(triple_to_natural_text(t) for t in triples)


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
            text_fn=triple_to_natural_text,
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

        # 重複除去（ノード列 + 関係タイプ列で判定。
        # ノード列だけだと同一ノード間の異なる関係のパスが消える）
        path_key = (tuple(node_ids), tuple(r.get("type", "") for r in rels))
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

    # 注: 以前はパス長降順（マルチホップ優先）でソートしていたが、
    # 質問に直接答える1-hopの事実エッジほど落とされやすく逆効果のため廃止。
    # Cypher側のスコア順（extraction_count/pagerank）をそのまま保持する。
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
        # cross-encoderには自然文化したパスを渡す（記号表記はスコアが不安定）。
        # doc_context の連結はクエリが長くなり質問の信号が薄まるため行わない
        # （doc_contextはLLM rerankフォールバック側でのみ使用）。
        texts = [path_to_natural_text(p) for p in paths]
        scores = score_candidates(question, texts)
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
    """質問からグラフコンテキストを取得する（3経路）。

    1. 低レベル経路: エンティティ抽出→search_keys照合→N-hopパス探索（従来）
    2. テーマ経路: 高レベルキーワード→関係キーワード索引（{collection}_relations）
       に埋め込み照合し、話題の合う関係を追加（LightRAG dual-level 流。
       固有名を含まない抽象質問への対策）
    3. フォールバック: 1が全滅時、質問全文→エンティティ埋め込みで起点を
       シードして再探索（llm-graph-builder の entity_vector 流）

    Returns:
        Dict with keys: triples / paths / extracted_entities
    """
    if precomputed_entities is not None:
        entity_result = precomputed_entities
    else:
        entity_result = extract_entities_from_question(
            question, llm, embeddings=embeddings, pg_conn=pg_conn, config=config
        )

    ctx = _graph_context_by_entities(
        question, graph, llm, embeddings=embeddings, pg_conn=pg_conn,
        config=config, doc_context=doc_context, entity_result=entity_result)

    # 経路2: テーマ経路（高レベルキーワード → 関係キーワード索引）
    themes = entity_result.get("theme_keywords") or []
    collection = (config or {}).get("pg_collection")
    if (themes and pg_conn and collection and embeddings
            and _cfg(config, 'enable_theme_keywords') is not False):
        try:
            from graphrag_core.graph.relation_keywords import search_relations_by_theme
            theme_triples = search_relations_by_theme(
                pg_conn, collection, embeddings, themes, k=8)
            seen = {(t.get("start"), t.get("type"), t.get("end"))
                    for t in ctx.get("triples", [])}
            added = [t for t in theme_triples
                     if (t["start"], t["type"], t["end"]) not in seen]
            if added:
                ctx["triples"] = list(ctx.get("triples", [])) + added
                log = (config or {}).get("_path_rerank_log")
                if isinstance(log, list):
                    log.append(f"テーマ経路: {themes} → 関係{len(added)}件追加")
                logger.info("[Theme Route] %s -> +%d relations", themes, len(added))
        except Exception as e:
            logger.warning("theme relation search failed (経路2スキップ): %s", e)

    # 経路3: エンティティ照合が全滅したら、質問全文→エンティティ埋め込みでシード
    if (not ctx.get("triples") and embeddings and pg_conn
            and _cfg(config, 'enable_entity_vector')):
        try:
            ev = EntityVectorizer(pg_conn, embeddings)
            seed = [eid for eid, _ in ev.search_similar_entities(question, k=3) if eid]
            if seed:
                fb_result = {**entity_result, "merged_entities": seed}
                ctx2 = _graph_context_by_entities(
                    question, graph, llm, embeddings=embeddings, pg_conn=pg_conn,
                    config=config, doc_context=doc_context, entity_result=fb_result)
                if ctx2.get("triples"):
                    ctx2["extracted_entities"] = {
                        **entity_result, "fallback_seed_entities": seed}
                    logger.info("[Entity Fallback] question-vector seed=%s", seed)
                    ctx = ctx2
        except Exception as e:
            logger.warning("entity-vector fallback failed (経路3スキップ): %s", e)

    return ctx


def _graph_context_by_entities(
    question: str,
    graph,
    llm,
    embeddings=None,
    pg_conn: str = None,
    config: dict = None,
    doc_context: str = "",
    entity_result: dict = None,
) -> Dict[str, Any]:
    """低レベル経路の本体: エンティティ→search_keys照合→N-hopパス探索（従来実装）。"""
    entity_result = entity_result or {}
    entities = entity_result.get("merged_entities", [])
    if not entities:
        return {"triples": [], "paths": [], "extracted_entities": entity_result}

    # 1.5. 照合用エンティティの正規化
    # ノード側の search_keys（NFKC+小文字化、enrichmentでバックフィル済み）と
    # 同じ正規化を質問側エンティティにも適用して表記揺れミスマッチを防ぐ。
    # かな揺れ骨格キー（送り仮名・助詞の揺れ: 「ガス軸受け」→「ガス軸受」）も
    # 追加し、ノード側のかな揺れキーと双方向で照合できるようにする。
    # CONTAINS で過剰一致する1文字エンティティは除外。
    from graphrag_core.text.japanese import normalize_entity_text, kana_variant_key
    match_entities = []
    _seen_me = set()
    for e in entities:
        ne = normalize_entity_text(e)
        if len(ne) >= 2 and ne not in _seen_me:
            _seen_me.add(ne)
            match_entities.append(ne)
        kv = kana_variant_key(e)
        if kv and len(kv) >= 2 and kv not in _seen_me:
            _seen_me.add(kv)
            match_entities.append(kv)
    if not match_entities:
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
    max_start_nodes = 20  # CONTAINS過剰一致による起点ノード爆発の抑制
    # チャンクノード(hex id)を除外。値ノード(is_value: 数値・日付のみ)は
    # 「起点・中継としては禁止 / パスの終端(葉)としては許可」:
    # - 中継禁止: 「令和6年度」のような日付ハブ経由で無関係パスに飛ぶのを防ぐ
    # - 終端許可: スペック値・数値の出典チャンク取得(値ノード→MENTIONS)は
    #   数値質問の救済に効くため残す（完全排除したら数値系で退行した実測あり）
    # 照応ノード(is_anaphor: 「本製品」等の偽統合ハブ)は起点・中継・終端すべて禁止
    _hex = ("NOT n.id =~ '[0-9a-f]{32,}' AND COALESCE(n.is_value, false) = false "
            "AND COALESCE(n.is_anaphor, false) = false")
    _hex_node = ("NOT node.id =~ '[0-9a-f]{32,}' "
                 "AND COALESCE(node.is_anaphor, false) = false")
    # search_keys（正規化済みid+aliases+canonical_form）に対して照合。
    # 未バックフィルのグラフでは toLower(n.id) にフォールバック
    _key_match = "ANY(k IN COALESCE(n.search_keys, [toLower(n.id)]) WHERE k CONTAINS entity)"
    # 起点ノード選択: 完全一致を最優先し、次いでpagerank降順で上位に絞る。
    # 旧実装は無作為に collect していたため、汎用語の部分一致ノードが
    # 起点を占有してパス候補が関連度と無関係に LIMIT で切られていた。
    _start_select = (
        "WITH n, max(CASE WHEN ANY(k IN COALESCE(n.search_keys, [toLower(n.id)]) WHERE k = entity) "
        "THEN 1 ELSE 0 END) AS exact "
        "ORDER BY exact DESC, COALESCE(n.pagerank, 0.0) DESC "
        f"WITH collect(n)[0..{max_start_nodes}] AS matched_nodes "
        "UNWIND matched_nodes AS start_node "
    )

    if hop_count == 1:
        query = (
            "UNWIND $entities AS entity "
            "MATCH (n) WHERE " + _key_match + " AND " + _hex + " "
            + _start_select +
            "MATCH path = (start_node)-[r]-(end_node) "
            "WHERE type(r) <> '" + chunk_edge() + "' AND NOT end_node.id =~ '[0-9a-f]{32,}' "
            "AND COALESCE(end_node.is_anaphor, false) = false "
            # extraction_count（同一エッジが何チャンクから抽出されたか）と
            # pagerank でスコアリングしてから LIMIT（無作為truncationを排除）
            "WITH path, COALESCE(r.extraction_count, 1.0) AS rel_score, "
            "COALESCE(start_node.pagerank, 0.0) + COALESCE(end_node.pagerank, 0.0) AS pr_score "
            "ORDER BY rel_score DESC, pr_score DESC "
            "RETURN [node IN nodes(path) | node.id] AS node_ids, "
            "[rel IN relationships(path) | {start: startNode(rel).id, type: type(rel), end: endNode(rel).id}] AS rels "
            f"LIMIT {limit}"
        )
    else:
        query = (
            "UNWIND $entities AS entity "
            "MATCH (n) WHERE " + _key_match + " AND " + _hex + " "
            + _start_select +
            f"MATCH path = (start_node)-[*1..{hop_count}]-(end_node) "
            "WHERE ALL(r IN relationships(path) WHERE type(r) <> '" + chunk_edge() + "') "
            "AND ALL(node IN nodes(path) WHERE " + _hex_node + ") "
            # 値ノードは中継(先頭・末尾以外)に置かない
            "AND ALL(node IN nodes(path)[1..-1] WHERE COALESCE(node.is_value, false) = false) "
            "AND start_node <> end_node "
            # パス長で正規化した平均スコアで順位付け（長さ自体への加点はしない）
            "WITH path, size(relationships(path)) AS plen, "
            "reduce(s = 0.0, rel IN relationships(path) | s + COALESCE(rel.extraction_count, 1.0)) AS rel_sum, "
            "reduce(s = 0.0, nd IN nodes(path) | s + COALESCE(nd.pagerank, 0.0)) AS pr_sum "
            "ORDER BY rel_sum / plen DESC, pr_sum / (plen + 1) DESC "
            "RETURN [node IN nodes(path) | node.id] AS node_ids, "
            "[rel IN relationships(path) | {start: startNode(rel).id, type: type(rel), end: endNode(rel).id}] AS rels "
            f"LIMIT {limit}"
        )

    try:
        result = graph.query(query, params={"entities": match_entities})
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
                "MATCH (n) WHERE " + _key_match + " AND " + _hex + " "
                "WITH collect(DISTINCT n) AS matched_nodes "
                "UNWIND matched_nodes AS start_node "
                "MATCH (start_node)-[r]-(connected_node) "
                "WHERE type(r) <> '" + chunk_edge() + "' AND NOT connected_node.id =~ '[0-9a-f]{32,}' "
                "WITH r, startNode(r) AS actual_start, endNode(r) AS actual_end "
                "RETURN DISTINCT actual_start.id AS start, type(r) AS type, actual_end.id AS end "
                f"LIMIT {limit}"
            )
            result = graph.query(fb_query, params={"entities": match_entities})
            if result and len(result) > top_k:
                result = rank_relations_by_relevance(question, result, llm, top_k=top_k, doc_context=doc_context)
            return {"triples": result if result else [], "paths": [], "extracted_entities": entity_result}
    except Exception as e:
        logger.error("primary graph query failed: %s", e, exc_info=True)
        log.append(f"⚠️ グラフクエリ失敗: {type(e).__name__}: {e} → フォールバックへ")
        # フォールバック: 単純な1-hopマッチング
        fallback_query = """
        MATCH (n)-[r]->(m)
        WHERE ANY(entity IN $entities WHERE
            ANY(k IN COALESCE(n.search_keys, [toLower(n.id)]) WHERE k CONTAINS entity)
            OR ANY(k IN COALESCE(m.search_keys, [toLower(m.id)]) WHERE k CONTAINS entity))
        AND type(r) <> '""" + chunk_edge() + """'
        AND NOT n.id =~ '[0-9a-f]{32,}' AND NOT m.id =~ '[0-9a-f]{32,}'
        RETURN DISTINCT n.id AS start, type(r) AS type, m.id AS end
        LIMIT 20
        """
        try:
            result = graph.query(fallback_query, params={"entities": match_entities})
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

    # テーマ経路（関係キーワード索引 {collection}_relations）がコレクション名を
    # 必要とするため config 経由で伝搬する
    config = dict(config or {})
    config.setdefault("pg_collection", pg_collection)

    # ── Phase 1: エンティティ抽出 + ドキュメント検索を並列実行 ──
    def _search_documents():
        """ドキュメント検索（+ オプションのLLMリランク）"""
        # search_mode="none" = 文書検索なし（グラフのみモード）。
        # コンテキストはKGソースチャンク＋グラフ関係だけで構成される
        if _cfg(config, 'search_mode') == 'none':
            return []
        if _cfg(config, 'enable_japanese_search') and SUDACHI_AVAILABLE:
            try:
                hybrid_retriever = HybridRetriever.get_instance(pg_conn, collection_name=pg_collection)
                query_embedding = embeddings.embed_query(question)
                search_type = _cfg(config, 'search_mode')
                retrieval_top_k = _cfg(config, 'retrieval_top_k')
                enable_rerank = _cfg(config, 'enable_rerank')

                # リランク有効時は広めにプールを取り、リランカーで retrieval_top_k に絞る
                fetch_k = retrieval_top_k
                if enable_rerank:
                    fetch_k = max(retrieval_top_k, int(_cfg(config, 'rerank_pool_size') or 0))

                hybrid_results = hybrid_retriever.search(
                    query_text=question,
                    query_vector=query_embedding,
                    k=fetch_k,
                    search_type=search_type
                )

                if enable_rerank and len(hybrid_results) > retrieval_top_k:
                    hybrid_results = rerank_with_llm(
                        question, hybrid_results, llm, k=retrieval_top_k
                    )
                else:
                    hybrid_results = hybrid_results[:retrieval_top_k]

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

        # 旧実装は無順位の LIMIT 5 で任意のチャンクを注入していた。
        # 広めに取得してから質問との関連度でリランクして絞る。
        kg_chunk_top_k = int(_cfg(config, 'kg_chunk_top_k') or 5)
        kg_chunk_fetch = max(30, kg_chunk_top_k * 3)

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
                MATCH (a {id: tk.s})-[r]-(b {id: tk.o})
                WHERE type(r) = tk.p AND r.source_chunks IS NOT NULL
                UNWIND r.source_chunks AS cid
                MATCH (d:""" + chunk_label() + """ {id: cid})
                RETURN DISTINCT
                  d.id AS chunk_id,
                  substring(d.text, 0, 2000) AS text,
                  d.source AS source,
                  d.page AS page
                LIMIT $limit
                """
                chunk_results = graph.query(
                    cypher_strict, params={"triples": triple_keys, "limit": kg_chunk_fetch}
                ) or []
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
                        chunk_results = graph.get_source_chunks_list(entity_names, limit=kg_chunk_fetch)
                    else:
                        # 「言及している対象エンティティ数」が多いチャンク優先。
                        # 無順位LIMITだと、MENTIONSが集約されたハブエンティティの
                        # チャンク群から任意の数件が返り、候補プールが不安定になる
                        chunk_query = """
                        UNWIND $entity_names AS entity_name
                        MATCH (e {id: entity_name})<-[:""" + chunk_edge() + """]-(doc:""" + chunk_label() + """)
                        WITH doc, count(DISTINCT entity_name) AS entity_hits
                        ORDER BY entity_hits DESC
                        RETURN doc.id AS chunk_id,
                               substring(doc.text, 0, 2000) AS text,
                               doc.source AS source,
                               doc.page AS page
                        LIMIT $limit
                        """
                        chunk_results = graph.query(
                            chunk_query,
                            params={"entity_names": entity_names, "limit": kg_chunk_fetch},
                        ) or []
                except Exception:
                    chunk_results = []

        # ── 質問との関連度で絞り込み（reranker無効時は先頭から）──
        if len(chunk_results) > kg_chunk_top_k:
            from graphrag_core.retrieval.reranker import is_reranker_enabled, rerank_by_score
            if is_reranker_enabled():
                chunk_results = rerank_by_score(
                    question, chunk_results,
                    text_fn=lambda r: (r.get('text') or '')[:1000],
                    top_k=kg_chunk_top_k,
                )
            else:
                chunk_results = chunk_results[:kg_chunk_top_k]

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

    # 3.5. 参照追跡: ヒットチャンク中の「◯◯参照/ご覧ください」を1ホップ辿り、
    # 参照先チャンクをコンテキスト候補に加える（節/ページ参照は直接、
    # 文書名参照は参照先文書スコープの再検索に展開）。リランカーで質問関連度ゲート
    if graph is not None and docs and _cfg(config, 'enable_reference_follow'):
        try:
            from graphrag_core.graph.references import follow_references
            ref_results = follow_references(
                graph, docs, question,
                embeddings=embeddings, pg_conn=pg_conn, pg_collection=pg_collection,
            )
            if ref_results:
                ref_top_k = int(_cfg(config, 'reference_follow_top_k') or 3)
                if len(ref_results) > ref_top_k:
                    from graphrag_core.retrieval.reranker import is_reranker_enabled, rerank_by_score
                    if is_reranker_enabled():
                        ref_results = rerank_by_score(
                            question, ref_results,
                            text_fn=lambda r: (r.get('text') or '')[:1000],
                            top_k=ref_top_k,
                        )
                    else:
                        ref_results = ref_results[:ref_top_k]
                existing_texts = {d.page_content for d in docs} | {d.page_content for d in kg_chunks}
                for r in ref_results:
                    if r.get('text') and r['text'] not in existing_texts:
                        kg_chunks.append(Document(
                            page_content=r['text'],
                            metadata={
                                'id': r.get('chunk_id'),
                                'source': r.get('source', 'REF'),
                                'page': r.get('page'),
                                'via': f"reference({r.get('kind', 'ref')})",
                            }
                        ))
                        existing_texts.add(r['text'])
        except Exception as e:
            logger.warning("reference follow failed: %s", e)

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
    # ただし search_mode="none"（グラフのみモード）では文書検索が無いため、
    # 関係トリプル行を含めないとLLMがグラフを見られない → 強制的に含める
    if not _cfg(config, "include_graph_lines") and _cfg(config, "search_mode") != "none":
        graph_lines = []

    # 4.5 条件起点ルーティング（規程・基準系・既定OFF）。
    # OFF時は classifier も Cypher も走らず、context/result は従来と完全一致。
    condition_block = ""
    condition_facts = []
    if _cfg(config, "enable_condition_routing") and graph is not None:
        if classify_condition_intent(question) != "none":
            cinfo = get_condition_origin_context(question, graph, config, pg_collection)
            condition_facts = cinfo.get("facts", [])
            condition_block = cinfo.get("block", "")

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
    # 条件起点ブロックは最優先で先頭に置く（rows が空なら condition_block="" で不変）
    if condition_block:
        context = condition_block + "\n\n" + context

    result = {
        "context": context,
        "question": question,
        "vector_sources": docs,
        "kg_source_chunks": kg_chunks,
        "graph_sources": triples,
        "graph_paths": paths,
        "extracted_entities": extracted_entities
    }
    if condition_facts:
        result["condition_facts"] = condition_facts
    return result
