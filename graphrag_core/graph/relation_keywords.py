"""エッジへのテーマキーワード注釈と関係ベクトル索引（LightRAG dual-level の高レベル側）。

質問が固有名を含まないテーマ型（「ボイラ設計における注意点は？」）のとき、
エンティティ照合（低レベル経路）は空振りする。本モジュールは各エッジに
「この関係が扱う話題」を表すテーマ語を付与・索引化し、質問の高レベル
キーワードから関係を直接引けるようにする（検索側は pipeline のテーマ経路）。

構築済みグラフへ後付けで実行できる（再抽出・再ビルド不要）:
1. annotate_relation_keywords: エンティティ間エッジに テーマ語2〜4個 をLLM付与
   （r.keywords）。抽出元チャンク冒頭を文脈としてプロンプトに含める
2. build_relation_vector_index: keywords＋トリプル文を埋め込み
   PGVector `{collection}_relations` へ索引化
3. search_relations_by_theme: 高レベルキーワード → 類似エッジ（検索時）

実行: scripts/build_relation_keywords.py（グラフ再構築後にも再実行すること）
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_ANNOTATE_PROMPT = """ナレッジグラフの各「関係」に、検索用のテーマキーワードを付与します。
キーワードは「この関係が扱う話題・概念」を表す日本語の名詞句を2〜4個。
固有名詞の繰り返しではなく、話題として検索されそうな語を選んでください
（例: 安全弁の整定なら「安全装置, 整定, 圧力管理」）。

関係リスト:
{lines}

出力はJSON配列のみ（説明不要）:
[{{"i": 1, "kw": ["...", "..."]}}, ...]"""


def _parse_json_array(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t.strip())
    m = re.search(r"\[.*\]", t, re.DOTALL)
    return json.loads(m.group(0) if m else t)


def _chunk_snippets(pg_conn: str, chunk_ids: List[str], width: int = 140) -> Dict[str, str]:
    """抽出元チャンクの冒頭を一括取得（注釈プロンプトの文脈用）。"""
    ids = [c for c in chunk_ids if c]
    if not ids or not pg_conn:
        return {}
    try:
        import psycopg
        from graphrag_core.db.utils import normalize_pg_connection_string
        with psycopg.connect(normalize_pg_connection_string(pg_conn)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT COALESCE(e.cmetadata->>'id', e.id), LEFT(e.document, %s)
                       FROM langchain_pg_embedding e
                       WHERE COALESCE(e.cmetadata->>'id', e.id) = ANY(%s)""",
                    (width, ids))
                return {r[0]: r[1] for r in cur.fetchall()}
    except Exception as e:
        logger.warning("chunk snippet fetch failed (文脈なしで続行): %s", e)
        return {}


def annotate_relation_keywords(graph, llm, pg_conn: Optional[str] = None, *,
                               batch_size: int = 20,
                               limit: Optional[int] = None,
                               protect_label_prefix: str = "Qms",
                               progress: Optional[Callable] = None,
                               should_cancel: Optional[Callable[[], bool]] = None) -> Dict:
    """keywords未付与のエンティティ間エッジにテーマ語をLLMで付与する（冪等）。"""
    from graphrag_core.graph.schema import chunk_edge

    guard = (
        f"AND NOT any(l IN labels(a) WHERE l STARTS WITH '{protect_label_prefix}') "
        f"AND NOT any(l IN labels(b) WHERE l STARTS WITH '{protect_label_prefix}') "
    ) if protect_label_prefix else ""
    rows = graph.query(
        f"""MATCH (a)-[r]->(b)
        WHERE type(r) <> '{chunk_edge()}' AND r.keywords IS NULL
          AND NOT a.id =~ '[0-9a-f]{{32,}}' AND NOT b.id =~ '[0-9a-f]{{32,}}'
          {guard}
        RETURN elementId(r) AS eid, a.id AS s, type(r) AS t, b.id AS o,
               COALESCE(r.source_chunks, [])[0] AS chunk_id
        {f'LIMIT {int(limit)}' if limit else ''}"""
    ) or []
    if not rows:
        return {"annotated": 0, "failed_batches": 0, "total": 0}

    annotated = 0
    failed = 0
    n_batches = (len(rows) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        if should_cancel and should_cancel():
            break
        batch = rows[bi * batch_size:(bi + 1) * batch_size]
        snippets = _chunk_snippets(pg_conn, [r["chunk_id"] for r in batch])
        lines = []
        for i, r in enumerate(batch, 1):
            ctx = snippets.get(r["chunk_id"], "")
            ctx_part = f" ｜ 出典: 「{ctx}…」" if ctx else ""
            lines.append(f'{i}. {r["s"]} -[{r["t"]}]-> {r["o"]}{ctx_part}')
        if progress:
            progress(bi + 1, n_batches, annotated)
        try:
            out = llm.invoke(_ANNOTATE_PROMPT.format(lines="\n".join(lines))).content
            items = _parse_json_array(out)
            updates = []
            for item in items:
                idx = int(item.get("i", 0)) - 1
                kw = [str(k).strip() for k in (item.get("kw") or []) if str(k).strip()]
                if 0 <= idx < len(batch) and kw:
                    updates.append({"eid": batch[idx]["eid"], "kw": kw[:4]})
            if updates:
                graph.query(
                    """UNWIND $rows AS row
                    MATCH ()-[r]->() WHERE elementId(r) = row.eid
                    SET r.keywords = row.kw""",
                    {"rows": updates})
                annotated += len(updates)
        except Exception as e:
            failed += 1
            logger.warning("annotate batch %d/%d failed: %s", bi + 1, n_batches, e)
    return {"annotated": annotated, "failed_batches": failed, "total": len(rows)}


def relation_collection_name(pg_collection: str) -> str:
    return f"{pg_collection}_relations"


def build_relation_vector_index(graph, embeddings, pg_conn: str, pg_collection: str, *,
                                protect_label_prefix: str = "Qms",
                                batch: int = 128) -> Dict:
    """keywords付きエッジを `{collection}_relations` へ埋め込み索引化（全消し再構築）。"""
    from langchain_core.documents import Document
    from langchain_postgres import PGVector
    from graphrag_core.graph.schema import chunk_edge

    guard = (
        f"AND NOT any(l IN labels(a) WHERE l STARTS WITH '{protect_label_prefix}') "
        f"AND NOT any(l IN labels(b) WHERE l STARTS WITH '{protect_label_prefix}') "
    ) if protect_label_prefix else ""
    rows = graph.query(
        f"""MATCH (a)-[r]->(b)
        WHERE type(r) <> '{chunk_edge()}' AND r.keywords IS NOT NULL
          AND NOT a.id =~ '[0-9a-f]{{32,}}' AND NOT b.id =~ '[0-9a-f]{{32,}}'
          {guard}
        RETURN elementId(r) AS eid, a.id AS s, type(r) AS t, b.id AS o,
               r.keywords AS kw"""
    ) or []
    store = PGVector(
        connection=pg_conn, embeddings=embeddings,
        collection_name=relation_collection_name(pg_collection),
        use_jsonb=True, pre_delete_collection=True,
    )
    if not rows:
        return {"indexed": 0}
    docs, ids = [], []
    for r in rows:
        kw = "、".join(r["kw"] or [])
        docs.append(Document(
            page_content=f"{kw}。{r['s']} {r['t']} {r['o']}",
            metadata={"start": r["s"], "type": r["t"], "end": r["o"]},
        ))
        ids.append(r["eid"])
    for i in range(0, len(docs), batch):
        store.add_documents(docs[i:i + batch], ids=ids[i:i + batch])
    return {"indexed": len(docs)}


def search_relations_by_theme(pg_conn: str, pg_collection: str, embeddings,
                              theme_keywords: List[str], *,
                              k: int = 8, min_similarity: float = 0.30) -> List[Dict]:
    """高レベルキーワードで関係索引を検索し、トリプル dict のリストを返す。

    Returns: [{"start", "type", "end", "theme_score"}]（類似度降順）
    """
    from langchain_postgres import PGVector
    if not theme_keywords:
        return []
    store = PGVector(
        connection=pg_conn, embeddings=embeddings,
        collection_name=relation_collection_name(pg_collection),
        use_jsonb=True,
    )
    query = "、".join(theme_keywords)
    results = store.similarity_search_with_score(query, k=k)
    out = []
    for doc, distance in results:
        sim = 1.0 - float(distance)
        if sim < min_similarity:
            continue
        m = doc.metadata or {}
        if m.get("start") and m.get("end"):
            out.append({"start": m["start"], "type": m.get("type", "RELATED_TO"),
                        "end": m["end"], "theme_score": round(sim, 3)})
    return out
