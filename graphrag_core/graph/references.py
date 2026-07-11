"""文書内・文書間の参照関係グラフ（ルールベース、LLM不要）

法令文の参照抽出手法（北野・天笠, NLP2026: パターンベースでF値0.935）を
マニュアル・レポートコーパスに翻案したもの。法令の「法令名・条項番号・略称」を
「文書名・節番号/ページ番号・略称定義」に対応させる:

  法令名の検索        → 文書タイトル照合（『内蔵無線WANをお使いになる方へ』をご覧ください）
  条項番号            → 節番号「3.3.1」・ページ番号「（→P.92）」
  略称定義（以下「法」という） → 「（以下「本公開買付者」という）」
  範囲（乃至/から〜まで）     → 「P.13〜18」

生成物:
  1. (chunk)-[:REFERS_TO {kind, ref_text}]->(chunk)  … 節・ページ参照
  2. chunk.ref_docs = [source, ...]                   … 文書名参照（検索時に
     参照先文書スコープの再検索に展開する。文書→特定チャンクは静的に決められないため）
  3. alias_maps: {source: {略称: 正式名称}}            … consolidate の照応解決へ渡す

検索時の利用は pipeline.follow_references() を参照。
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List

from graphrag_core.graph.schema import chunk_label

logger = logging.getLogger(__name__)

# ── 正規表現パターン ──────────────────────────────────────────────────

# 節見出し（チャンク本文の行頭）: 「3.3.1 注意事項」
_HEADING_RX = re.compile(r"(?m)^\s*((?:[0-9]+\.){1,3}[0-9]+)\s*[ 　]?\S")

# 略称定義: 「（以下「X」という）」「（以下、X）」
_ALIAS_RX = re.compile(
    r"[（(]以下[、，]?\s*[「『]?([^」』）)、，\n]{2,25})[」』]?\s*(?:という|と称する|と呼ぶ|と記載|と略す)?[）)]")

# 文書名参照: 『X』をご覧ください/を参照
# PDF抽出テキストはタイトル途中で改行が入ることがあるため 『』内の改行を許容する
# （照合は _norm() で空白・改行を除去して行う）
_DOC_REF_RX = re.compile(r"[『]([^』]{3,40})[』][\s\S]{0,12}?(?:をご覧|を参照|参照して)")

# ページ参照: 「（→P.92）」「P.92」「92ページ」+ 範囲「P.13〜18」
_PAGE_RANGE_RX = re.compile(r"P\.?\s?([0-9]{1,3})\s?[〜~～-]\s?(?:P\.?\s?)?([0-9]{1,3})")
_PAGE_RX = re.compile(r"(?:→\s?P\.?\s?([0-9]{1,3})|P\.?\s?([0-9]{1,3})|([0-9]{1,3})\s?ページ)")

# 節参照: 「3.3 nanoSIMカード」（→…）/ 3.3.1をご覧 + 範囲「3.1〜3.3」
_SEC_RANGE_RX = re.compile(r"((?:[0-9]+\.)+[0-9]+)\s?[〜~～]\s?((?:[0-9]+\.)+[0-9]+)")
_SEC_REF_RX = re.compile(
    r"[「]?((?:[0-9]+\.){1,3}[0-9]+)[^」\n]{0,25}[」]?[^\n]{0,10}?(?:をご覧|を参照|参照|をお読み|→)")

# タイトル候補行（文書冒頭ページから）: 4〜30字、句点なし、数字始まりでない。
# 「内蔵無線WAN をお使いになる方へ」のように内部スペースを含むタイトルを許容する
# （照合時は _norm() で空白を除去するため問題ない）
_TITLE_LINE_RX = re.compile(r"^(?![0-9０-９])[^。、．]{4,30}$")

# 参照表現として意味を持つ語の近傍チェック用
_MAX_REFS_PER_CHUNK = 12


def _norm(s: str) -> str:
    """タイトル照合用の正規化（NFKC + 小文字化 + 空白除去）"""
    return "".join(unicodedata.normalize("NFKC", s or "").lower().split())


# ── Phase 0: インベントリ構築 ─────────────────────────────────────────

def load_chunks(graph) -> List[dict]:
    """Neo4jのDocumentチャンクを文書順で全件取得（SKIP/LIMITページング）"""
    chunks: List[dict] = []
    skip, batch = 0, 2000
    while True:
        rows = graph.query(
            "MATCH (d:" + chunk_label() + ") WHERE d.id =~ '[0-9a-f]{32,}' AND d.text IS NOT NULL "
            "RETURN d.id AS id, d.text AS text, d.source AS source, d.page AS page "
            "ORDER BY d.source, d.page, d.id SKIP $skip LIMIT $batch",
            {"skip": skip, "batch": batch},
        )
        if not rows:
            break
        chunks.extend(rows)
        if len(rows) < batch:
            break
        skip += batch
    return chunks


def build_inventory(chunks: List[dict]) -> dict:
    """文書ごとの節索引・ページ索引・タイトル索引・略称マップを構築"""
    section_index: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    page_index: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    title_map: Dict[str, str] = {}           # 正規化タイトル → source
    alias_maps: Dict[str, Dict[str, str]] = defaultdict(dict)
    first_page: Dict[str, Any] = {}

    for c in chunks:
        src, pg = c["source"], c["page"]
        if src is None:
            continue
        page_index[src].setdefault(str(pg), []).append(c["id"])
        # 節見出し（本文前半に出る想定）
        for sec in _HEADING_RX.findall((c["text"] or "")[:800]):
            section_index[src].setdefault(sec, []).append(c["id"])
        if src not in first_page or (pg is not None and pg < first_page[src]):
            first_page[src] = pg

    for c in chunks:
        src, pg, text = c["source"], c["page"], c["text"] or ""
        if src is None:
            continue
        # タイトル候補: 文書の先頭2ページの短い独立行
        if pg is not None and first_page.get(src) is not None and pg <= first_page[src] + 1:
            for line in text.splitlines()[:30]:
                line = line.strip()
                if _TITLE_LINE_RX.match(line) and len(_norm(line)) >= 4:
                    title_map.setdefault(_norm(line), src)
        # 略称定義
        for m in _ALIAS_RX.finditer(text):
            alias = m.group(1).strip().rstrip("という").strip()
            if not alias or len(alias) < 2:
                continue
            # 正式名称: 定義の直前テキストを境界記号で切った最長スパン（最大40字）
            head = text[max(0, m.start() - 40): m.start()]
            formal = re.split(r"[、。（）「」『』\n]", head)[-1].strip()
            if len(formal) >= 3 and formal != alias:
                alias_maps[src].setdefault(alias, formal)

    return {
        "section_index": dict(section_index),
        "page_index": dict(page_index),
        "title_map": title_map,
        "alias_maps": {k: dict(v) for k, v in alias_maps.items()},
    }


# ── Phase 1: 参照抽出と解決 ──────────────────────────────────────────

def extract_references(chunks: List[dict], inv: dict) -> dict:
    """全チャンクから参照を抽出し、解決済みエッジと文書参照を返す

    Returns:
        {"edges": [(src_chunk, tgt_chunk, kind, ref_text), ...],
         "doc_refs": {chunk_id: [target_source, ...]},
         "stats": {...}}
    """
    edges = []
    doc_refs: Dict[str, List[str]] = {}
    n_unresolved = 0

    for c in chunks:
        src, cid, text = c["source"], c["id"], c["text"] or ""
        if src is None or not text:
            continue
        sec_idx = inv["section_index"].get(src, {})
        pg_idx = inv["page_index"].get(src, {})
        chunk_edges = []

        # 1. 文書名参照（→ ref_docs プロパティ。検索時にスコープ再検索へ展開）
        targets = []
        for m in _DOC_REF_RX.finditer(text):
            tgt_src = inv["title_map"].get(_norm(m.group(1)))
            if tgt_src and tgt_src != src and tgt_src not in targets:
                targets.append(tgt_src)
        if targets:
            doc_refs[cid] = targets[:3]

        # 2. ページ範囲参照（P.13〜18）→ 展開
        consumed_spans = []
        for m in _PAGE_RANGE_RX.finditer(text):
            consumed_spans.append(m.span())
            p1, p2 = int(m.group(1)), int(m.group(2))
            if 0 < p2 - p1 <= 10:
                for p in range(p1, p2 + 1):
                    for tid in pg_idx.get(str(p), []):
                        if tid != cid:
                            chunk_edges.append((cid, tid, "page", m.group(0)))

        # 3. 単一ページ参照
        for m in _PAGE_RX.finditer(text):
            if any(s <= m.start() < e for s, e in consumed_spans):
                continue
            pno = m.group(1) or m.group(2) or m.group(3)
            tids = pg_idx.get(pno, [])
            if not tids:
                n_unresolved += 1
                continue
            for tid in tids:
                if tid != cid:
                    chunk_edges.append((cid, tid, "page", f"P.{pno}"))

        # 4. 節範囲参照（3.1〜3.3）→ 前方一致で展開
        for m in _SEC_RANGE_RX.finditer(text):
            s1, s2 = m.group(1), m.group(2)
            for sec, tids in sec_idx.items():
                if s1 <= sec <= s2:
                    for tid in tids:
                        if tid != cid:
                            chunk_edges.append((cid, tid, "section", m.group(0)))

        # 5. 節参照
        for m in _SEC_REF_RX.finditer(text):
            sec = m.group(1)
            tids = sec_idx.get(sec, [])
            if not tids:
                n_unresolved += 1
                continue
            for tid in tids:
                if tid != cid:
                    chunk_edges.append((cid, tid, "section", sec))

        # 重複除去 + チャンクあたり上限
        seen = set()
        for e in chunk_edges:
            key = (e[0], e[1], e[2])
            if key not in seen:
                seen.add(key)
                edges.append(e)
                if len(seen) >= _MAX_REFS_PER_CHUNK:
                    break

    stats = {
        "edges": len(edges),
        "doc_ref_chunks": len(doc_refs),
        "unresolved": n_unresolved,
    }
    return {"edges": edges, "doc_refs": doc_refs, "stats": stats}


# ── エッジ書き込み ────────────────────────────────────────────────────

def write_reference_graph(graph, extraction: dict, batch_size: int = 1000) -> dict:
    """REFERS_TOエッジと ref_docs プロパティをNeo4jへ書き込む（冪等）"""
    edges = extraction["edges"]
    written = 0
    for i in range(0, len(edges), batch_size):
        batch = [
            {"src": s, "tgt": t, "kind": k, "ref": r}
            for s, t, k, r in edges[i: i + batch_size]
        ]
        graph.query(
            """
            UNWIND $rows AS row
            MATCH (a:""" + chunk_label() + """ {id: row.src})
            MATCH (b:""" + chunk_label() + """ {id: row.tgt})
            MERGE (a)-[r:REFERS_TO {kind: row.kind}]->(b)
            SET r.ref_text = row.ref
            """,
            {"rows": batch},
        )
        written += len(batch)

    doc_rows = [{"id": cid, "docs": docs} for cid, docs in extraction["doc_refs"].items()]
    for i in range(0, len(doc_rows), batch_size):
        graph.query(
            """
            UNWIND $rows AS row
            MATCH (c:""" + chunk_label() + """ {id: row.id})
            SET c.ref_docs = row.docs
            """,
            {"rows": doc_rows[i: i + batch_size]},
        )

    logger.info("write_reference_graph: %d edges, %d doc-ref chunks", written, len(doc_rows))
    return {"edges_written": written, "doc_ref_chunks": len(doc_rows)}


# ── 検索時: 参照追跡 ──────────────────────────────────────────────────

def follow_references(
    graph,
    docs: List[Any],
    question: str,
    embeddings=None,
    pg_conn: str = None,
    pg_collection: str = "graphrag",
    per_doc_k: int = 3,
    limit: int = 30,
) -> List[dict]:
    """検索ヒットチャンクから参照を1ホップ辿り、参照先チャンク候補を返す。

    - 節・ページ参照（REFERS_TOエッジ）→ 参照先チャンクを直接取得
    - 文書名参照（ref_docs プロパティ）→ 参照先文書スコープでベクトル再検索
      （文書→特定チャンクは静的に決まらないため、質問で絞り込む）

    PGとNeo4jのチャンクIDは別体系のため、(source, page) で対応付ける。

    Returns:
        [{"chunk_id", "text", "source", "page", "kind"}, ...]
    """
    locs = []
    seen_locs = set()
    for d in docs:
        m = getattr(d, "metadata", None) or {}
        src, pg = m.get("source"), m.get("page")
        if src is None or pg is None:
            continue
        key = (src, str(pg))
        if key not in seen_locs:
            seen_locs.add(key)
            locs.append({"source": src, "page": str(pg)})
    if not locs:
        return []

    edge_candidates: List[dict] = []
    doc_candidates: List[dict] = []
    ref_doc_sources: List[str] = []
    try:
        rows = graph.query(
            """
            UNWIND $locs AS loc
            MATCH (c:""" + chunk_label() + """)
            WHERE c.source = loc.source AND toString(c.page) = loc.page
            OPTIONAL MATCH (c)-[r:REFERS_TO]->(t:""" + chunk_label() + """)
            RETURN collect(DISTINCT {
                       id: t.id, text: substring(t.text, 0, 2000),
                       source: t.source, page: t.page, kind: r.kind
                   }) AS tgts,
                   collect(DISTINCT c.ref_docs) AS ref_docs_list
            """,
            {"locs": locs},
        )
        for row in rows:
            for t in row.get("tgts") or []:
                if t and t.get("id"):
                    edge_candidates.append({
                        "chunk_id": t["id"], "text": t.get("text"),
                        "source": t.get("source"), "page": t.get("page"),
                        "kind": t.get("kind", "ref"),
                    })
            for rd in row.get("ref_docs_list") or []:
                for s in rd or []:
                    if s and s not in ref_doc_sources:
                        ref_doc_sources.append(s)
    except Exception as e:
        logger.warning("follow_references graph query failed: %s", e)
        return []

    # 文書名参照: 参照先文書に絞ったベクトル再検索。
    # メイン検索で取得済みのページと重複しやすいため、広めに取得して
    # 既出ページを除いた上位 per_doc_k 件を新規候補とする
    if ref_doc_sources and embeddings is not None and pg_conn:
        try:
            import psycopg
            from graphrag_core.db.utils import normalize_pg_connection_string
            qvec = embeddings.embed_query(question)
            conn_str = normalize_pg_connection_string(pg_conn)
            with psycopg.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    for tgt_src in ref_doc_sources[:3]:
                        cur.execute(
                            """
                            SELECT e.id, e.document, e.cmetadata
                            FROM langchain_pg_embedding e
                            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                            WHERE c.name = %s AND e.cmetadata->>'source' = %s
                            ORDER BY e.embedding <=> %s::vector
                            LIMIT %s
                            """,
                            (pg_collection, tgt_src, qvec, per_doc_k * 4),
                        )
                        added = 0
                        for cid, text, meta in cur.fetchall():
                            pg_no = (meta or {}).get("page")
                            if (tgt_src, str(pg_no)) in seen_locs:
                                continue
                            doc_candidates.append({
                                "chunk_id": cid, "text": text,
                                "source": (meta or {}).get("source", tgt_src),
                                "page": pg_no,
                                "kind": "doc",
                            })
                            added += 1
                            if added >= per_doc_k:
                                break
        except Exception as e:
            logger.warning("follow_references scoped search failed: %s", e)

    # doc参照（文書スコープ再検索の結果）を優先し、エッジ候補を続ける。
    # エッジ候補は1ヒットチャンクあたり最大12本×ヒット数で limit を
    # 食い潰しやすく、後置だとdoc候補が刈られるため。
    # ヒット元と同一(source,page)・重複chunk_idは除外
    out = []
    seen_ids = set()
    for c in doc_candidates + edge_candidates:
        cid = c.get("chunk_id")
        if cid in seen_ids:
            continue
        if (c.get("source"), str(c.get("page"))) in seen_locs:
            continue
        # ヘッダ行だけの極小サブチャンク（ページ冒頭の文書名のみ等）は情報がない
        if len((c.get("text") or "").strip()) < 80:
            continue
        seen_ids.add(cid)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def build_reference_graph(graph) -> dict:
    """インベントリ構築→参照抽出→書き込みの一括実行。alias_mapsを返す"""
    chunks = load_chunks(graph)
    logger.info("build_reference_graph: %d chunks loaded", len(chunks))
    inv = build_inventory(chunks)
    logger.info("inventory: %d docs sections, %d titles, %d docs with aliases",
                len(inv["section_index"]), len(inv["title_map"]), len(inv["alias_maps"]))
    extraction = extract_references(chunks, inv)
    logger.info("extraction: %s", extraction["stats"])
    write_stats = write_reference_graph(graph, extraction)
    return {
        "chunks": len(chunks),
        "titles": len(inv["title_map"]),
        "alias_maps": inv["alias_maps"],
        **extraction["stats"],
        **write_stats,
    }
