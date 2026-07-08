#!/usr/bin/env python
"""update_doc.py - 単一ドキュメントの部分グラフ更新 CLI
========================================================
改訂されたドキュメント1件を、フル再構築なしで反映する（approach D: 文書スコープ置換）。

- 差分 = 内容ハッシュID (sha256(doc_id+本文)) の集合差。不変チャンクは再抽出しない。
- グラフ剪定（edge.source_chunks / 孤立エンティティ）+ added のみ LLM 再抽出
- PGVector / エンティティベクトル / BM25(tokenized) も文書スコープで同期
- 後処理は既定 light（mention_count/search_keys のみ。pagerank と consolidate GC は
  定期フル再構築で実施）

既存グラフのID体系（位置ID等）と異なる文書への初回実行は「文書フル置換」になる
（以降の更新から差分になる）。

使用例:
    # 事前チャンク済みJSONL（chunk_id/doc_id/text/section_path 形式）で更新
    python scripts/update_doc.py --doc-id 46_kikai_kentei \
        --chunks-jsonl C:/data/chunks/46_kikai_kentei.jsonl --collection plant_v15

    # md/txt/pdf ファイルから再チャンクして更新
    python scripts/update_doc.py --doc-id 就業規則 --file ./docs/就業規則.md

    # 差分の見積りだけ（削除・抽出しない）
    python scripts/update_doc.py --doc-id X --chunks-jsonl ... --dry-run

    # 文書をグラフ/ベクトルから完全削除
    python scripts/update_doc.py --doc-id X --delete
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv()
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ── 入力ロード ────────────────────────────────────────────────────
def load_chunks_jsonl(path: Path, doc_id: str):
    """事前チャンク済みJSONL（makedataset chunker形式）を Document 群に変換。"""
    from langchain_core.documents import Document
    docs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        text = (r.get("text") or "").strip()
        if not text:
            continue
        sec = " > ".join(r.get("section_path") or [])
        docs.append(Document(
            page_content=(f"[{sec}]\n{text}" if sec else text),
            metadata={"source": doc_id, "page": r.get("page")},
        ))
    return docs


def load_file_chunks(path: Path, doc_id: str):
    """md/txt/pdf を読み込み、build_kg と同じ2段階チャンキングにかける。"""
    from langchain_core.documents import Document
    from graphrag_core.text.chunking import create_markdown_chunks
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from graphrag_core.document.pdf import load_pdf_text
        text = load_pdf_text(str(path))
    else:
        text = path.read_text(encoding="utf-8")
    src_doc = Document(page_content=text, metadata={"source": doc_id})
    return create_markdown_chunks([src_doc], chunk_size=1024, chunk_overlap=100)


# ── 抽出関数（build_kg_plant / fujitsu_build_kg と同じ流儀） ─────────
def build_add_chunk_fn(graph, llm):
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from graphrag_core.config import get_settings
    from graphrag_core.graph.schema import get_allowed_node_types, get_allowed_relations
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
        graph.query(
            "MATCH (d:Document {id: $id}) SET d.source = $src, d.page = $page",
            {"id": cid, "src": chunk.metadata.get("source"),
             "page": chunk.metadata.get("page")})
        return cid

    return add_chunk


def main():
    ap = argparse.ArgumentParser(description="単一ドキュメントの部分グラフ更新")
    ap.add_argument("--doc-id", required=True, help="文書ID（Document.source / metadata source）")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--chunks-jsonl", type=Path, help="事前チャンク済みJSONL")
    src.add_argument("--file", type=Path, help="md/txt/pdf（2段階チャンキングを適用）")
    ap.add_argument("--delete", action="store_true", help="この文書を完全削除する")
    ap.add_argument("--collection", default=None, help="PGコレクション（既定: .env の PG_COLLECTION）")
    ap.add_argument("--post", choices=["light", "full", "none"], default="light",
                    help="後処理: light=mention_count+search_keys / full=consolidate+参照+pagerank / none")
    ap.add_argument("--dry-run", action="store_true", help="差分と剪定見積りの表示のみ")
    ap.add_argument("--workers", type=int, default=4, help="抽出の並列数（既定4）")
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="LLMリクエストタイムアウト秒（既定90。永続失敗チャンクをfail-fast）")
    args = ap.parse_args()

    if not args.delete and not (args.chunks_jsonl or args.file):
        ap.error("--chunks-jsonl / --file のいずれか、または --delete が必要です")

    if args.collection:
        os.environ["PG_COLLECTION"] = args.collection
    from graphrag_core.config import reset_settings, get_settings
    reset_settings()
    s = get_settings()

    from langchain_neo4j import Neo4jGraph
    from graphrag_core.db.utils import add_connection_timeout
    from graphrag_core.graph.incremental import (
        compute_delta, prune_chunks, update_document, delete_document)

    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user,
                       password=s.neo4j_pw, enhanced_schema=False)
    pg_conn = add_connection_timeout(s.pg_conn, timeout=30) if s.pg_conn else None

    t0 = time.time()

    # ── 削除モード ──
    if args.delete:
        from graphrag_core.llm.factory import create_embeddings
        embeddings = create_embeddings() if pg_conn else None
        result = delete_document(graph, args.doc_id, pg_conn=pg_conn,
                                 pg_collection=s.pg_collection, embeddings=embeddings)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        print(f"完了 {time.time()-t0:.0f}s")
        return

    # ── 更新モード ──
    if args.chunks_jsonl:
        new_chunks = load_chunks_jsonl(args.chunks_jsonl, args.doc_id)
    else:
        new_chunks = load_file_chunks(args.file, args.doc_id)
    print(f"新チャンク: {len(new_chunks)}件 (doc={args.doc_id}, collection={s.pg_collection})")

    if args.dry_run:
        delta = compute_delta(graph, args.doc_id, new_chunks)
        est = prune_chunks(graph, delta["removed"], dry_run=True) if delta["removed"] else {}
        print(json.dumps({"added": len(delta["added"]), "removed": len(delta["removed"]),
                          "unchanged": len(delta["unchanged"]), "prune_estimate": est},
                         ensure_ascii=False, indent=2, default=str))
        return

    # 日本語トークン化（BM25用。added だけで良いが判定前なので全件・安価）
    if s.enable_japanese_search:
        from graphrag_core.text.japanese import get_japanese_processor
        jp = get_japanese_processor()
        if jp:
            for c in new_chunks:
                try:
                    c.metadata["tokenized_content"] = jp.tokenize(c.page_content)
                except Exception:
                    c.metadata["tokenized_content"] = None

    from graphrag_core.llm.factory import create_chat_llm, create_embeddings
    # timeout/max_retries は必ず factory 引数で渡す（初期化後の属性代入は
    # openai クライアントに伝播しない）。永続失敗チャンクは fail-fast し、
    # 次回実行で失敗分だけ再試行される。
    llm = create_chat_llm(temperature=0, timeout=args.timeout, max_retries=1)
    embeddings = create_embeddings()
    add_chunk_fn = build_add_chunk_fn(graph, llm)

    # 後処理
    def run_post():
        if args.post == "none":
            return
        if args.post == "light":
            from graphrag_core.graph.enrichment import enrich_post_update
            st = enrich_post_update(graph)
            print(f"post(light): {st}")
        else:  # full
            from graphrag_core.graph.consolidate import consolidate_post_build, resolve_anaphora_nodes
            from graphrag_core.graph.references import build_reference_graph
            from graphrag_core.graph.enrichment import enrich_post_build
            cstats = consolidate_post_build(graph)
            ref = build_reference_graph(graph)
            resolve_anaphora_nodes(graph, ref.get("alias_maps", {}))
            st = enrich_post_build(graph)
            print(f"post(full): consolidate={cstats.get('duplicate_merge')} enrich={st}")
        try:
            from graphrag_core.graph.provenance import stamp_graph_provenance
            stamp_graph_provenance(graph, s.pg_collection)
        except Exception:
            pass

    result = update_document(
        graph, args.doc_id, new_chunks, add_chunk_fn,
        pg_conn=pg_conn, pg_collection=s.pg_collection, embeddings=embeddings,
        run_post=run_post, workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if result.get("add_failed_ids"):
        print(f"⚠️ 抽出失敗 {len(result['add_failed_ids'])}件: 再実行すれば失敗分だけ再試行されます")
    print(f"完了 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
