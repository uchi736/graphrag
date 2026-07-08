#!/usr/bin/env python
"""plant_v15 コーパス（事前チャンク済み）を Neo4j にKG構築する。

fujitsu_build_kg.py の plant 版。
- 入力: C:/work/makedataset/data/chunks_plant/*.jsonl（chunk_id/doc_id/text/section_path）
- Document ノードは id=chunk_id でキー（dataset の relation_edges / 参照グラフと整合）
- LLMGraphTransformer で Term/関係抽出 + edge に source_chunks 付与
- 後処理: consolidate → 参照グラフ(REFERS_TO) → enrichment → provenance刻印(plant_v15)
- ★ PGVector(plant_v15) には一切触れない（ingest_plant.py が投入済み）

Usage:
    python _bench/plant/build_kg_plant.py --fresh              # Neo4j全クリアして全チャンク
    python _bench/plant/build_kg_plant.py --fresh --limit 15   # スモークテスト
    python _bench/plant/build_kg_plant.py --resume             # 既処理スキップで続き
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import re
import sys
import time
import unicodedata as _ud
from pathlib import Path

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from dotenv import load_dotenv
load_dotenv()

# Windows cp932 コンソール/リダイレクトでも絵文字でクラッシュしないよう utf-8 化
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CHUNKS_DIR = Path("C:/work/makedataset/data/chunks_plant")


def load_plant_docs(limit=None):
    from langchain_core.documents import Document
    docs = []
    for fp in sorted(glob.glob(str(CHUNKS_DIR / "*.jsonl"))):
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = (r.get("text") or "").strip()
            cid = r.get("chunk_id")
            if not text or not cid:
                continue
            sec = " > ".join(r.get("section_path") or [])
            docs.append(Document(
                page_content=(f"[{sec}]\n{text}" if sec else text),
                metadata={
                    "id": cid,
                    "source": r.get("doc_id"),
                    "page": r.get("page"),
                },
            ))
    if limit:
        docs = docs[:limit]
    return docs


def _normalize_id(name, has_neologdn, neologdn_mod):
    if not name:
        return name
    s = _ud.normalize("NFKC", str(name))
    if has_neologdn:
        s = neologdn_mod.normalize(s)
    s = re.sub(r"\s+", "", s).strip()
    return s or str(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="Neo4j を MATCH (n) DETACH DELETE n で全削除してから始める")
    ap.add_argument("--resume", action="store_true", help="既処理(ProcessedChunk)をスキップして続行")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("KG_BUILD_WORKERS", "6")))
    ap.add_argument("--limit", type=int, default=None, help="先頭N チャンクのみ（スモーク用）")
    ap.add_argument("--no-post", action="store_true", help="後処理(consolidate/参照/enrich)をスキップ")
    ap.add_argument("--chunks-dir", default=None, help="チャンクJSONLディレクトリ（既定: chunks_plant）")
    ap.add_argument("--provenance", default="plant_v15", help="provenance刻印するPGコレクション名")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="LLMタイムアウト秒（高密度列挙チャンクは抽出出力が長く300-600s要ることがある）")
    args = ap.parse_args()
    if args.chunks_dir:
        global CHUNKS_DIR
        CHUNKS_DIR = Path(args.chunks_dir)

    docs = load_plant_docs(limit=args.limit)
    print(f"=== plant KG builder === chunks={len(docs)} fresh={args.fresh} workers={args.workers}")

    from graphrag_core.config import reset_settings, get_settings
    reset_settings()
    s = get_settings()
    from langchain_neo4j import Neo4jGraph
    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw, enhanced_schema=True)
    print(f"  Neo4j: {s.neo4j_uri}  LLM: {s.llm_provider}  schema: {os.getenv('SHARED_SCHEMA_PATH')}")

    if args.fresh:
        print("🗑️  Neo4j 全削除中...")
        graph.query("MATCH (n) DETACH DELETE n")
        print("  完了")

    processed = graph.query("MATCH (c:ProcessedChunk) RETURN c.hash AS hash")
    processed_hashes = {r["hash"] for r in processed} if processed else set()
    pending = [d for d in docs if d.metadata["id"] not in processed_hashes]
    print(f"  pending: {len(pending)} / total: {len(docs)} (skipped: {len(docs)-len(pending)})")

    if pending:
        from langchain_experimental.graph_transformers import LLMGraphTransformer
        from graphrag_core.llm.factory import create_chat_llm
        from graphrag_core.graph.schema import get_allowed_node_types, get_allowed_relations
        from graphrag_core.graph.enrichment import attach_source_chunks

        try:
            import neologdn as _neologdn
            has_neo = True
        except Exception:
            _neologdn = None
            has_neo = False
            print("⚠️  neologdn 未インストール → NFKCのみ")

        # timeout は factory 引数で渡す（初期化後の属性代入はクライアントに伝播しない）
        llm = create_chat_llm(temperature=0, timeout=args.timeout, max_retries=2)
        additional = (
            "抽出する: 技術用語、概念、固有名詞、プロセス名、規格名、組織、製品。"
            "抽出しない: 一般的な名詞（『こと』『もの』『方法』）、代名詞、動詞。"
            "抽出しない: 数値・日付・単位のみの値。値はノードにしない。"
            "RELATED_TOは他に適切な関係がない場合の最終手段として使用。"
        )
        is_vllm = s.llm_provider.lower() == "vllm"
        transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=get_allowed_node_types(),
            allowed_relationships=get_allowed_relations(),
            strict_mode=False,
            ignore_tool_usage=is_vllm,
            additional_instructions=additional,
        )

        def process_chunk(doc):
            chunk_docs = transformer.convert_to_graph_documents([doc])
            for gd in chunk_docs:
                for node in gd.nodes:
                    orig = node.id
                    node.id = _normalize_id(orig, has_neo, _neologdn)
                    if orig != node.id:
                        props = getattr(node, "properties", None) or {}
                        ex = props.get("surface_forms", [])
                        if isinstance(ex, str):
                            ex = [ex]
                        if orig not in ex:
                            props["surface_forms"] = ex + [orig]
                        node.properties = props
                for rel in gd.relationships:
                    rel.source.id = _normalize_id(rel.source.id, has_neo, _neologdn)
                    rel.target.id = _normalize_id(rel.target.id, has_neo, _neologdn)
            graph.add_graph_documents(chunk_docs, include_source=True)
            cid = doc.metadata["id"]
            attach_source_chunks(graph, chunk_docs, cid)
            graph.query("MERGE (c:ProcessedChunk {hash: $h}) SET c.processed_at = datetime()", {"h": cid})
            graph.query(
                "MATCH (d:Document {id: $id}) SET d.source = $src, d.page = $page",
                {"id": cid, "src": doc.metadata["source"], "page": doc.metadata["page"]},
            )
            return cid

        print(f"🕸️ KG構築: {len(pending)} chunks × {args.workers} workers "
              f"(見積 {len(pending)*8/args.workers/60:.0f}分 @8s/chunk)")
        success, error = 0, 0
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_chunk, d): d for d in pending}
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                try:
                    fut.result(); success += 1
                except Exception as e:
                    error += 1
                    if error <= 8:
                        print(f"  ⚠️ err: {type(e).__name__}: {str(e)[:100]}")
                done += 1
                if done % 50 == 0 or done == len(pending):
                    avg = (time.time()-t0)/done
                    eta = avg*(len(pending)-done)/60
                    print(f"  {done}/{len(pending)} ok={success} err={error} "
                          f"({avg:.1f}s/chunk, ETA {eta:.0f}m)", flush=True)
        print(f"✅ 抽出完了: ok={success} err={error} ({time.time()-t0:.0f}s)")

    nc = graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
    rc = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    print(f"📊 グラフ: nodes={nc} rels={rc}")

    if args.no_post:
        print("--no-post: 後処理スキップ")
        return

    print("🔧 consolidate...")
    try:
        from graphrag_core.graph.consolidate import consolidate_post_build, resolve_anaphora_nodes
        cstats = consolidate_post_build(graph)
        print(f"  値ノードflag={cstats['value_nodes_flagged']} 型分裂={cstats['duplicate_merge']['removed_nodes']} "
              f"かな={cstats['kana_variant_merge']['removed_nodes']}")
    except Exception as e:
        print(f"  ⚠️ consolidate err: {e}")

    print("🔗 参照グラフ...")
    try:
        from graphrag_core.graph.references import build_reference_graph
        from graphrag_core.graph.consolidate import resolve_anaphora_nodes
        ref = build_reference_graph(graph)
        print(f"  参照エッジ={ref['edges_written']} 文書名参照={ref['doc_ref_chunks']}")
        ana = resolve_anaphora_nodes(graph, ref.get("alias_maps", {}))
        print(f"  照応解決={ana['resolved']} 除外フラグ={ana['flagged']}")
    except Exception as e:
        print(f"  ⚠️ 参照グラフ err: {e}")

    print("🔧 enrichment...")
    try:
        from graphrag_core.graph.enrichment import enrich_post_build
        est = enrich_post_build(graph)
        print(f"  mention_count={est['mention_count']} pagerank={est['pagerank']} search_keys={est['search_keys']}")
    except Exception as e:
        print(f"  ⚠️ enrich err: {e}")

    print("🏷️ provenance刻印(plant_v15)...")
    try:
        from graphrag_core.graph.provenance import stamp_graph_provenance
        ok = stamp_graph_provenance(graph, args.provenance, doc_count=len(docs))
        print(f"  stamped={ok}")
    except Exception as e:
        print(f"  ⚠️ provenance err: {e}")

    try:
        from graphrag_core.graph.schema import stamp_schema_metadata
        stamp_schema_metadata(graph)
    except Exception:
        pass

    print("PLANT KG BUILD COMPLETE")


if __name__ == "__main__":
    main()
