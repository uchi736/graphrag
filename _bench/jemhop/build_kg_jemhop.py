#!/usr/bin/env python
"""JEMHopQA dev-gold 記事に対する KG を vLLM(gemma) で構築（Fujitsuグラフは退避）。

Neo4j は community(単一DB)で別DBを作れないため:
  1. 現行グラフ(Fujitsu)を gzip JSONL にフルバックアップ
  2. Neo4j を全削除（バッチ）
  3. dev-gold 219記事のチャンクから LLMGraphTransformer(vLLM, オープンドメイン抽出)で
     KG を構築（include_source=True で Document+MENTIONS 付与）
  4. enrich_post_build（jemhop専用、search_keys/pagerank）

復元: _bench/jemhop/restore_neo4j.py（別途）でバックアップから戻す。
コスト: 生成は vLLM gemma-4-26B(@8000) を使用（Azure課金なし）。

Usage:
    python _bench/jemhop/build_kg_jemhop.py
"""
from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_proj = Path(__file__).resolve().parents[2]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

# 生成は vLLM を使う（Azure課金回避）。create_chat_llm より前に設定
os.environ["LLM_PROVIDER"] = "vllm"

from dotenv import load_dotenv
load_dotenv()
os.environ["LLM_PROVIDER"] = "vllm"  # load_dotenv の上書きを再度上書き

HERE = Path(__file__).resolve().parent
BACKUP = HERE / "neo4j_backup_pre_jemhop.jsonl.gz"


def backup_graph(graph):
    if BACKUP.exists():
        print(f"  backup already exists: {BACKUP}")
        return
    n_nodes = n_rels = 0
    with gzip.open(BACKUP, "wt", encoding="utf-8") as f:
        skip = 0
        while True:
            rows = graph.query(
                "MATCH (n) WITH n ORDER BY elementId(n) SKIP $skip LIMIT 5000 "
                "RETURN labels(n) AS labels, properties(n) AS props",
                {"skip": skip})
            if not rows:
                break
            for r in rows:
                f.write(json.dumps({"kind": "node", "labels": r["labels"], "props": r["props"]},
                                   ensure_ascii=False, default=str) + "\n")
            n_nodes += len(rows)
            if len(rows) < 5000:
                break
            skip += 5000
        skip = 0
        while True:
            rows = graph.query(
                "MATCH (a)-[r]->(b) WITH a,r,b ORDER BY elementId(r) SKIP $skip LIMIT 5000 "
                "RETURN type(r) AS t, properties(r) AS props, "
                "COALESCE(a.id,a.hash) AS sid, labels(a) AS sl, "
                "COALESCE(b.id,b.hash) AS eid, labels(b) AS el",
                {"skip": skip})
            if not rows:
                break
            for r in rows:
                f.write(json.dumps({"kind": "rel", "type": r["t"], "props": r["props"],
                                    "start_id": r["sid"], "start_labels": r["sl"],
                                    "end_id": r["eid"], "end_labels": r["el"]},
                                   ensure_ascii=False, default=str) + "\n")
            n_rels += len(rows)
            if len(rows) < 5000:
                break
            skip += 5000
    print(f"  backup: nodes={n_nodes} rels={n_rels} -> {BACKUP} ({BACKUP.stat().st_size/1e6:.1f}MB)")


def wipe(graph):
    while True:
        r = graph.query("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS c")
        if not r or r[0]["c"] == 0:
            break
    print("  wiped Neo4j")


def load_dev_gold_docs():
    from langchain_core.documents import Document
    dev = json.loads((HERE / "dev.json").read_text(encoding="utf-8"))
    # JEMHOP_KG_SCOPE: compositional(既定, KGが効く型のみ・高速) / all(全dev-gold)
    scope = os.environ.get("JEMHOP_KG_SCOPE", "compositional")
    rows = [r for r in dev if (scope == "all" or r.get("type") == "compositional")]
    gold = sorted({str(p) for r in rows for p in (r.get("page_ids") or [])})
    # bridge関係(父/出身地/所属等)は記事冒頭に集中するため、記事あたり先頭Nチャンクに絞り高速化
    per_art = int(os.environ.get("JEMHOP_CHUNKS_PER_ART", "3"))
    print(f"  KG scope={scope} (questions={len(rows)}) chunks/article<={per_art}")
    docs = []
    for pid in gold:
        fp = HERE / "chunks_jemhop" / f"{pid}.jsonl"
        if not fp.exists():
            continue
        taken = 0
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line or taken >= per_art:
                continue
            r = json.loads(line)
            taken += 1
            text = (r.get("text") or "").strip()
            if not text:
                continue
            title = r.get("title") or ""
            content = f"【{title}】\n{text}" if title else text
            docs.append(Document(
                page_content=content,
                metadata={"id": hashlib.sha256(content.encode()).hexdigest(),
                          "source": pid, "title": title},
            ))
    print(f"  dev-gold articles={len(gold)} chunks={len(docs)}")
    return docs


def main():
    from langchain_neo4j import Neo4jGraph
    from graphrag_core.config import get_settings
    s = get_settings()
    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw,
                       enhanced_schema=False)

    print("=== 1. backup current graph ===")
    backup_graph(graph)
    print("=== 2. wipe ===")
    wipe(graph)

    print("=== 3. build KG (vLLM gemma, open-domain) ===")
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from graphrag_core.llm.factory import create_chat_llm
    from graphrag_core.graph.enrichment import attach_source_chunks

    llm = create_chat_llm(temperature=0)
    print(f"  LLM provider={s.llm_provider} model={getattr(s,'vllm_model',None)}")
    # オープンドメイン抽出（plantスキーマは使わない）。日本語Wikipediaの人物/作品/地名と
    # 父/母/出身地/所属/監督/発売日 等の関係を自由抽出。
    transformer = LLMGraphTransformer(
        llm=llm, ignore_tool_usage=True,
        additional_instructions=(
            "日本語Wikipediaのテキストから、人物・作品・組織・地名・製品などのエンティティと、"
            "それらの間の関係（例: 父, 母, 配偶者, 出身地, 所属, 監督, 著者, 発売日, 所在地, "
            "前身, 後継, メンバー）を抽出する。固有名詞をノードとし、関係名は日本語/英語どちらでもよい。"),
    )

    docs = load_dev_gold_docs()

    def _proc(chunk):
        gdocs = transformer.convert_to_graph_documents([chunk])
        graph.add_graph_documents(gdocs, include_source=True)
        h = chunk.metadata.get("id")
        if h:
            attach_source_chunks(graph, gdocs, h)
        return h

    workers = int(os.environ.get("KG_BUILD_WORKERS", "8"))
    ok = err = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_proc, d) for d in docs]
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            try:
                fut.result(); ok += 1
            except Exception as e:
                err += 1
                if err <= 5:
                    print(f"  err: {str(e)[:100]}")
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(docs)} ({(time.time()-t0)/(i+1):.2f}s/chunk, ok={ok} err={err})", flush=True)

    nodes = graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
    edges = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    print(f"  built: nodes={nodes} edges={edges} ok={ok} err={err} ({time.time()-t0:.0f}s)")

    print("=== 4. enrich (jemhop-only) ===")
    from graphrag_core.graph.enrichment import enrich_post_build
    est = enrich_post_build(graph)
    print(f"  enrich: {est}")
    print("BUILD COMPLETE")


if __name__ == "__main__":
    main()
