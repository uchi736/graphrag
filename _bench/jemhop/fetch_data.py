#!/usr/bin/env python
"""JEMHopQA (aiishii/JEMHopQA, CC BY-SA 4.0) のデータ取得 + 検索コーパス構築。

JEMHopQA は QA + derivation triples + page_ids(日本語Wikipedia数値ID) のみで
本文(検索コーパス)を同梱しない。本スクリプトは:
  1. dev.json / train.json を GitHub raw から取得
  2. dev(gold) + train(distractor) の page_id を集約
  3. 各 page_id の Wikipedia 本文を ja.wikipedia API (prop=extracts) で取得
  4. 1記事=1ファイルでチャンク化し _bench/jemhop/chunks_jemhop/<page_id>.jsonl に保存
     （plant の chunks 形式に合わせ ingest_plant 系で投入可能に）

Wikipediaは現行版を取得（JEMHopQAの2021版とは軽微なドリフトあり。
time_dependent=false の比較事実は概ね安定）。

Usage:
    python _bench/jemhop/fetch_data.py            # dev gold + train distractor 全取得
    python _bench/jemhop/fetch_data.py --dev-only # devのgold記事のみ
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
RAW = "https://raw.githubusercontent.com/aiishii/JEMHopQA/main/corpus"
CHUNKS_DIR = HERE / "chunks_jemhop"
WIKI_API = "https://ja.wikipedia.org/w/api.php"
UA = "graphrag-research-bench/0.1 (https://example.org; contact local)"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def download_qa():
    for name in ("dev.json", "train.json"):
        out = HERE / name
        if out.exists():
            print(f"  {name} already present")
            continue
        r = requests.get(f"{RAW}/{name}", timeout=30, headers={"User-Agent": UA})
        r.raise_for_status()
        out.write_text(r.text, encoding="utf-8")
        print(f"  downloaded {name} ({len(r.text)} bytes)")


def load_qa(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def collect_page_ids(dev_only: bool):
    dev = load_qa("dev.json")
    ids = set()
    for r in dev:
        ids.update(str(p) for p in (r.get("page_ids") or []))
    n_gold = len(ids)
    if not dev_only:
        train = load_qa("train.json")
        for r in train:
            ids.update(str(p) for p in (r.get("page_ids") or []))
    print(f"page_ids: dev_gold={n_gold} total={len(ids)} (distractor={len(ids)-n_gold})")
    return sorted(ids)


def _fetch_one(sess, pid):
    """1 pageid の full extract を取得。redirectは解決し、要求pidに紐付ける。
    （prop=extracts の full extract はバッチだと1件しか返らない制約があるため単発）"""
    params = {
        "action": "query", "format": "json", "pageids": pid,
        "prop": "extracts", "explaintext": 1, "exlimit": 1, "redirects": 1,
    }
    for attempt in range(4):
        try:
            r = sess.get(WIKI_API, params=params, timeout=15)
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", {})
            for _, page in pages.items():
                if "missing" in page:
                    continue
                txt = (page.get("extract") or "").strip()
                if txt:
                    return pid, {"title": page.get("title", ""), "text": txt}
            return pid, None
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return pid, None


def fetch_extracts(page_ids):
    """pageid -> {'title','text'} を単発リクエスト×スレッドプールで取得"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out = {}
    sess = requests.Session()
    sess.headers["User-Agent"] = UA
    done = 0
    # 2ワーカー（4並列はWikipediaにスロットルされ成功率30%に低下した実測あり）
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_fetch_one, sess, pid): pid for pid in page_ids}
        for fut in as_completed(futs):
            pid, res = fut.result()
            if res:
                out[pid] = res
            done += 1
            if done % 100 == 0 or done == len(page_ids):
                print(f"  fetched {done}/{len(page_ids)} (ok={len(out)})", flush=True)
    return out


def chunk_text(text):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    sp = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "、", " ", ""], length_function=len,
    )
    return sp.split_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-only", action="store_true")
    args = ap.parse_args()

    print("=== download QA ===")
    download_qa()

    print("=== collect page_ids ===")
    page_ids = collect_page_ids(args.dev_only)

    print("=== fetch Wikipedia extracts ===")
    arts = fetch_extracts(page_ids)
    print(f"fetched {len(arts)}/{len(page_ids)} articles (missing/empty {len(page_ids)-len(arts)})")

    print("=== chunk + write ===")
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    for pid, art in arts.items():
        chunks = chunk_text(art["text"])
        recs = []
        for j, c in enumerate(chunks):
            recs.append({
                "chunk_id": f"{pid}__c{j:04d}",
                "doc_id": pid,
                "title": art["title"],
                "text": c,
            })
        (CHUNKS_DIR / f"{pid}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
        n_chunks += len(recs)
    # タイトル索引（採点・可読性用）
    (HERE / "page_titles.json").write_text(
        json.dumps({pid: a["title"] for pid, a in arts.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"wrote {n_chunks} chunks across {len(arts)} articles -> {CHUNKS_DIR}")
    print("FETCH COMPLETE")


if __name__ == "__main__":
    main()
