#!/usr/bin/env python
"""edc_schema_sync.py - EDCフレームワークからスキーマを取得し graphrag 形式へ変換
=================================================================================
EDC API (/extract, doctype自動分類 + enrich_schema) にコーパスのサンプルを流し、
発見・正規化されたスキーマ(関係)と型を、graphrag の SHARED_SCHEMA_PATH が読める
JSON `{domain, version, generated_at, node_types, relations[{name,...}]}` に変換する。

これにより新コーパスのスキーマ手キュレーション（fujitsu_kg_schema_v2.json 相当の作業）を
EDC のスキーマ発見で自動化する。

使用例:
    # 事前チャンク済みコーパスから文書サンプルを抽出してスキーマ生成
    python scripts/edc_schema_sync.py \
        --chunks-dir C:/work/makedataset/data/chunks_synth \
        --domain synth_v1 --out _bench/edc_schema_synth.json \
        --docs 4 --chunks-per-doc 6

    # 生成したスキーマでKGビルド:
    #   SHARED_SCHEMA_PATH=_bench/edc_schema_synth.json python _bench/plant/build_kg_plant.py ...
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import requests


def sample_docs(chunks_dir: str, n_docs: int, chunks_per_doc: int) -> dict:
    """チャンクJSONLディレクトリから文書ごとのサンプルテキストを作る。

    文書は多様性優先で「サイズ降順に種類が異なるものから」ではなく単純に
    先頭 n_docs 件（ファイル名順）を取る。必要なら --docs を増やす。
    """
    samples = {}
    files = sorted(glob.glob(str(Path(chunks_dir) / "*.jsonl")))[:n_docs]
    for fp in files:
        doc_id = Path(fp).stem
        texts = []
        for line in open(fp, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            t = (r.get("text") or "").strip()
            if t:
                texts.append(t)
            if len(texts) >= chunks_per_doc:
                break
        if texts:
            samples[doc_id] = "\n".join(texts)
    return samples


def call_edc(endpoint: str, text: str, timeout: float) -> dict:
    r = requests.post(
        f"{endpoint.rstrip('/')}/extract",
        json={"text": text, "doctype": "auto", "chunk_method": "recursive",
              "enrich_schema": True},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser(description="EDCスキーマ→graphrag形式 変換")
    ap.add_argument("--chunks-dir", required=True, help="チャンクJSONLディレクトリ")
    ap.add_argument("--out", required=True, help="出力スキーマJSONパス（SHARED_SCHEMA_PATHに指定する）")
    ap.add_argument("--domain", default="edc-discovered", help="スキーマのdomain名")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080", help="EDC APIエンドポイント")
    ap.add_argument("--docs", type=int, default=4, help="サンプリングする文書数")
    ap.add_argument("--chunks-per-doc", type=int, default=6, help="文書ごとのサンプルチャンク数")
    ap.add_argument("--timeout", type=float, default=1800, help="EDC呼び出しタイムアウト秒")
    args = ap.parse_args()

    # 疎通
    try:
        requests.get(f"{args.endpoint.rstrip('/')}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"❌ EDC API に到達できません ({args.endpoint}): {e}")
        print("   起動例: cd C:/work/RAG/EDC/edc && myenv/Scripts/python -m uvicorn api:app --port 8080")
        sys.exit(1)

    samples = sample_docs(args.chunks_dir, args.docs, args.chunks_per_doc)
    print(f"=== EDC schema sync === docs={len(samples)} endpoint={args.endpoint}")

    relations: dict[str, str] = {}   # name -> definition（先着優先）
    types: dict[str, str] = {}
    doc_notes = []
    for doc_id, text in samples.items():
        print(f"  [{doc_id}] {len(text)}字 を /extract へ...", flush=True)
        try:
            d = call_edc(args.endpoint, text, args.timeout)
        except Exception as e:
            print(f"    ⚠️ 失敗: {type(e).__name__}: {e}")
            continue
        sch = d.get("schema_") or {}
        typ = d.get("types") or {}
        for k, v in sch.items():
            relations.setdefault(k, v or "")
        for k, v in typ.items():
            types.setdefault(k, v or "")
        doc_notes.append(f"{doc_id}: doctype={d.get('doctype')} "
                         f"(+{len(sch)}関係/{len(typ)}型, triplets={d.get('n_triplets')})")
        print(f"    doctype={d.get('doctype')} 関係+{len(sch)} 型+{len(typ)}")

    if not relations:
        print("❌ スキーマを1件も取得できませんでした")
        sys.exit(1)

    out = {
        "domain": args.domain,
        "version": f"edc-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edc_schema_sync.py (EDC /extract, doctype=auto, enrich_schema=True)",
        "notes": ["EDCフレームワークによる自動スキーマ発見（Extract-Define-Canonicalize）"] + doc_notes,
        "node_types": sorted(types.keys()),
        "node_type_definitions": types,
        "relations": [{"name": k, "description": v} for k, v in sorted(relations.items())],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 書き出し: {args.out}")
    print(f"   node_types={len(out['node_types'])}: {out['node_types']}")
    print(f"   relations={len(out['relations'])}: {[r['name'] for r in out['relations']]}")
    print(f"\n使い方: SHARED_SCHEMA_PATH={args.out} を .env か環境変数で指定して build_kg を実行")


if __name__ == "__main__":
    main()
