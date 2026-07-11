"""スキーマ表示・EDCスキーマ同期サービス。

- schema_report: 「現グラフ構築時のスキーマ（Neo4jのSchemaMeta刻印）」と
  「次回ビルドで使われるスキーマ（SHARED_SCHEMA_PATH）」を突き合わせる。
  不一致のまま再構築すると黙ってスキーマが変わる事故をUIで警告するため。
- sync_edc_schema: 現コレクションの文書サンプルを EDC API (/extract,
  enrich_schema=True) に流してスキーマを自動発見し、graphrag形式JSONへ書き出す。
  scripts/edc_schema_sync.py のジョブ版（サンプリング元がチャンクディレクトリ
  ではなく PGVector の登録文書）。CLI版はオフライン一括用に併存。

EDC API 仕様の罠: レスポンスの関係辞書キーは `schema_`（pydantic の
BaseModel.schema シャドウ回避のため末尾アンダースコア）。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

from graphrag_core.services.progress import JobCancelled, ProgressEvent, ProgressFn


def default_edc_endpoint() -> str:
    return os.getenv("EDC_ENDPOINT", "http://127.0.0.1:8080")


def _norm_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    try:
        return str(Path(p).resolve()).lower()
    except Exception:
        return str(p).lower()


def schema_report(graph) -> Dict:
    """アクティブ（グラフ刻印）と設定中（SHARED_SCHEMA_PATH）のスキーマを返す。

    Returns:
        {"active": {...}|None, "configured": {...}, "match": bool|None,
         "edc_endpoint": str}
        match は active が無い場合 None（判定不能）。
    """
    from graphrag_core.graph.schema import load_schema

    active = None
    try:
        rows = graph.query(
            "MATCH (m:SchemaMeta {kind:'active'}) RETURN properties(m) AS p")
        if rows:
            p = dict(rows[0]["p"])
            p["stamped_at"] = str(p.get("stamped_at") or "")
            p["source"] = p.get("source") or None
            active = p
    except Exception:
        pass

    configured = load_schema()
    match = None
    if active is not None:
        match = _norm_path(active.get("source")) == _norm_path(configured.get("source"))
    return {
        "active": active,
        "configured": {
            "domain": configured["domain"],
            "version": configured["version"],
            "source": configured.get("source"),
            "node_types": configured["node_types"],
            "relations": configured["relations"],
        },
        "match": match,
        "edc_endpoint": default_edc_endpoint(),
    }


def sync_edc_schema(pg_conn: str, pg_collection: str, *,
                    out_path: Optional[str] = None,
                    endpoint: Optional[str] = None,
                    n_docs: int = 4, chunks_per_doc: int = 6,
                    timeout: float = 1800.0,
                    progress: Optional[ProgressFn] = None,
                    should_cancel: Optional[Callable[[], bool]] = None) -> Dict:
    """現コレクションの文書サンプルからEDCでスキーマを発見しJSONへ書き出す。

    出力先既定: <graphragルート>/schemas/edc_<collection>.json
    書き出すだけで SHARED_SCHEMA_PATH は変更しない（ビルド時に明示指定する）。
    """
    import requests
    from graphrag_core.services.documents import (
        list_document_chunks, list_registered_documents)

    def _p(**kw):
        if progress:
            progress(ProgressEvent(**kw))

    endpoint = (endpoint or default_edc_endpoint()).rstrip("/")
    _p(stage="health", message=f"EDC API 疎通確認 ({endpoint})...")
    try:
        requests.get(f"{endpoint}/health", timeout=5).raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"EDC API に到達できません ({endpoint}): {e}。"
            "起動例: cd <EDCリポジトリ> && python -m uvicorn api:app --port 8080") from e

    docs = list_registered_documents(pg_conn, pg_collection)["documents"][:max(1, n_docs)]
    if not docs:
        raise RuntimeError(f"コレクション {pg_collection} に文書がありません")

    relations: Dict[str, str] = {}   # name -> definition（先着優先）
    types: Dict[str, str] = {}
    doc_notes = []
    for i, d in enumerate(docs, 1):
        if should_cancel and should_cancel():
            raise JobCancelled()
        src = d["source"]
        chunks = list_document_chunks(
            pg_conn, pg_collection, src, limit=max(1, chunks_per_doc))["chunks"]
        text = "\n".join(c["text"] for c in chunks if c.get("text"))
        _p(stage="extract", current=i, total=len(docs),
           message=f"{src}（{len(text)}字）を EDC /extract へ...")
        r = requests.post(
            f"{endpoint}/extract",
            json={"text": text, "doctype": "auto", "chunk_method": "recursive",
                  "enrich_schema": True},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        sch = data.get("schema_") or {}
        typ = data.get("types") or {}
        for k, v in sch.items():
            relations.setdefault(k, v or "")
        for k, v in typ.items():
            types.setdefault(k, v or "")
        doc_notes.append(f"{src}: doctype={data.get('doctype')} "
                         f"(+{len(sch)}関係/{len(typ)}型)")

    if not relations:
        raise RuntimeError("EDCからスキーマを1件も取得できませんでした")

    out = {
        "domain": pg_collection,
        "version": f"edc-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "services/schema_sync.sync_edc_schema (EDC /extract, doctype=auto, enrich_schema=True)",
        "notes": ["EDCフレームワークによる自動スキーマ発見（Extract-Define-Canonicalize）"] + doc_notes,
        "node_types": sorted(types.keys()),
        "node_type_definitions": types,
        "relations": [{"name": k, "description": v} for k, v in sorted(relations.items())],
    }
    if out_path is None:
        root = Path(__file__).resolve().parents[2]
        out_path = str(root / "schemas" / f"edc_{pg_collection}.json")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    _p(stage="done", message=f"書き出し: {p}")
    return {
        "out_path": str(p),
        "node_types": out["node_types"],
        "relations": [r["name"] for r in out["relations"]],
        "sampled_docs": doc_notes,
        "hint": f"SHARED_SCHEMA_PATH={p} を設定して再構築するとこのスキーマが使われます",
    }
