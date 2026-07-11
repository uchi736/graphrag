"""スキーマ表示・EDCスキーマ同期サービス。

- schema_report: 「現グラフ構築時のスキーマ（Neo4jのSchemaMeta刻印）」と
  「次回ビルドで使われるスキーマ（SHARED_SCHEMA_PATH）」を突き合わせる。
  不一致のまま再構築すると黙ってスキーマが変わる事故をUIで警告するため。
- sync_edc_schema: 現コレクションの文書サンプルをEDCに流してスキーマを
  自動発見し、graphrag形式JSONへ書き出す。
  scripts/edc_schema_sync.py のジョブ版（サンプリング元がチャンクディレクトリ
  ではなく PGVector の登録文書）。CLI版はオフライン一括用に併存。

EDCの実行モード（EDCはgraphragの一機能。既定でサーバ不要）:
- builtin: vendor/EDC/extract_cli.py を子プロセス実行（既定）。
  依存はgraphrag venvの部分集合、接続先(VLLM_*)は自プロセスの環境を継承。
- http:    環境変数 EDC_ENDPOINT が明示されている場合のみ。外部で共用EDC
  （Dify連携等）を運用しているケース用。

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


def _vendor_edc_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "vendor" / "EDC"


def edc_mode() -> Dict[str, Optional[str]]:
    """EDCの実行モードを返す。{"mode": "http"|"builtin", "endpoint": str|None}"""
    ep = (os.getenv("EDC_ENDPOINT") or "").strip()
    if ep:
        return {"mode": "http", "endpoint": ep.rstrip("/")}
    return {"mode": "builtin", "endpoint": None}


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
    mode = edc_mode()
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
        "edc_mode": mode["mode"],
        "edc_endpoint": mode["endpoint"],
    }


def read_schema_file() -> Dict:
    """SHARED_SCHEMA_PATH のスキーマJSONを生のまま返す（編集UI用）。

    未設定/未存在なら組み込みデフォルト相当を編集開始点として返す。
    """
    from graphrag_core.graph.schema import (
        DEFAULT_NODE_TYPES, DEFAULT_RELATIONS, load_schema)
    configured = load_schema()
    src = configured.get("source")
    if src and Path(src).exists():
        data = json.loads(Path(src).read_text(encoding="utf-8"))
        return {"path": src, "exists": True, "data": data}
    return {
        "path": None,
        "exists": False,
        "data": {
            "domain": "custom",
            "version": "custom-v1",
            "node_types": list(DEFAULT_NODE_TYPES),
            "node_type_definitions": {},
            "relations": [{"name": r, "description": ""} for r in DEFAULT_RELATIONS],
        },
    }


def save_schema_file(node_types: list, relations: list, *,
                     pg_collection: str,
                     domain: Optional[str] = None,
                     path: Optional[str] = None) -> Dict:
    """人手キュレーション済みスキーマをJSONへ保存する（編集UI用）。

    - path 省略時: SHARED_SCHEMA_PATH のファイルを上書き（次回ビルドに即反映）。
      SHARED_SCHEMA_PATH 未設定なら schemas/custom_<collection>.json に新規作成。
    - 元ファイルの node_type_definitions / description は名前が残っている限り引き継ぐ。
    - 上書き前に <path>.bak へバックアップ。
    """
    current = read_schema_file()
    base = current["data"]
    old_defs = base.get("node_type_definitions") or {}
    old_rel_desc = {r.get("name"): r.get("description", "")
                    for r in (base.get("relations") or []) if isinstance(r, dict)}

    norm_relations = []
    for r in relations:
        if isinstance(r, str):
            norm_relations.append({"name": r, "description": old_rel_desc.get(r, "")})
        elif isinstance(r, dict) and r.get("name"):
            norm_relations.append({"name": r["name"],
                                   "description": r.get("description") or old_rel_desc.get(r["name"], "")})
    node_types = [t for t in node_types if t]
    if not node_types or not norm_relations:
        raise ValueError("node_types と relations は1件以上必要です")

    out = {
        "domain": domain or base.get("domain") or pg_collection,
        "version": f"curated-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "UI手動キュレーション (services/schema_sync.save_schema_file)",
        "notes": (base.get("notes") or []) + ["UIでキュレーション（削除/追加）済み"],
        "node_types": sorted(set(node_types)),
        "node_type_definitions": {k: v for k, v in old_defs.items() if k in set(node_types)},
        "relations": sorted(norm_relations, key=lambda r: r["name"]),
    }
    if path is None:
        path = current["path"]
    if path is None:
        root = Path(__file__).resolve().parents[2]
        path = str(root / "schemas" / f"custom_{pg_collection}.json")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.with_suffix(p.suffix + ".bak").write_text(
            p.read_text(encoding="utf-8"), encoding="utf-8")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    hint = None
    if _norm_path(str(p)) != _norm_path(current["path"]):
        hint = f"SHARED_SCHEMA_PATH={p} を設定すると次回ビルドで使われます"
    return {"path": str(p), "node_types": out["node_types"],
            "relations": [r["name"] for r in out["relations"]], "hint": hint}


def _extract_via_http(endpoint: str, text: str, timeout: float) -> Dict:
    """外部EDC API（EDC_ENDPOINT明示時のみ）で /extract を呼ぶ。"""
    import requests
    r = requests.post(
        f"{endpoint}/extract",
        json={"text": text, "doctype": "auto", "chunk_method": "recursive",
              "enrich_schema": True},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def _extract_via_builtin(text: str, timeout: float) -> Dict:
    """同梱EDC（vendor/EDC）を子プロセスでワンショット実行する（サーバ不要）。

    - 依存は graphrag venv の部分集合（requirements-api.txt で実測）
    - 接続先(VLLM_*)は自プロセスの環境変数を継承（graphragの.env由来）
    - cwd=vendor/EDC で起動し、EDC側のリソース解決と環境を隔離する
    """
    import subprocess
    import sys
    import tempfile

    edc_dir = _vendor_edc_dir()
    cli = edc_dir / "extract_cli.py"
    if not cli.exists():
        raise RuntimeError(
            f"同梱EDCが見つかりません ({cli})。EDC_ENDPOINT で外部EDCを指定するか、"
            "vendor/EDC を復元してください")
    env = dict(os.environ)
    env.setdefault("VLLM_API_KEY", "dummy")
    env.setdefault("VLLM_MAX_RETRIES", "5")
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    req = {"text": text, "doctype": "auto", "chunk_method": "recursive",
           "enrich_schema": True}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "result.json"
        p = subprocess.run(
            [sys.executable, str(cli), "--out", str(out)],
            input=json.dumps(req, ensure_ascii=False).encode("utf-8"),
            cwd=str(edc_dir), env=env, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if p.returncode != 0 or not out.exists():
            tail = p.stderr.decode("utf-8", "replace")[-800:]
            raise RuntimeError(f"EDC抽出サブプロセスが失敗 (exit {p.returncode}): {tail}")
        return json.loads(out.read_text(encoding="utf-8"))


def sync_edc_schema(pg_conn: str, pg_collection: str, *,
                    out_path: Optional[str] = None,
                    endpoint: Optional[str] = None,
                    n_docs: int = 4, chunks_per_doc: int = 6,
                    timeout: float = 1800.0,
                    progress: Optional[ProgressFn] = None,
                    should_cancel: Optional[Callable[[], bool]] = None) -> Dict:
    """現コレクションの文書サンプルからEDCでスキーマを発見しJSONへ書き出す。

    実行モード: endpoint 引数 or EDC_ENDPOINT があればHTTP、なければ同梱EDCを
    子プロセス実行（既定・サーバ不要）。
    出力先既定: <graphragルート>/schemas/edc_<collection>.json
    書き出すだけで SHARED_SCHEMA_PATH は変更しない（ビルド時に明示指定する）。
    """
    from graphrag_core.services.documents import (
        list_document_chunks, list_registered_documents)

    def _p(**kw):
        if progress:
            progress(ProgressEvent(**kw))

    endpoint = (endpoint or "").strip() or edc_mode()["endpoint"]
    if endpoint:
        endpoint = endpoint.rstrip("/")
        mode_label = f"HTTP ({endpoint})"
        import requests
        _p(stage="health", message=f"EDC API 疎通確認 ({endpoint})...")
        try:
            requests.get(f"{endpoint}/health", timeout=5).raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"EDC API に到達できません ({endpoint}): {e}。"
                "EDC_ENDPOINT を外すと同梱EDC（サーバ不要）で実行します") from e
        _extract = lambda text: _extract_via_http(endpoint, text, timeout)  # noqa: E731
    else:
        mode_label = "内蔵 (vendor/EDC 子プロセス)"
        _extract = lambda text: _extract_via_builtin(text, timeout)  # noqa: E731

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
           message=f"{src}（{len(text)}字）をEDC抽出中 [{mode_label}]...")
        data = _extract(text)
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
        "generated_by": f"services/schema_sync.sync_edc_schema ({mode_label}, doctype=auto, enrich_schema=True)",
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
