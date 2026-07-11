"""専門用語辞書サービス（名寄せ用途）。

辞書 = {canonical: 正式名, aliases: [別名...], category, definition} のリスト。
用途は主に **名寄せ**: 同一対象が「労基法」「労働基準法」のように別ノードに
分裂しているとき、辞書を根拠に1ノードへ統合する（エッジ付け替え・破壊的）。
統合後は aliases が search_keys に反映され、質問の別名表記でもヒットする。

- read/save: TERM_DICTIONARY_PATH（未設定なら schemas/term_dictionary_<collection>.json）
- report: 各エントリのグラフ内マッチ状況（統合候補/一致/未マッチ）
- apply: 名寄せ統合 → プロパティ付与 → search_keys 再計算（ジョブ実行想定）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from graphrag_core.services.progress import JobCancelled, ProgressEvent, ProgressFn


def dictionary_path(pg_collection: str) -> Path:
    from graphrag_core.config import get_settings
    raw = (get_settings().kg_dictionary_path or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[2] / raw
        return p
    return Path(__file__).resolve().parents[2] / "schemas" / f"term_dictionary_{pg_collection}.json"


def read_dictionary_file(pg_collection: str) -> Dict:
    """辞書ファイルを読む。無ければ空エントリで返す（新規作成の開始点）。"""
    from graphrag_core.graph.dictionary import load_dictionary
    p = dictionary_path(pg_collection)
    if p.exists():
        entries = load_dictionary(p)
    else:
        entries = []
    return {"path": str(p), "exists": p.exists(), "entries": entries}


def save_dictionary_file(entries: List[Dict], pg_collection: str) -> Dict:
    """辞書エントリをJSON保存（.bakバックアップ付き）。canonical必須・重複除去。"""
    seen = set()
    norm: List[Dict] = []
    for e in entries:
        canonical = (e.get("canonical") or "").strip()
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        aliases = [a.strip() for a in (e.get("aliases") or [])
                   if a.strip() and a.strip() != canonical]
        norm.append({
            "canonical": canonical,
            "aliases": aliases,
            "category": (e.get("category") or "").strip(),
            "definition": (e.get("definition") or "").strip(),
        })
    if not norm:
        raise ValueError("有効なエントリがありません（canonical は必須です）")
    p = dictionary_path(pg_collection)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.with_suffix(p.suffix + ".bak").write_text(
            p.read_text(encoding="utf-8"), encoding="utf-8")
    p.write_text(json.dumps({"entries": norm}, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return {"path": str(p), "n_entries": len(norm)}


def dictionary_report(graph, pg_collection: str) -> Dict:
    """各エントリのグラフ内マッチ状況を返す（適用前のプレビュー）。

    status: merge_candidate（2ノード以上→統合される）/ matched（1ノード）/
            unmatched（0ノード）
    """
    from graphrag_core.graph.schema import entity_node_predicate
    d = read_dictionary_file(pg_collection)
    _pred = entity_node_predicate("n")
    detail = []
    counts = {"merge_candidate": 0, "matched": 0, "unmatched": 0}
    for e in d["entries"]:
        keys = [e["canonical"]] + e["aliases"]
        try:
            rows = graph.query(
                f"MATCH (n) WHERE n.id IN $keys AND {_pred} "
                "RETURN n.id AS id ORDER BY n.id",
                {"keys": keys}) or []
        except Exception:
            rows = []
        ids = [r["id"] for r in rows]
        status = ("merge_candidate" if len(ids) >= 2
                  else "matched" if len(ids) == 1 else "unmatched")
        counts[status] += 1
        detail.append({**e, "matched_ids": ids, "status": status})
    return {"path": d["path"], "exists": d["exists"], "entries": detail,
            "counts": counts}


def apply_dictionary_full(graph, pg_collection: str, *,
                          merge: bool = True,
                          progress: Optional[ProgressFn] = None,
                          should_cancel: Optional[Callable[[], bool]] = None) -> Dict:
    """辞書を既存グラフへ適用する（名寄せ→プロパティ付与→search_keys再計算）。

    LLM不要（Cypherのみ）。ジョブとして実行する想定。
    """
    from graphrag_core.graph.dictionary import (
        apply_dictionary, merge_dictionary_aliases)
    from graphrag_core.graph.enrichment import enrich_post_update

    def _p(**kw):
        if progress:
            progress(ProgressEvent(**kw))

    d = read_dictionary_file(pg_collection)
    entries = d["entries"]
    if not entries:
        raise RuntimeError(f"辞書が空です: {d['path']}")

    result: Dict = {"path": d["path"], "n_entries": len(entries)}
    if merge:
        _p(stage="merge", message=f"名寄せ統合中（{len(entries)}エントリ）...")
        if should_cancel and should_cancel():
            raise JobCancelled()
        result["merge"] = merge_dictionary_aliases(graph, entries)
    _p(stage="apply", message="プロパティ付与（canonical_form/aliases/definition）...")
    result["apply"] = apply_dictionary(graph, entries)
    _p(stage="post", message="search_keys / mention_count 再計算...")
    enrich_post_update(graph)
    return result
