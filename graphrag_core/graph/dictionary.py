"""専門用語辞書をTermノードに適用する

外部JSON/CSV辞書を読み、既存Termノードに canonical_form / aliases /
category / definition / notes を後付けする。

辞書フォーマット (JSON):
{
  "entries": [
    {
      "canonical": "労働基準法",
      "aliases": ["労基法"],
      "category": "法令",
      "definition": "昭和22年法律第49号"
    },
    ...
  ]
}

マッチ規則:
- Term.id が canonical または aliases のいずれかと完全一致するTermをマッチ対象とする
- マッチしたTermに canonical_form, category, definition, aliases (リスト) を SET
- 既存値があれば上書き
"""

from __future__ import annotations

import csv
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _normalize(s: Any) -> str:
    """NFKC正規化 + 内部空白潰し"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip()
    return " ".join(s.split())


def load_dictionary(path: str | Path) -> list[dict]:
    """JSON or CSV から entries を読み込み、正規化済みリストを返す。

    CSV列: canonical, aliases (|区切り), category, definition, notes (任意)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"用語辞書が見つかりません: {p}")

    entries: list[dict] = []
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        raw_entries = data.get("entries", [])
    elif p.suffix.lower() == ".csv":
        with open(p, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            raw_entries = list(reader)
            # CSVの aliases は | 区切りで分割
            for e in raw_entries:
                if isinstance(e.get("aliases"), str):
                    e["aliases"] = [a.strip() for a in e["aliases"].split("|") if a.strip()]
    else:
        raise ValueError(f"未対応の辞書フォーマット: {p.suffix}")

    for e in raw_entries:
        canonical = _normalize(e.get("canonical", ""))
        if not canonical:
            continue
        aliases = [_normalize(a) for a in (e.get("aliases") or []) if _normalize(a)]
        entries.append({
            "canonical": canonical,
            "aliases": aliases,
            "category": _normalize(e.get("category", "")),
            "definition": (e.get("definition") or "").strip(),
            "notes": (e.get("notes") or "").strip(),
        })

    logger.info("load_dictionary: %s → %d entries", p, len(entries))
    return entries


def merge_dictionary_aliases(graph, entries: list[dict]) -> dict:
    """辞書の canonical/aliases に基づき別名ノードを正規形ノードへ統合する（名寄せ）。

    エントリごとに id が canonical/aliases に一致するエンティティノードを収集し:
    - 2ノード以上 → keeper（canonical と同名のノード。無ければ次数最大を canonical へ改名）
      に他ノードのエッジを付け替えて統合（source_chunks 合算・ラベル引き継ぎ、
      consolidate._move_rels を再利用）。吸収した旧IDは keeper.aliases に積む
    - 1ノードで id≠canonical → canonical へ改名（旧IDは aliases へ）
    - 0ノード → 何もしない（unmatched として報告）

    破壊的操作。実行後は apply_dictionary（プロパティ付与）→ enrich_post_update
    （search_keys 再計算）を続けて呼ぶこと。

    Returns:
        {"merged_groups": 統合したエントリ数, "removed_nodes": 吸収削除ノード数,
         "renamed": 改名ノード数, "unmatched": 一致なしエントリ数}
    """
    from graphrag_core.graph.consolidate import _move_rels
    from graphrag_core.graph.schema import entity_node_predicate

    merged_groups = removed = renamed = unmatched = 0
    _pred = entity_node_predicate("n")

    for entry in entries:
        canonical = entry["canonical"]
        keys = [canonical] + entry["aliases"]
        rows = graph.query(
            f"""
            MATCH (n) WHERE n.id IN $keys AND {_pred}
            RETURN elementId(n) AS eid, n.id AS id, labels(n) AS labels,
                   COUNT {{ (n)--() }} AS deg
            """,
            {"keys": keys},
        ) or []
        if not rows:
            unmatched += 1
            continue

        # keeper 選定: canonical と同名 > 次数最大
        keeper = next((r for r in rows if r["id"] == canonical), None)
        if keeper is None:
            keeper = max(rows, key=lambda r: r["deg"])
        dups = [r for r in rows if r["eid"] != keeper["eid"]]

        # keeper の id を canonical へ（旧IDを aliases に退避）
        if keeper["id"] != canonical:
            graph.query(
                """
                MATCH (k) WHERE elementId(k) = $eid
                SET k.aliases = COALESCE(k.aliases, []) +
                    (CASE WHEN k.id IN COALESCE(k.aliases, []) THEN [] ELSE [k.id] END),
                    k.id = $canonical
                """,
                {"eid": keeper["eid"], "canonical": canonical},
            )
            renamed += 1

        keep_labels = set(keeper["labels"])
        for dup in dups:
            try:
                for direction in (True, False):
                    arrow = "-[r]->" if direction else "<-[r]-"
                    types = graph.query(
                        f"MATCH (d){arrow}() WHERE elementId(d) = $dup "
                        "RETURN DISTINCT type(r) AS t",
                        {"dup": dup["eid"]},
                    ) or []
                    for row in types:
                        _move_rels(graph, dup["eid"], keeper["eid"], row["t"],
                                   outgoing=direction)
                new_labels = [l for l in dup["labels"] if l not in keep_labels]
                if new_labels:
                    label_expr = "".join(f":`{l}`" for l in new_labels)
                    graph.query(
                        f"MATCH (k) WHERE elementId(k) = $keep SET k{label_expr}",
                        {"keep": keeper["eid"]},
                    )
                    keep_labels.update(new_labels)
                # 吸収した旧IDを aliases へ退避してから削除
                graph.query(
                    """
                    MATCH (k) WHERE elementId(k) = $keep
                    SET k.aliases = COALESCE(k.aliases, []) +
                        (CASE WHEN $old IN COALESCE(k.aliases, []) THEN [] ELSE [$old] END)
                    """,
                    {"keep": keeper["eid"], "old": dup["id"]},
                )
                graph.query("MATCH (d) WHERE elementId(d) = $dup DETACH DELETE d",
                            {"dup": dup["eid"]})
                removed += 1
            except Exception as e:
                logger.warning("dictionary merge failed for %s ← %s: %s",
                               canonical, dup["id"], e)
        if dups:
            merged_groups += 1

    logger.info("merge_dictionary_aliases: %d groups merged, %d nodes removed, "
                "%d renamed, %d unmatched", merged_groups, removed, renamed, unmatched)
    return {"merged_groups": merged_groups, "removed_nodes": removed,
            "renamed": renamed, "unmatched": unmatched}


def apply_dictionary(graph, entries: list[dict]) -> dict:
    """辞書entriesを既存Termノードに適用する。

    Args:
        graph: langchain Neo4jGraph
        entries: load_dictionary() の戻り値

    Returns:
        {"matched": int, "untouched_entries": int, "term_updates": int}
    """
    if not entries:
        return {"matched": 0, "untouched_entries": 0, "term_updates": 0}

    matched_terms = 0
    term_updates = 0
    untouched = 0

    for entry in entries:
        # canonical または aliases のいずれかに一致する Term を探す
        keys = [entry["canonical"]] + entry["aliases"]
        try:
            from graphrag_core.graph.schema import entity_node_predicate
            rows = graph.query(
                f"""
                MATCH (t)
                WHERE t.id IN $keys AND {entity_node_predicate("t")}
                RETURN COLLECT(t.id) AS ids
                """,
                {"keys": keys},
            )
            ids = (rows[0]["ids"] if rows else []) or []
        except Exception as e:
            logger.warning("dictionary match query failed for %s: %s", entry["canonical"], e)
            continue

        if not ids:
            untouched += 1
            continue

        matched_terms += len(ids)

        # 該当Termに property をセット (canonical/aliases/category/definition/notes)
        params = {
            "ids": ids,
            "canonical": entry["canonical"],
            "aliases": entry["aliases"],
            "category": entry["category"] or None,
            "definition": entry["definition"] or None,
            "notes": entry["notes"] or None,
        }
        try:
            graph.query(
                """
                UNWIND $ids AS tid
                MATCH (t {id: tid})
                SET t.canonical_form = $canonical,
                    t.aliases = $aliases,
                    t.category = $category,
                    t.definition = $definition,
                    t.notes = $notes
                """,
                params,
            )
            term_updates += len(ids)
        except Exception as e:
            logger.warning("dictionary update failed for %s: %s", entry["canonical"], e)
            continue

    return {
        "matched_terms": matched_terms,
        "untouched_entries": untouched,
        "applied_entries": len(entries) - untouched,
        "term_updates": term_updates,
    }
