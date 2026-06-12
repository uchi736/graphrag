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
