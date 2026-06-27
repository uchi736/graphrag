"""ナレッジグラフスキーマのロードと Neo4j へのメタ刻印

現時点ではハードコードの12関係を返すだけだが、`SHARED_SCHEMA_PATH` が
指す JSON が存在すれば EDC 等の外部ツールが発見したスキーマを使う。
将来の EDC 連携時はこのモジュール経由で差し替わる。

スキーマJSON形式（EDC側のエクスポート契約）:
{
  "domain": "manufacturing/gas_bearings",
  "edc_version": "v0.3.0",
  "generated_at": "2026-04-18T10:00:00Z",
  "node_types": ["Term"],
  "relations": [
    {"name": "BEARS_LOAD", "definition": "...", "examples": [...]}
  ]
}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from graphrag_core.config import get_settings

logger = logging.getLogger(__name__)


# 現行のデフォルト12関係（EDCが無い場合のフォールバック）
DEFAULT_RELATIONS: List[str] = [
    "IS_A", "BELONGS_TO_CATEGORY", "PART_OF", "HAS_STEP",
    "HAS_ATTRIBUTE", "RELATED_TO", "AFFECTS", "CAUSES",
    "DEPENDS_ON", "APPLIES_TO", "OWNED_BY", "SAME_AS",
]
DEFAULT_NODE_TYPES: List[str] = ["Term"]
DEFAULT_DOMAIN: str = "default/builtin"
DEFAULT_VERSION: str = "builtin-v1"


def entity_node_predicate(var: str) -> str:
    """エンティティノード判定の共通Cypher述語（ラベル非依存）。

    KGのノードタイプは外部スキーマで差し替わる（Term だけとは限らない）ため、
    ラベルのホワイトリストではなく「チャンク/管理ノードの除外」で判定する:
    - id を持たないノード（ProcessedChunk 等）を除外
    - 管理ノード (ProcessedChunk, SchemaMeta) を除外
    - チャンクノード（idが32桁以上のhex）を除外
    - 値ノード（is_value=true: 数値・日付のみのノード、consolidate.pyで付与）を除外。
      値ノードはトラバーサルのノイズ源で、数値の根拠はチャンク側にあるため
      KGの検索・enrichment・entity vectorの対象にしない
    - 照応ノード（is_anaphor=true: 「本製品」「当社」等で解決不能なもの）を除外。
      複数文書の別対象を1ノードに偽統合した誤ハブになるため
    """
    base = (
        f"{var}.id IS NOT NULL "
        f"AND NOT {var}:ProcessedChunk AND NOT {var}:SchemaMeta "
        f"AND NOT {var}:GraphProvenance "
        f"AND NOT {var}.id =~ '[0-9a-f]{{32,}}' "
        f"AND COALESCE({var}.is_value, false) = false "
        f"AND COALESCE({var}.is_anaphor, false) = false"
    )
    # 条件付き関係(reify)のノードは実体検索/pagerank/mention/value-flag から除外する。
    # フラグOFF時は何も足さない＝文字列は byte-identical（高fanout関数なので不変性が重要）。
    try:
        if get_settings().enable_conditional_relations:
            base += f" AND NOT {var}:CondFact AND NOT {var}:Cond"
    except Exception:
        pass
    return base


def _schema_path() -> Optional[Path]:
    """SHARED_SCHEMA_PATH の解決。相対なら graphrag ルート基準"""
    s = get_settings()
    raw = (s.shared_schema_path or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = (Path(__file__).resolve().parents[2] / raw).resolve()
    return p


def load_schema() -> Dict:
    """共有スキーマJSONを読み、無ければデフォルトを返す

    Returns:
        {"domain", "version", "generated_at", "node_types", "relations"}
    """
    path = _schema_path()
    if path and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            relations = data.get("relations", []) or []
            names = [r["name"] for r in relations if isinstance(r, dict) and r.get("name")]
            if names:
                logger.info("Loaded external schema from %s (%d relations)", path, len(names))
                return {
                    "domain": data.get("domain", "external/unknown"),
                    "version": data.get("edc_version") or data.get("version") or "external",
                    "generated_at": data.get("generated_at"),
                    "node_types": data.get("node_types") or DEFAULT_NODE_TYPES,
                    "relations": names,
                    "source": str(path),
                }
            logger.warning("External schema %s has no relations, fallback to default", path)
        except Exception as e:
            logger.warning("Failed to load external schema %s: %s (fallback to default)", path, e)

    return {
        "domain": DEFAULT_DOMAIN,
        "version": DEFAULT_VERSION,
        "generated_at": None,
        "node_types": list(DEFAULT_NODE_TYPES),
        "relations": list(DEFAULT_RELATIONS),
        "source": None,
    }


def get_allowed_relations() -> List[str]:
    """LLMGraphTransformer の allowed_relationships に渡すリスト"""
    return load_schema()["relations"]


def get_allowed_node_types() -> List[str]:
    """LLMGraphTransformer の allowed_nodes に渡すリスト"""
    return load_schema()["node_types"]


def stamp_schema_metadata(graph) -> None:
    """Neo4j に :SchemaMeta ノードで現在のスキーマ情報を刻印する（ビルド直後に呼ぶ）"""
    schema = load_schema()
    try:
        graph.query(
            """
            MERGE (s:SchemaMeta {kind: 'active'})
            SET s.domain = $domain,
                s.version = $version,
                s.source = $source,
                s.relations = $relations,
                s.node_types = $node_types,
                s.stamped_at = datetime($stamped_at)
            """,
            params={
                "domain": schema["domain"],
                "version": schema["version"],
                "source": schema.get("source") or "",
                "relations": schema["relations"],
                "node_types": schema["node_types"],
                "stamped_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info("Stamped SchemaMeta: %s/%s (%d relations)",
                    schema["domain"], schema["version"], len(schema["relations"]))
    except Exception as e:
        logger.warning("Failed to stamp SchemaMeta: %s", e)


def describe_schema() -> str:
    """人間可読な1行要約（UI表示用）"""
    s = load_schema()
    rel_cnt = len(s["relations"])
    ts = s.get("generated_at") or ""
    if s["source"]:
        return f"{s['domain']} / {s['version']} ({rel_cnt}関係){' 生成:' + ts if ts else ''}"
    return f"デフォルト ({rel_cnt}関係、組み込み)"
