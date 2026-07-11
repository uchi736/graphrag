"""グラフ探索サービス（ui/graph_tab.py から移設・st非依存化）。

- get_enhanced_graph_data / get_enhanced_subgraph_data: 可視化・テーブル用エッジ取得
- search_node_ids: 中心ノード選択用の部分一致検索
- natural_language_to_cypher: NL→Cypher（LLM 1回）
- execute_readonly_cypher: 参照専用実行（書込語彙 reject + LIMIT 500 自動付与）

エラーは raise する（Streamlit 側は従来どおり try/except、API 側は HTTP 化）。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from graphrag_core.llm.factory import create_chat_llm
from graphrag_core.llm.langfuse_utils import get_langfuse_callback, observe
from graphrag_core.prompts import NL_TO_CYPHER_PROMPT
from graphrag_core.graph.schema import chunk_edge, chunk_label


class WriteQueryRejected(ValueError):
    """参照専用エンドポイントに書き込み系 Cypher が投入された。"""


def get_enhanced_graph_data(graph, limit: int = 200) -> List[Dict]:
    """全体エッジリスト（degree / 言及ドキュメント付き）。"""
    if hasattr(graph, "get_graph_data"):
        return graph.get_graph_data(limit=limit)
    limit = max(1, min(int(limit), 20000))
    query = f"""
    MATCH (n)-[r]->(m)
    WHERE type(r) <> '{chunk_edge()}'
    AND NOT n.id =~ '[0-9a-f]{{32,}}'
    AND NOT m.id =~ '[0-9a-f]{{32,}}'
    OPTIONAL MATCH (n)<-[:{chunk_edge()}]-(doc_n:{chunk_label()})
    OPTIONAL MATCH (m)<-[:{chunk_edge()}]-(doc_m:{chunk_label()})
    WITH n, r, m, labels(n) as source_labels, labels(m) as target_labels,
         COLLECT(DISTINCT doc_n.source) AS source_docs,
         COLLECT(DISTINCT doc_m.source) AS target_docs
    RETURN
      n.id AS source,
      CASE WHEN size(source_labels) > 0 THEN source_labels[0] ELSE 'Unknown' END AS source_type,
      type(r) AS relation,
      m.id AS target,
      CASE WHEN size(target_labels) > 0 THEN target_labels[0] ELSE 'Unknown' END AS target_type,
      COUNT {{ (n)--() }} AS source_degree,
      COUNT {{ (m)--() }} AS target_degree,
      source_docs,
      target_docs
    LIMIT {limit}
    """
    return graph.query(query)


def get_enhanced_subgraph_data(graph, center_nodes: List[str], hop: int = 1,
                               limit: int = 500) -> List[Dict]:
    """指定ノード群を中心に N-hop 範囲のエッジを取得（hop は 1-3 にクランプ）。"""
    hop = max(1, min(int(hop), 3))
    limit = max(1, min(int(limit), 2000))
    query = f"""
    MATCH (c) WHERE c.id IN $entities
    MATCH (c)-[*1..{hop}]-(n)
    MATCH (n)-[r]->(m)
    WHERE type(r) <> '{chunk_edge()}'
      AND NOT n.id =~ '[0-9a-f]{{32,}}' AND NOT m.id =~ '[0-9a-f]{{32,}}'
    RETURN DISTINCT
      n.id AS source,
      COALESCE(labels(n)[0], 'Unknown') AS source_type,
      type(r) AS relation,
      m.id AS target,
      COALESCE(labels(m)[0], 'Unknown') AS target_type
    LIMIT {limit}
    """
    results = graph.query(query, params={"entities": center_nodes})
    return [
        {
            "source": r.get("source", ""),
            "source_type": r.get("source_type", "Unknown"),
            "target": r.get("target", ""),
            "target_type": r.get("target_type", "Unknown"),
            "relation": r.get("relation", "RELATED"),
            "edge_key": 0,
            "source_degree": 0,
            "target_degree": 0,
            "source_docs": [],
            "target_docs": [],
        }
        for r in (results or [])
    ]


def search_node_ids(graph, q: str = "", limit: int = 100) -> List[str]:
    """ノードIDの部分一致検索（中心ノード選択用）。q空なら先頭からlimit件。"""
    limit = max(1, min(int(limit), 500))
    if q:
        rows = graph.query(
            "MATCH (n) WHERE n.id IS NOT NULL AND NOT n:" + chunk_label() + " AND NOT n:ProcessedChunk "
            "AND NOT n.id =~ '[0-9a-f]{32,}' "
            "AND toLower(n.id) CONTAINS toLower($q) "
            "RETURN n.id AS id ORDER BY size(n.id) LIMIT $limit",
            params={"q": q, "limit": limit},
        )
    else:
        rows = graph.query(
            "MATCH (n) WHERE n.id IS NOT NULL AND NOT n:" + chunk_label() + " AND NOT n:ProcessedChunk "
            "AND NOT n.id =~ '[0-9a-f]{32,}' "
            "RETURN n.id AS id ORDER BY n.id LIMIT $limit",
            params={"limit": limit},
        )
    return [r["id"] for r in (rows or [])]


@observe(name="cypher_generation")
def natural_language_to_cypher(query: str, llm=None) -> str:
    """自然言語→Cypher 変換（コードフェンス除去込み）。"""
    llm = llm or create_chat_llm(temperature=0)
    prompt = NL_TO_CYPHER_PROMPT.format(query=query)
    response = llm.invoke(prompt, config=get_langfuse_callback())
    cypher_query = response.content.strip()
    if cypher_query.startswith("```"):
        lines = cypher_query.split("\n")
        cypher_query = "\n".join(lines[1:-1]) if len(lines) > 2 else cypher_query
    return cypher_query


def execute_readonly_cypher(graph, cypher_query: str) -> Dict[str, Any]:
    """参照専用 Cypher 実行。

    - 書き込み系キーワード（単語境界）を検出したら WriteQueryRejected
    - LIMIT 無しなら LIMIT 500 を自動付与
    Returns: {"rows": [...], "applied_limit": bool, "executed_query": str}
    """
    if re.search(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DETACH|DROP|CALL|LOAD)\b",
                 cypher_query, flags=re.IGNORECASE):
        raise WriteQueryRejected(
            "参照クエリ(MATCH/RETURN)のみ実行できます。書き込み系キーワードが含まれています。")
    run_query = cypher_query
    applied_limit = False
    if not re.search(r"\bLIMIT\b", cypher_query, flags=re.IGNORECASE):
        run_query = cypher_query.rstrip().rstrip(";") + " LIMIT 500"
        applied_limit = True
    rows = graph.query(run_query) or []
    return {"rows": rows, "applied_limit": applied_limit, "executed_query": run_query}
