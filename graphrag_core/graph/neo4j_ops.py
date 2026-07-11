"""
Neo4j CRUD Operations
=====================
Neo4j グラフバックエンド用の CRUD 関数群。
app.py から抽出し、session_state 依存を排除した純粋関数版。

全関数は第一引数に Neo4j ドライバ (graph) を受け取る。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_label(name: Any) -> str:
    """Cypher のバッククォート識別子に埋め込むラベル/リレーションタイプを安全化する。

    enrichment.attach_source_chunks の安全化イディオムと同一:
    英数字 (`str.isalnum()`) とアンダースコアのみを許可する。CJK 文字は
    `isalnum()` が True を返すため日本語ラベル (Fujitsu 8ラベル等) は保持され、
    バッククォート/空白/記号などクエリを破壊する文字のみが除去される。

    Returns:
        安全化した識別子。除去後に空文字なら ""（呼び出し側はクエリを実行せず失敗扱いにする）。
    """
    return "".join(c for c in str(name) if c.isalnum() or c == "_")


# ------------------------------------------------------------------
# ノード操作
# ------------------------------------------------------------------

def neo4j_add_node(graph, node_id: str, node_type: str = "Term", properties: Optional[Dict[str, Any]] = None) -> bool:
    """Neo4j: ノード追加

    Args:
        graph: Neo4j グラフドライバ
        node_id: ノードID
        node_type: ノードラベル
        properties: ノードプロパティ

    Returns:
        成功したら True
    """
    safe_type = _safe_label(node_type)
    if not safe_type:
        logger.warning("neo4j_add_node: invalid node_type %r for id=%s; aborting", node_type, node_id)
        return False
    try:
        props = dict(properties) if properties else {}
        props["id"] = node_id
        graph.query(
            f"MERGE (n:`{safe_type}` {{id: $id}}) SET n += $props",
            {"id": node_id, "props": props},
        )
        return True
    except Exception as e:
        logger.warning("neo4j_add_node failed for id=%s type=%s: %s", node_id, safe_type, e)
        return False


def neo4j_update_node(graph, node_id: str, node_type: Optional[str] = None, properties: Optional[Dict[str, Any]] = None) -> bool:
    """Neo4j: ノード更新

    Args:
        graph: Neo4j グラフドライバ
        node_id: ノードID
        node_type: 新しいノードラベル (None なら変更なし)
        properties: 新しいプロパティ (None なら変更なし)

    Returns:
        成功したら True
    """
    props = dict(properties) if properties else {}
    if node_type:
        safe_type = _safe_label(node_type)
        if not safe_type:
            logger.warning("neo4j_update_node: invalid node_type %r for id=%s; aborting", node_type, node_id)
            return False
        # ラベル変更が無い場合は APOC 不要のプロパティ更新で済ませる。
        # 編集ダイアログは type を常に再送するため、プロパティのみ編集でも node_type が渡る。
        # ここで短絡しないと APOC 非搭載の Neo4j では全ノード編集が失敗扱い (False) になる。
        current = neo4j_get_node_info(graph, node_id)
        current_type = current.get("type") if current else None
        if current_type == node_type:
            try:
                graph.query(
                    "MATCH (n {id: $id}) SET n += $props",
                    {"id": node_id, "props": props},
                )
                return True
            except Exception as e:
                logger.warning("neo4j_update_node failed (props-only, label unchanged) for id=%s: %s", node_id, e)
                return False
        try:
            # ラベル変更: 既存ラベルを削除して新ラベルを付与
            graph.query(
                "MATCH (n {id: $id}) "
                "WITH n, labels(n) AS lbls "
                "CALL { WITH n, lbls UNWIND lbls AS l WITH n, l WHERE l <> 'ProcessedChunk' CALL apoc.node.removeLabel(n, l) YIELD node RETURN count(*) } "
                f"SET n:`{safe_type}`, n += $props",
                {"id": node_id, "props": props},
            )
            return True
        except Exception as e:
            # APOC未使用などでラベル変更ができなかった場合、プロパティのみ更新を試みるが、
            # ラベル変更が反映されていないことを明示し、失敗 (False) を返す。
            # （以前はラベル変更を黙って捨てつつ True を返す silent no-op だった）
            logger.warning(
                "neo4j_update_node: label change to %s skipped for id=%s (APOC unavailable?): %s",
                safe_type, node_id, e,
            )
            try:
                graph.query(
                    "MATCH (n {id: $id}) SET n += $props",
                    {"id": node_id, "props": props},
                )
                logger.warning(
                    "neo4j_update_node: properties updated for id=%s but requested label %s was NOT applied",
                    node_id, safe_type,
                )
            except Exception as e2:
                logger.warning("neo4j_update_node: property fallback also failed for id=%s: %s", node_id, e2)
            return False
    else:
        try:
            graph.query(
                "MATCH (n {id: $id}) SET n += $props",
                {"id": node_id, "props": props},
            )
            return True
        except Exception as e:
            logger.warning("neo4j_update_node failed (props-only) for id=%s: %s", node_id, e)
            return False


def neo4j_delete_node(graph, node_id: str) -> bool:
    """Neo4j: ノード削除（関連エッジも削除）

    Args:
        graph: Neo4j グラフドライバ
        node_id: ノードID

    Returns:
        成功したら True
    """
    try:
        graph.query("MATCH (n {id: $id}) DETACH DELETE n", {"id": node_id})
        return True
    except Exception as e:
        logger.warning("neo4j_delete_node failed for id=%s: %s", node_id, e)
        return False


def neo4j_get_node_info(graph, node_id: str) -> Optional[Dict[str, Any]]:
    """Neo4j: ノード情報取得

    Args:
        graph: Neo4j グラフドライバ
        node_id: ノードID

    Returns:
        ノード情報の辞書、存在しなければ None
    """
    try:
        result = graph.query(
            "MATCH (n {id: $id}) "
            "RETURN n.id AS id, labels(n) AS labels, properties(n) AS props, "
            "COUNT { (n)--() } AS degree",
            {"id": node_id},
        )
        if result:
            r = result[0]
            labels = [l for l in r.get("labels", []) if l not in ("ProcessedChunk", "Chunk")]
            props = dict(r.get("props", {}))
            props.pop("id", None)
            return {
                "id": r["id"],
                "type": labels[0] if labels else "Unknown",
                "properties": props,
                "degree": r.get("degree", 0),
            }
    except Exception as e:
        logger.warning("neo4j_get_node_info failed for id=%s: %s", node_id, e)
    return None


# ------------------------------------------------------------------
# エッジ操作
# ------------------------------------------------------------------

def neo4j_add_edge(graph, source: str, target: str, rel_type: str = "RELATED", properties: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Neo4j: エッジ追加

    Args:
        graph: Neo4j グラフドライバ
        source: 始点ノードID
        target: 終点ノードID
        rel_type: リレーションタイプ
        properties: エッジプロパティ

    Returns:
        edge_key 相当 (0)、失敗時は None
    """
    safe_type = _safe_label(rel_type)
    if not safe_type:
        logger.warning("neo4j_add_edge: invalid rel_type %r for (%s)->(%s); aborting", rel_type, source, target)
        return None
    try:
        props = dict(properties) if properties else {}
        graph.query(
            f"MATCH (s {{id: $src}}), (t {{id: $tgt}}) CREATE (s)-[r:`{safe_type}`]->(t) SET r += $props",
            {"src": source, "tgt": target, "props": props},
        )
        return 0  # edge_key相当
    except Exception as e:
        logger.warning("neo4j_add_edge failed for (%s)-[%s]->(%s): %s", source, safe_type, target, e)
        return None


def neo4j_update_edge(graph, source: str, target: str, edge_key: int = 0, rel_type: Optional[str] = None, properties: Optional[Dict[str, Any]] = None) -> bool:
    """Neo4j: エッジ更新（type 変更は delete+create で実現）

    Args:
        graph: Neo4j グラフドライバ
        source: 始点ノードID
        target: 終点ノードID
        edge_key: エッジキー (Neo4j では未使用、互換性のため)
        rel_type: 新しいリレーションタイプ (None なら変更なし)
        properties: 新しいプロパティ (None なら変更なし)

    Returns:
        成功したら True
    """
    try:
        # 既存のエッジ情報取得
        old_info = neo4j_get_edge_info(graph, source, target, edge_key)
        if not old_info:
            logger.warning("neo4j_update_edge: edge (%s)->(%s) not found; aborting", source, target)
            return False
        old_type = _safe_label(old_info.get("type", "RELATED"))
        new_type = _safe_label(rel_type) if rel_type else old_type
        if not old_type or not new_type:
            logger.warning(
                "neo4j_update_edge: invalid rel_type (old=%r new=%r) for (%s)->(%s); aborting",
                old_info.get("type"), rel_type, source, target,
            )
            return False
        props = dict(properties) if properties else {}

        if new_type != old_type:
            # タイプ変更: 既存削除→新規作成
            graph.query(
                f"MATCH (s {{id: $src}})-[r:`{old_type}`]->(t {{id: $tgt}}) DELETE r",
                {"src": source, "tgt": target},
            )
            graph.query(
                f"MATCH (s {{id: $src}}), (t {{id: $tgt}}) CREATE (s)-[r:`{new_type}`]->(t) SET r += $props",
                {"src": source, "tgt": target, "props": props},
            )
        else:
            graph.query(
                f"MATCH (s {{id: $src}})-[r:`{old_type}`]->(t {{id: $tgt}}) SET r += $props",
                {"src": source, "tgt": target, "props": props},
            )
        return True
    except Exception as e:
        logger.warning(
            "neo4j_update_edge failed for (%s)-[%s->%s]->(%s): %s",
            source, old_info.get("type") if isinstance(old_info, dict) else "?", rel_type, target, e,
        )
        return False


def neo4j_delete_edge(graph, source: str, target: str, edge_key: Optional[int] = None) -> bool:
    """Neo4j: エッジ削除

    Args:
        graph: Neo4j グラフドライバ
        source: 始点ノードID
        target: 終点ノードID
        edge_key: エッジキー (Neo4j では未使用、互換性のため)

    Returns:
        成功したら True
    """
    try:
        graph.query(
            "MATCH (s {id: $src})-[r]->(t {id: $tgt}) DELETE r",
            {"src": source, "tgt": target},
        )
        return True
    except Exception as e:
        logger.warning("neo4j_delete_edge failed for (%s)->(%s): %s", source, target, e)
        return False


def neo4j_get_edge_info(graph, source: str, target: str, edge_key: int = 0) -> Optional[Dict[str, Any]]:
    """Neo4j: エッジ情報取得

    Args:
        graph: Neo4j グラフドライバ
        source: 始点ノードID
        target: 終点ノードID
        edge_key: エッジキー (Neo4j では未使用、互換性のため)

    Returns:
        エッジ情報の辞書、存在しなければ None
    """
    try:
        result = graph.query(
            "MATCH (s {id: $src})-[r]->(t {id: $tgt}) "
            "RETURN s.id AS source, t.id AS target, type(r) AS type, properties(r) AS props",
            {"src": source, "tgt": target},
        )
        if result:
            r = result[0]
            return {
                "source": r["source"],
                "target": r["target"],
                "type": r.get("type", "RELATED"),
                "properties": dict(r.get("props", {})),
                "edge_key": 0,
            }
    except Exception as e:
        logger.warning("neo4j_get_edge_info failed for (%s)->(%s): %s", source, target, e)
    return None


# ------------------------------------------------------------------
# リスト取得
# ------------------------------------------------------------------

def neo4j_list_all_nodes(graph) -> List[Dict[str, Any]]:
    """Neo4j: 全ノード一覧

    Args:
        graph: Neo4j グラフドライバ

    Returns:
        ノード情報のリスト
    """
    try:
        result = graph.query(
            "MATCH (n) "
            "WHERE NOT n:ProcessedChunk AND NOT n:Chunk "
            "AND NOT n.id =~ '[0-9a-f]{32,}' "
            "RETURN n.id AS id, labels(n) AS labels, COUNT { (n)--() } AS degree "
            "ORDER BY degree DESC LIMIT 500"
        )
        nodes = []
        for r in (result or []):
            labels = [l for l in r.get("labels", []) if l not in ("ProcessedChunk", "Chunk")]
            nodes.append({
                "id": r["id"],
                "type": labels[0] if labels else "Unknown",
                "degree": r.get("degree", 0),
            })
        return nodes
    except Exception as e:
        logger.warning("neo4j_list_all_nodes failed: %s", e)
        return []


def neo4j_list_all_edges(graph) -> List[Dict[str, Any]]:
    """Neo4j: 全エッジ一覧

    Args:
        graph: Neo4j グラフドライバ

    Returns:
        エッジ情報のリスト
    """
    try:
        result = graph.query(
            "MATCH (s)-[r]->(t) "
            "WHERE type(r) <> 'MENTIONS' "
            "AND NOT s.id =~ '[0-9a-f]{32,}' AND NOT t.id =~ '[0-9a-f]{32,}' "
            "RETURN s.id AS source, type(r) AS type, t.id AS target "
            "LIMIT 500"
        )
        return [
            {"source": r["source"], "type": r["type"], "target": r["target"], "edge_key": 0}
            for r in (result or [])
        ]
    except Exception as e:
        logger.warning("neo4j_list_all_edges failed: %s", e)
        return []


# ------------------------------------------------------------------
# JSON エクスポート
# ------------------------------------------------------------------

def export_graph_json(graph, output_path: str = "graph.json") -> dict:
    """Neo4j の全ノード・全エッジを graph.json 形式でエクスポートする。

    出力形式は旧 NetworkX (node_link_data) と同等:
    {
      "graph": {
        "directed": true,
        "multigraph": true,
        "nodes": [{"id": "..."}, ...],
        "edges": [{"source": "A", "target": "B", "type": "RELATED"}, ...]
      }
    }

    Args:
        graph: Neo4j グラフドライバ
        output_path: 出力先パス（None ならファイル書き出しをスキップ）

    Returns:
        エクスポートされた dict
    """
    import json

    # ノード取得（ProcessedChunk/Chunk とハッシュIDを除外）
    try:
        node_result = graph.query(
            "MATCH (n) "
            "WHERE NOT n:ProcessedChunk AND NOT n:Chunk "
            "AND NOT n.id =~ '[0-9a-f]{32,}' "
            "RETURN n.id AS id, labels(n) AS labels, properties(n) AS props "
            "ORDER BY n.id"
        )
    except Exception as e:
        logger.warning("export_graph_json: node query failed, exporting empty nodes: %s", e)
        node_result = []

    nodes = []
    for r in (node_result or []):
        node = {"id": r["id"]}
        labels = [l for l in r.get("labels", []) if l not in ("ProcessedChunk", "Chunk")]
        if labels:
            node["type"] = labels[0]
        props = dict(r.get("props", {}))
        props.pop("id", None)
        if props:
            node.update(props)
        nodes.append(node)

    # エッジ取得（MENTIONS を除外、ハッシュIDノードを除外）
    try:
        edge_result = graph.query(
            "MATCH (s)-[r]->(t) "
            "WHERE type(r) <> 'MENTIONS' "
            "AND NOT s.id =~ '[0-9a-f]{32,}' AND NOT t.id =~ '[0-9a-f]{32,}' "
            "RETURN s.id AS source, type(r) AS type, t.id AS target, properties(r) AS props "
            "ORDER BY s.id, t.id"
        )
    except Exception as e:
        logger.warning("export_graph_json: edge query failed, exporting empty edges: %s", e)
        edge_result = []

    edges = []
    _edge_keys: dict = {}  # (u, v) -> 重複時の連番
    for r in (edge_result or []):
        u, v = r["source"], r["target"]
        key = _edge_keys.get((u, v), 0)
        _edge_keys[(u, v)] = key + 1
        edge = {
            "source": u,
            "target": v,
            "type": r.get("type", "RELATED"),
            "key": key,  # MultiDiGraph 互換のためエッジIDを付与
        }
        props = dict(r.get("props", {}))
        if props:
            edge.update(props)
        edges.append(edge)

    data = {
        "graph": {
            "directed": True,
            "multigraph": True,
            "nodes": nodes,
            "edges": edges,
        }
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return data
