"""KG後処理: ノード・関係の統合と正規化

スキーマ診断 (_bench/_schema_diag.py, 2026-06-11) で確認された構造問題を
ビルド後処理で解消する:

1. flag_value_nodes()
   数値・日付・年度のみのノード（5,715件確認）に `is_value=true` を付与。
   値はトラバーサル・entity vector・enrichmentの対象から除外する
   （削除はしない: 可逆性を保ち、HAS_ATTRIBUTE経由の表示用途は残す）。
   ベンチ実証: KGが負けるのは数値表・時系列質問で、値ノードはそのノイズ源。

2. merge_duplicate_id_nodes()
   同一idが複数ラベルに分裂したノード（2,600 id / 5,557ノード確認）を
   次数最大のノードへマージ。LLMGraphTransformerが (id, type) 単位で
   ノードを作るために起きる「型揺れ分裂」で、エッジ・pagerank・
   extraction_count が薄まる名寄せ問題の最大成分。

3. normalize_relations()
   逆方向ペア（HAS_PART/PART_OF等）・同義関係（HAS_VALUE→HAS_ATTRIBUTE等）・
   typo関係（MANAGES_BY等）を正規タイプへ統合。検索Cypherは無向マッチの
   ため逆関係は情報量ゼロで、重複とextraction_count分散だけを生む。

実行後は enrichment.enrich_post_build() の再実行が必要
（mention_count / pagerank / search_keys がエッジ移動で変わるため）。
"""

from __future__ import annotations

import logging
import re

from graphrag_core.graph.schema import entity_node_predicate, chunk_edge, chunk_label

logger = logging.getLogger(__name__)


# ── 1. 値ノードのフラグ付け ──────────────────────────────────────────

# id全体が数値・日付・単位・年号のみで構成されるノードを「値ノード」と判定
VALUE_NODE_REGEXES = [
    # 数値・単位・記号のみ（例: "210円", "53", "1,000", "約10時間", "8.1%"）
    r"^[0-9０-９,，.．%％~〜+±▲△一-九十百千万億兆年月日円人件台時分秒回歳割約-]+$",
    # 年号年度（例: "令和6年度", "平成30年", "令和元年度"）
    r"^(令和|平成|昭和)[0-9０-９元一-九十]*(年度?|年)?$",
    # 西暦の期・四半期（例: "2024年3月期", "2025年度第3四半期"）
    r"^[0-9０-９]{4}年?[0-9０-９]*月?(期|度)?(第[0-9０-９一-四]四半期)?$",
]


def flag_value_nodes(graph) -> int:
    """値ノードに is_value=true を付与する。

    Returns:
        フラグを付けたノード数
    """
    _pred = entity_node_predicate("n")
    regex_clause = " OR ".join(f"n.id =~ '{rx}'" for rx in VALUE_NODE_REGEXES)
    try:
        graph.query(
            f"MATCH (n) WHERE {_pred} AND ({regex_clause}) SET n.is_value = true"
        )
        r = graph.query("MATCH (n) WHERE n.is_value = true RETURN count(n) AS c")
        n = r[0]["c"] if r else 0
        logger.info("flag_value_nodes: %d nodes flagged", n)
        return n
    except Exception as e:
        logger.error("flag_value_nodes failed: %s", e)
        return 0


# ── 2. 型分裂ノードのマージ ──────────────────────────────────────────

def _move_rels(graph, dup_eid: str, keep_eid: str, rel_type: str, outgoing: bool) -> None:
    """dupノードの指定タイプのエッジをkeeperへ付け替える（source_chunks合算）"""
    arrow_l, arrow_r = ("-", "->") if outgoing else ("<-", "-")
    graph.query(
        f"""
        MATCH (d){arrow_l}[r:`{rel_type}`]{arrow_r}(m)
        WHERE elementId(d) = $dup AND elementId(m) <> $keep
        MATCH (k) WHERE elementId(k) = $keep
        MERGE (k){arrow_l}[nr:`{rel_type}`]{arrow_r}(m)
        ON CREATE SET nr = properties(r)
        ON MATCH SET
            nr.extraction_count =
                COALESCE(nr.extraction_count, 0) + COALESCE(r.extraction_count, 1),
            nr.source_chunks = COALESCE(nr.source_chunks, []) +
                [x IN COALESCE(r.source_chunks, []) WHERE NOT x IN COALESCE(nr.source_chunks, [])]
        DELETE r
        """,
        {"dup": dup_eid, "keep": keep_eid},
    )


def merge_duplicate_id_nodes(graph) -> dict:
    """同一idで複数ノードに分裂しているエンティティを1ノードへマージする。

    keeper = 次数（MENTIONS含む）最大のノード。dupの全エッジをkeeperへ
    付け替え、dupのラベルをkeeperに追加してから dup を削除する。

    Returns:
        {"merged_ids": マージしたid数, "removed_nodes": 削除ノード数}
    """
    _pred = entity_node_predicate("n")
    groups = graph.query(
        f"""
        MATCH (n) WHERE {_pred}
        WITH n, COUNT {{ (n)--() }} AS deg
        WITH n.id AS id, collect({{eid: elementId(n), labels: labels(n), deg: deg}}) AS nodes
        WHERE size(nodes) > 1
        RETURN id, nodes
        """
    )
    merged_ids = 0
    removed = 0
    for g_row in groups:
        nodes = sorted(g_row["nodes"], key=lambda x: -x["deg"])
        keep = nodes[0]
        keep_labels = set(keep["labels"])
        for dup in nodes[1:]:
            try:
                # エッジタイプを列挙して方向別に付け替え
                out_types = graph.query(
                    "MATCH (d)-[r]->() WHERE elementId(d) = $dup "
                    "RETURN DISTINCT type(r) AS t",
                    {"dup": dup["eid"]},
                )
                for row in out_types:
                    _move_rels(graph, dup["eid"], keep["eid"], row["t"], outgoing=True)
                in_types = graph.query(
                    "MATCH (d)<-[r]-() WHERE elementId(d) = $dup "
                    "RETURN DISTINCT type(r) AS t",
                    {"dup": dup["eid"]},
                )
                for row in in_types:
                    _move_rels(graph, dup["eid"], keep["eid"], row["t"], outgoing=False)

                # dupのラベルをkeeperへ追加（型情報を失わない）
                new_labels = [l for l in dup["labels"] if l not in keep_labels]
                if new_labels:
                    label_expr = "".join(f":`{l}`" for l in new_labels)
                    graph.query(
                        f"MATCH (k) WHERE elementId(k) = $keep SET k{label_expr}",
                        {"keep": keep["eid"]},
                    )
                    keep_labels.update(new_labels)

                graph.query(
                    "MATCH (d) WHERE elementId(d) = $dup DETACH DELETE d",
                    {"dup": dup["eid"]},
                )
                removed += 1
            except Exception as e:
                logger.warning("merge failed for id=%s dup=%s: %s",
                               g_row["id"], dup["eid"], e)
        merged_ids += 1
        if merged_ids % 200 == 0:
            logger.info("merge_duplicate_id_nodes: %d/%d ids processed",
                        merged_ids, len(groups))

    logger.info("merge_duplicate_id_nodes: merged %d ids, removed %d nodes",
                merged_ids, removed)
    return {"merged_ids": merged_ids, "removed_nodes": removed}


# ── 2.5. かな揺れノードのマージ（文字種限定レーベンシュタイン）───────

def _is_kana_variant_pair(a: str, b: str, max_diff: int = 3) -> bool:
    """2つの正規化済みエンティティ名が「かな揺れ」の関係か判定する。

    条件（すべて満たす場合のみ同一と見なす）:
    - 語頭1文字が一致（「この文書/文書」「すべてのアプリ/アプリ」を弾く）
    - 差分文字がひらがな・長音「ー」のみ
      （数字・英字・漢字の差分は距離1でも別物: U7314/U7414、第3/第4四半期、
       取締役/監査役 を構造的に排除する）
    - 差分文字数の合計が max_diff 以下
    """
    import difflib

    if not a or not b or a == b or a[0] != b[0]:
        return False
    # 長音「ー」は末尾のみ揺れと見なす（語中のーは音価がある: サーバー≠サバ）
    a2, b2 = a.rstrip("ー"), b.rstrip("ー")
    if a2 == b2:
        return True
    diff_chars = []
    sm = difflib.SequenceMatcher(None, a2, b2)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        diff_chars.extend(a2[i1:i2])
        diff_chars.extend(b2[j1:j2])
    if not diff_chars or len(diff_chars) > max_diff:
        return False
    return all("ぁ" <= c <= "ん" for c in diff_chars)


def merge_kana_variant_nodes(graph) -> dict:
    """かな揺れ（送り仮名・助詞・長音）だけが違うエンティティノードをマージする。

    例: ガス軸受/ガス軸受け、データの連携/データ連携、サーバ/サーバー。
    候補生成は kana_variant_key の骨格一致、最終判定は _is_kana_variant_pair
    （文字種限定の編集距離）の2段。埋め込み類似ベースのEL（FJH-06+ELで
    過剰マージ-5ptの実績）と異なり、誤統合クラスを文字種レベルで遮断する。

    マージされた側のidはkeeperの aliases に追加し、search_keys 再計算時に
    照合キーとして残す。

    Returns:
        {"merged_groups": マージしたグループ数, "removed_nodes": 削除ノード数}
    """
    from graphrag_core.text.japanese import kana_variant_key, normalize_entity_text

    _pred = entity_node_predicate("n")
    rows = graph.query(
        f"MATCH (n) WHERE {_pred} "
        "RETURN n.id AS id, elementId(n) AS eid, labels(n) AS labels, "
        "COUNT { (n)--() } AS deg"
    )

    # 候補生成: 骨格キーでグルーピング
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        k = kana_variant_key(r["id"])
        if k:
            groups[k].append(r)

    merged_groups = removed = 0
    for key, nodes in groups.items():
        if len(nodes) < 2:
            continue
        # ペア判定で連結成分を作る（正規化名で比較）
        norm = {n["eid"]: normalize_entity_text(n["id"]).replace(" ", "") for n in nodes}
        parent = {n["eid"]: n["eid"] for n in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i]["eid"], nodes[j]["eid"]
                if _is_kana_variant_pair(norm[a], norm[b]):
                    parent[find(a)] = find(b)

        comps = defaultdict(list)
        for n in nodes:
            comps[find(n["eid"])].append(n)

        for comp in comps.values():
            if len(comp) < 2:
                continue
            comp.sort(key=lambda x: -x["deg"])
            keep = comp[0]
            keep_labels = set(keep["labels"])
            for dup in comp[1:]:
                try:
                    for direction in (True, False):
                        types = graph.query(
                            f"MATCH (d){'-[r]->' if direction else '<-[r]-'}() "
                            "WHERE elementId(d) = $dup RETURN DISTINCT type(r) AS t",
                            {"dup": dup["eid"]},
                        )
                        for row in types:
                            _move_rels(graph, dup["eid"], keep["eid"], row["t"], direction)
                    new_labels = [l for l in dup["labels"] if l not in keep_labels]
                    if new_labels:
                        label_expr = "".join(f":`{l}`" for l in new_labels)
                        graph.query(
                            f"MATCH (k) WHERE elementId(k) = $keep SET k{label_expr}",
                            {"keep": keep["eid"]},
                        )
                        keep_labels.update(new_labels)
                    # マージされた表記をaliasesに保存（search_keys再計算で照合キー化）
                    graph.query(
                        """
                        MATCH (k) WHERE elementId(k) = $keep
                        SET k.aliases = COALESCE(k.aliases, []) +
                            CASE WHEN $alias IN COALESCE(k.aliases, [])
                                 THEN [] ELSE [$alias] END
                        """,
                        {"keep": keep["eid"], "alias": dup["id"]},
                    )
                    graph.query(
                        "MATCH (d) WHERE elementId(d) = $dup DETACH DELETE d",
                        {"dup": dup["eid"]},
                    )
                    removed += 1
                except Exception as e:
                    logger.warning("kana merge failed for %s: %s", dup["id"], e)
            merged_groups += 1

    logger.info("merge_kana_variant_nodes: %d groups merged, %d nodes removed",
                merged_groups, removed)
    return {"merged_groups": merged_groups, "removed_nodes": removed}


# ── 3. 関係タイプの正規化 ────────────────────────────────────────────

# {旧タイプ: (正規タイプ, 方向反転するか)}
# 確信度の高いもの（同義・逆関係・typo・出現5件以上）のみ。
# 否定系 (NOT_*, EXCLUDES, DOES_NOT_HAVE) は意味が変わるため対象外。
RELATION_NORMALIZE_MAP: dict[str, tuple[str, bool]] = {
    # 同義への統合
    "BELONGS_TO_CATEGORY": ("IS_A", False),
    "HAS_VALUE": ("HAS_ATTRIBUTE", False),
    "MEASURED_IN": ("HAS_ATTRIBUTE", False),
    "MITIGATES": ("PREVENTS", False),
    "DEPENDS_ON": ("REQUIRES", False),
    "COVERS": ("APPLIES_TO", False),
    "TARGETS": ("APPLIES_TO", False),
    "OPERATED_BY": ("MANAGED_BY", False),
    "GOVERNED_BY": ("MANAGED_BY", False),
    "PUBLISHED_BY": ("ISSUED_BY", False),
    "ENACTED_BY": ("ISSUED_BY", False),
    "REQUIRES_BEFORE": ("FOLLOWS", False),   # 「Aの前にBが必要」= AはBの後
    "REQUIRED_BEFORE": ("FOLLOWS", False),
    "RELATES_TO": ("RELATED_TO", False),
    # typo修正
    "MANAGES_BY": ("MANAGED_BY", False),
    "DEFINES_BY": ("DEFINED_BY", False),
    "DESCRIBES_IN": ("DESCRIBED_IN", False),
    # 逆方向の正規化（方向反転して正規タイプへ）
    "HAS_PART": ("PART_OF", True),
    "PRECEDES": ("FOLLOWS", True),
    "USED_BY": ("USES", True),
    "REQUIRED_BY": ("REQUIRES", True),
    "CONTAINS": ("PART_OF", True),
    "INCLUDES": ("PART_OF", True),
    "DEFINES": ("DEFINED_BY", True),
    "DESCRIBES": ("DESCRIBED_IN", True),
    "MANAGES": ("MANAGED_BY", True),
    "GOVERNS": ("MANAGED_BY", True),
}


def normalize_relations(graph, batch_size: int = 2000) -> dict:
    """RELATION_NORMALIZE_MAP に従って関係タイプを正規化する。

    既に正規タイプのエッジが同一ペアに存在する場合はマージ
    （extraction_count合算・source_chunks和集合）。

    スキーマガード: 外部スキーマ（SHARED_SCHEMA_PATH）使用時は
    - スキーマで許可されているタイプを別名へ改名しない
      （例: EDCスキーマでは GOVERNED_BY が正であり MANAGED_BY へ潰してはならない）
    - スキーマに無いタイプへの改名も行わない
    静的マップはfujitsu語彙前提のため、この2条件で対象を絞る。

    Returns:
        {旧タイプ: 処理エッジ数}
    """
    from graphrag_core.graph.schema import load_schema
    schema = load_schema()
    allowed_upper = ({r.upper() for r in schema.get("relations", [])}
                     if schema.get("source") else None)  # 外部スキーマ時のみガード

    stats: dict[str, int] = {}
    for old, (new, swap) in RELATION_NORMALIZE_MAP.items():
        if allowed_upper is not None:
            if old.upper() in allowed_upper:
                continue  # 許可タイプは正。改名しない
            if new.upper() not in allowed_upper:
                continue  # スキーマ外への改名はしない（埋め込み正規化に委ねる）
        moved = 0
        # MERGE方向: swap時は (b)->(a)
        merge_pat = "(b)-[nr:`%s`]->(a)" % new if swap else "(a)-[nr:`%s`]->(b)" % new
        try:
            while True:
                result = graph.query(
                    f"""
                    MATCH (a)-[r:`{old}`]->(b)
                    WITH a, r, b LIMIT $batch
                    MERGE {merge_pat}
                    ON CREATE SET nr = properties(r)
                    ON MATCH SET
                        nr.extraction_count =
                            COALESCE(nr.extraction_count, 0) + COALESCE(r.extraction_count, 1),
                        nr.source_chunks = COALESCE(nr.source_chunks, []) +
                            [x IN COALESCE(r.source_chunks, [])
                             WHERE NOT x IN COALESCE(nr.source_chunks, [])]
                    DELETE r
                    RETURN count(*) AS c
                    """,
                    {"batch": batch_size},
                )
                c = result[0]["c"] if result else 0
                moved += c
                if c < batch_size:
                    break
        except Exception as e:
            logger.warning("normalize_relations failed for %s: %s", old, e)
        if moved:
            stats[old] = moved
            logger.info("normalize_relations: %s -> %s%s (%d edges)",
                        old, new, " (reversed)" if swap else "", moved)
    return stats


# ── 3.5. スキーマ語彙への埋め込み正規化（EDCのCanonicalizeと同思想） ──

def canonicalize_relations_to_schema(graph, embeddings=None, llm=None, *,
                                     threshold: float = 0.80,
                                     shortlist_k: int = 3,
                                     protect_label_prefix: str = "Qms",
                                     dry_run: bool = False,
                                     batch_size: int = 2000) -> dict:
    """スキーマ外の関係タイプを、EDCのCanonicalizeと同じ2段構成でスキーマ語彙へ名寄せする。

    1. 埋め込み検索: 野良タイプ名 vs スキーマ関係「名前+定義文」の類似度で
       上位 shortlist_k 候補に絞る（実測で正誤が同スコアに並ぶため単独では使わない）
    2. LLM検証: 実例トリプルを見せて「同義の候補番号 or 該当なし」を判定
       （llm=None の場合は埋め込みのみ＝類似度 threshold 以上で採用）

    - 外部スキーマ（SHARED_SCHEMA_PATH）が無い場合は何もしない
    - 方向反転を伴う正規化（HAS_PART→PART_OF等）は判定できないため
      静的マップ側に残す
    - protect_label_prefix のラベルを持つノードに接続するエッジは対象外
      （共有Neo4j上の他プログラムのグラフを書き換えないため）

    Returns:
        {"mapping": {旧: {"to", "score", "edges", "by"}}, "unmatched": {...}, "applied": bool}
    """
    import json as _json

    import numpy as np

    from graphrag_core.graph.schema import _schema_path, chunk_edge, load_schema

    schema = load_schema()
    if not schema.get("source"):
        return {"mapping": {}, "unmatched": {}, "applied": False}
    allowed = list(schema.get("relations", []))
    allowed_upper = {r.upper(): r for r in allowed}

    # スキーマJSONから定義文を取得（あれば埋め込みの手がかりにする）
    desc = {}
    try:
        raw = _json.loads(_schema_path().read_text(encoding="utf-8"))
        desc = {r["name"]: r.get("description", "") for r in raw.get("relations", [])
                if isinstance(r, dict) and r.get("name")}
    except Exception:
        pass

    skip = {chunk_edge().upper(), "MENTIONS", "REFERS_TO"}
    rows = graph.query(
        "MATCH ()-[r]->() RETURN DISTINCT type(r) AS t") or []
    strays = [r["t"] for r in rows
              if r["t"].upper() not in allowed_upper and r["t"].upper() not in skip]
    if not strays:
        return {"mapping": {}, "unmatched": {}, "applied": not dry_run}

    if embeddings is None:
        from graphrag_core.llm.factory import create_embeddings
        embeddings = create_embeddings()

    def _txt(name: str, d: str = "") -> str:
        base = name.lower().replace("_", " ")
        return f"{base}: {d}" if d else base

    schema_vecs = np.array(embeddings.embed_documents(
        [_txt(n, desc.get(n, "")) for n in allowed]))
    stray_vecs = np.array(embeddings.embed_documents([_txt(s) for s in strays]))

    def _cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    guard = (
        f"AND NOT any(l IN labels(a) WHERE l STARTS WITH '{protect_label_prefix}') "
        f"AND NOT any(l IN labels(b) WHERE l STARTS WITH '{protect_label_prefix}') "
    ) if protect_label_prefix else ""

    def _llm_pick(stray: str, candidates: list, examples: list) -> int:
        """LLMに同義候補を選ばせる。戻り値: candidates のindex、該当なしは -1。"""
        cand_lines = "\n".join(
            f"{k+1}. {name}: {desc.get(name, '')}" for k, (name, _) in enumerate(candidates))
        ex_lines = "\n".join(f"- ({a}) -[{stray}]-> ({b})" for a, b in examples) or "（例なし）"
        prompt = (
            "ナレッジグラフの関係タイプを正規スキーマ語彙へ統合する判定です。\n"
            f"対象の関係タイプ: {stray}\n実例:\n{ex_lines}\n\n"
            f"候補（正規語彙）:\n{cand_lines}\n\n"
            "対象の関係が候補のいずれかと同義（実例の主語→目的語の向きのまま置換しても"
            "意味が保たれる）なら、その番号だけを出力してください。"
            "どれとも同義でない、または向きが逆になる場合は 0 を出力してください。番号のみ。"
        )
        try:
            out = llm.invoke(prompt).content.strip()
            n = int("".join(c for c in out if c.isdigit()) or "0")
            return n - 1 if 1 <= n <= len(candidates) else -1
        except Exception as e:
            logger.warning("canonicalize LLM verify failed for %s: %s", stray, e)
            return -1

    mapping: dict = {}
    unmatched: dict = {}
    for i, stray in enumerate(strays):
        sims = [_cos(stray_vecs[i], sv) for sv in schema_vecs]
        order = sorted(range(len(allowed)), key=lambda j: -sims[j])[:shortlist_k]
        candidates = [(allowed[j], round(sims[j], 3)) for j in order]
        cnt_rows = graph.query(
            f"MATCH (a)-[r:`{stray}`]->(b) WHERE true {guard} RETURN count(r) AS c")
        n_edges = cnt_rows[0]["c"] if cnt_rows else 0
        if n_edges == 0:
            unmatched[stray] = {"nearest": candidates[0][0], "score": candidates[0][1],
                                "edges": 0}
            continue

        if llm is not None:
            ex_rows = graph.query(
                f"MATCH (a)-[r:`{stray}`]->(b) WHERE true {guard} "
                "RETURN a.id AS a, b.id AS b LIMIT 3") or []
            pick = _llm_pick(stray, candidates, [(r["a"], r["b"]) for r in ex_rows])
            if pick < 0:
                unmatched[stray] = {"nearest": candidates[0][0], "score": candidates[0][1],
                                    "edges": n_edges, "by": "llm_rejected"}
                continue
            best, score, by = candidates[pick][0], candidates[pick][1], "llm"
        else:
            best, score, by = candidates[0][0], candidates[0][1], "embedding"
            if score < threshold:
                unmatched[stray] = {"nearest": best, "score": score, "edges": n_edges}
                continue

        mapping[stray] = {"to": best, "score": score, "edges": n_edges, "by": by}
        if dry_run:
            continue
        try:
            while True:
                result = graph.query(
                    f"""
                    MATCH (a)-[r:`{stray}`]->(b) WHERE true {guard}
                    WITH a, r, b LIMIT $batch
                    MERGE (a)-[nr:`{best}`]->(b)
                    ON CREATE SET nr = properties(r)
                    ON MATCH SET
                        nr.extraction_count =
                            COALESCE(nr.extraction_count, 0) + COALESCE(r.extraction_count, 1),
                        nr.source_chunks = COALESCE(nr.source_chunks, []) +
                            [x IN COALESCE(r.source_chunks, [])
                             WHERE NOT x IN COALESCE(nr.source_chunks, [])]
                    DELETE r
                    RETURN count(*) AS c
                    """,
                    {"batch": batch_size},
                )
                c = result[0]["c"] if result else 0
                if c < batch_size:
                    break
            logger.info("canonicalize_relations: %s -> %s (sim=%.3f, %d edges)",
                        stray, best, score, n_edges)
        except Exception as e:
            logger.warning("canonicalize_relations failed for %s: %s", stray, e)
    return {"mapping": mapping, "unmatched": unmatched, "applied": not dry_run}


# ── 4. 照応（アナフォラ）ノードの解決とフラグ付け ─────────────────────

# 指示対象が文脈依存の照応語・汎用参照語パターン
_ANAPHORA_PATTERNS = [
    r"^(本|当|弊|同)[\w一-龯ぁ-んァ-ヶー]{1,8}$",   # 本製品/当社/同調査/本ガイドライン
    r"^(その他|上記|下記|前述|後述|以下)",
    r"^(これ|それ|あれ)(ら)?$",
]


def resolve_anaphora_nodes(graph, alias_maps: dict) -> dict:
    """照応ノードを2段構えで処理する。

    1. 解決: ノードidが略称定義（references.build_inventory の alias_maps）に
       あり、言及元文書すべてで同一の正式名称に解決できる場合
       → canonical_form / search_keys に正式名称を追加（正式名称での検索から到達可能に）
       → 正式名称のエンティティノードが存在すれば ALIAS_OF エッジも張る
    2. フラグ: 解決できない照応パターンのノードに is_anaphor=true
       → entity_node_predicate で検索・enrichment・entity vector から除外
       （「本製品」が複数マニュアルの別製品を1ノードに偽統合してdeg=114の
       ハブになっていた実測への対処）

    Args:
        alias_maps: {source: {略称: 正式名称}} (references.build_reference_graph の戻り値)

    Returns:
        {"resolved": 解決数, "flagged": フラグ数}
    """
    from graphrag_core.text.japanese import normalize_entity_text

    _pred = entity_node_predicate("n")
    anaphora_rx = [re.compile(p) for p in _ANAPHORA_PATTERNS]

    # 全略称の集合（どの文書かを問わず候補として引く）
    all_aliases = set()
    for amap in (alias_maps or {}).values():
        all_aliases.update(amap.keys())

    rows = graph.query(
        f"MATCH (n) WHERE {_pred} RETURN n.id AS id"
    )
    resolved = flagged = 0
    for r in rows:
        nid = r["id"]
        is_anaphora = any(rx.match(nid) for rx in anaphora_rx)
        is_alias = nid in all_aliases
        if not (is_anaphora or is_alias):
            continue

        # 言及元文書を取得し、文書スコープで解決を試みる
        srcs = graph.query(
            "MATCH (n {id: $id})<-[:" + chunk_edge() + "]-(d:" + chunk_label() + ") "
            "RETURN DISTINCT d.source AS s",
            {"id": nid},
        )
        sources = [x["s"] for x in srcs if x["s"]]
        formals = set()
        unresolved_src = False
        for s in sources:
            f = (alias_maps or {}).get(s, {}).get(nid)
            if f:
                formals.add(f)
            else:
                unresolved_src = True

        if len(formals) == 1 and not unresolved_src and sources:
            formal = next(iter(formals))
            norm_formal = normalize_entity_text(formal)
            graph.query(
                """
                MATCH (n {id: $id})
                SET n.canonical_form = $formal,
                    n.search_keys = COALESCE(n.search_keys, []) +
                        CASE WHEN $norm IN COALESCE(n.search_keys, [])
                             THEN [] ELSE [$norm] END
                """,
                {"id": nid, "formal": formal, "norm": norm_formal},
            )
            # 正式名称ノードが存在すれば ALIAS_OF を張る
            graph.query(
                """
                MATCH (a {id: $alias})
                MATCH (f) WHERE f.id = $formal OR f.norm_id = $norm
                WITH a, f LIMIT 1
                MERGE (a)-[:ALIAS_OF]->(f)
                """,
                {"alias": nid, "formal": formal, "norm": norm_formal},
            )
            resolved += 1
        elif is_anaphora:
            graph.query(
                "MATCH (n {id: $id}) SET n.is_anaphor = true", {"id": nid}
            )
            flagged += 1

    logger.info("resolve_anaphora_nodes: resolved=%d flagged=%d", resolved, flagged)
    return {"resolved": resolved, "flagged": flagged}


# ── 一括実行 ──────────────────────────────────────────────────────────

def consolidate_post_build(graph) -> dict:
    """ビルド後のKG統合処理をまとめて実行する。

    実行後に enrichment.enrich_post_build() を呼び直すこと。
    関係の埋め込み正規化は外部スキーマ使用時のみ動く（内部でembeddings生成、
    失敗しても他の統合処理は完了させる）。
    """
    result = {
        "value_nodes_flagged": flag_value_nodes(graph),
        "duplicate_merge": merge_duplicate_id_nodes(graph),
        "kana_variant_merge": merge_kana_variant_nodes(graph),
        "relation_normalize": normalize_relations(graph),
    }
    try:
        from graphrag_core.llm.factory import create_chat_llm
        llm = create_chat_llm(temperature=0, timeout=60, max_retries=1)
        result["relation_canonicalize"] = canonicalize_relations_to_schema(graph, llm=llm)
    except Exception as e:
        logger.warning("canonicalize_relations_to_schema skipped: %s", e)
        result["relation_canonicalize"] = {"error": str(e)}
    return result
