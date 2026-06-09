#!/usr/bin/env python
"""Fujitsu KG Entity Linking (3段階: 正規化 → fuzzy → embedding+LLM)

Stage A: NFKC + neologdn + 空白除去で正規化 → 完全一致グルーピング
         (全角半角、合字、組版スペース等を一掃)
Stage B: rapidfuzz Levenshtein で edit_distance<=2 を type内で fuzzy一致
         (typo, 些細な表記差)
Stage C: ruri embedding 類似度
         - >=0.95: 自動マージ
         - 0.85-0.95: gemma4 LLM 判定 (隣接triples + 文脈つき)
Stage D: Union-Find クラスタリング → canonical 選定 → APOC で Neo4j マージ

Usage:
    python _bench/fujitsu_entity_linking.py                       # 実行 (Neo4j書込み)
    python _bench/fujitsu_entity_linking.py --dry-run             # 判定結果のみ出力
    python _bench/fujitsu_entity_linking.py --high 0.95 --low 0.85
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))


# ─── 正規化 ────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """Stage A 正規化: NFKC + neologdn + 空白除去"""
    import unicodedata
    s = unicodedata.normalize("NFKC", name or "")
    try:
        import neologdn
        s = neologdn.normalize(s)
    except ImportError:
        pass
    return re.sub(r"\s+", "", s).strip()


# 数値/パーセント/年月日: マージ禁止 (2.3% vs 25.5%が同じになるのを防ぐ)
_NUMERIC_PAT = re.compile(r"^[\d,.\-+%]+$")
_DATE_PAT = re.compile(r"^\d{1,4}[年/月日.\-]\d{1,2}")
_PERCENT_PAT = re.compile(r"^\d+(?:[\.,]\d+)?[%％]")


def is_numeric_like(s: str) -> bool:
    """数値/日付/パーセント風の名前は EL の対象から外す"""
    s = s.strip()
    if not s: return False
    if _NUMERIC_PAT.match(s): return True
    if _DATE_PAT.match(s): return True
    if _PERCENT_PAT.match(s): return True
    # 「100億円」「2024年度」みたいに数値が主成分のもの
    digits = sum(1 for c in s if c.isdigit())
    return digits * 2 > len(s)


def is_substring_pair(a: str, b: str) -> bool:
    """片方が他方の真部分文字列なら別概念扱い (リユース部品 ≠ リユース部品供給団体)

    diff>1 で除外: 8GB↔8GB×2 / USB↔USB機器 / CCP↔CCP番号 等の short-suffix 救済
    """
    if a == b:
        return False
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return short in long_ and len(short) >= 2 and len(long_) > len(short) + 1


# ─── Term ノード取得 ────────────────────────────────────────────────────

ENTITY_LABELS = ["Person", "Organization", "Product", "Process",
                 "Standard", "Indicator", "Concept", "Document"]


def fetch_terms(graph, limit: int = None) -> list[dict]:
    """entity ノード (8 labels) を取得 + normalized field 追加

    limit指定時は pagerank 順で上位だけ (低使用Termは EL対象外)
    """
    import sys
    print(f"  Neo4j 取得中 (limit={limit})...", flush=True); sys.stdout.flush()
    q = f"""
        MATCH (n)
        WHERE ANY(l IN labels(n) WHERE l IN {ENTITY_LABELS})
          AND NOT 'Document' IN labels(n)
          AND NOT 'ProcessedChunk' IN labels(n)
          AND n.id IS NOT NULL
        RETURN n.id AS id,
               labels(n) AS labels,
               COALESCE(n.mention_count, 0) AS mc,
               COALESCE(n.pagerank, 0.0) AS pr
        ORDER BY n.pagerank DESC
    """
    if limit:
        q += f" LIMIT {limit}"
    rows = graph.query(q)
    print(f"  fetched {len(rows)} terms", flush=True); sys.stdout.flush()
    for r in rows:
        r["normalized"] = normalize_name(r["id"])
    return rows


# ─── Stage A: 正規化ベース クラスタリング ─────────────────────────────

def normalize_clusters(terms: list[dict]) -> tuple[list[set], dict]:
    """同一の (normalized name, label) を持つ Term をクラスタリング

    type-strict: 同じ name でも label違いは別物扱い (Concept「部品」と Product「部品」は別)
    Returns:
        clusters: list of sets of indices (各クラスタは2件以上)
        norm_to_indices: dict[(normalized_str, label) -> list[idx]]
    """
    # key = (normalized, label) のタプル
    norm_to_indices: dict = {}
    for i, t in enumerate(terms):
        label = t["labels"][0] if t["labels"] else "?"
        key = (t["normalized"], label)
        norm_to_indices.setdefault(key, []).append(i)

    clusters = [set(idxs) for idxs in norm_to_indices.values() if len(idxs) > 1]
    return clusters, norm_to_indices


# ─── Stage B: Fuzzy (Levenshtein) マッチ ───────────────────────────────

def fuzzy_pairs(terms: list[dict], norm_to_indices: dict,
                max_distance: int = 2, max_len_diff_ratio: float = 0.2) -> list[tuple]:
    """同一 type内で Levenshtein 距離が近いペアを抽出

    type違いは embedding 段階に任せる (Concept vs Product の同字面など)
    """
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        print("  rapidfuzz 未インストール → Stage B スキップ")
        return []

    # type別に unique normalized name の代表 idx をまとめる
    by_type: dict[str, list[int]] = {}
    for norm, idxs in norm_to_indices.items():
        # 各 norm からは代表1つ (Stage A で同norm はクラスタ済)
        rep = idxs[0]
        label = terms[rep]["labels"][0] if terms[rep]["labels"] else "?"
        by_type.setdefault(label, []).append(rep)

    pairs = []
    for label, idxs in by_type.items():
        for ii, i in enumerate(idxs):
            ni = terms[i]["normalized"]
            min_len = len(ni)
            if min_len < 4:
                continue   # 短すぎる文字列はLevenshtein意味なし (Stage C に任せる)
            # 数値風はスキップ (10.0%↔10.8%, 2055↔2005 等の混同防止)
            if is_numeric_like(terms[i]["id"]):
                continue
            for j in idxs[ii + 1:]:
                nj = terms[j]["normalized"]
                if len(nj) < 4:
                    continue
                if is_numeric_like(terms[j]["id"]):
                    continue
                # 真部分文字列ペアもスキップ
                if is_substring_pair(terms[i]["id"], terms[j]["id"]):
                    continue
                max_len = max(len(ni), len(nj))
                if abs(len(ni) - len(nj)) > max_distance:
                    continue
                # 編集距離が文字列長の 20% を超えるなら別物扱い
                # (例: 4字 → 距離1まで, 6字 → 1-2まで)
                allowed = min(max_distance, max(1, int(max_len * 0.20)))
                d = Levenshtein.distance(ni, nj)
                if d <= allowed:
                    pairs.append((i, j, 1.0 - d / max_len))
    return pairs


# ─── Stage C: Embedding 類似度 ─────────────────────────────────────────


# ─── Embedding 計算 ────────────────────────────────────────────────────

def embed_terms(terms: list[dict], embeddings, batch_size: int = 64,
                use_normalized: bool = True) -> "np.ndarray":
    """ruri で Term name を embed

    use_normalized=True なら正規化後の name を使う (Stage A/B で漏れたケース用)
    """
    import numpy as np
    texts = [t["normalized"] if use_normalized else t["id"] for t in terms]
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vs = embeddings.embed_documents(batch)
        vecs.extend(vs)
        if (i // batch_size) % 20 == 0:
            print(f"  embed {i+len(batch)}/{len(texts)}")
    arr = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# ─── kNN で候補ペア抽出 ────────────────────────────────────────────────

def find_candidate_pairs(vecs: "np.ndarray", terms: list[dict],
                         low_th: float, high_th: float, top_k: int = 20):
    """各Termに対する top-K 類似Termを取り、閾値で分類

    精緻化:
    - 数値風の name (2.3%, 100億円, 2024年度) は両方マージ対象外
    - 包含関係 (A ⊂ B with len diff > 2) は別概念扱い → 除外
    - type違い (label違い) は auto_merge せず LLM judge へ

    Returns:
        auto_merge: list of (i, j, sim) where sim >= high_th AND same type AND not substring
        llm_judge:  list of (i, j, sim) where low_th <= sim
    """
    import numpy as np
    n = len(terms)
    auto_merge = []
    llm_judge = []

    block = 1000
    seen = set()
    skipped_numeric = 0
    skipped_substr = 0
    type_diff_to_llm = 0
    for start in range(0, n, block):
        end = min(start + block, n)
        sims = vecs[start:end] @ vecs.T
        for li, gi in enumerate(range(start, end)):
            sims[li, gi] = -1.0
        if top_k < n:
            topk_idx = np.argpartition(-sims, top_k, axis=1)[:, :top_k]
        else:
            topk_idx = np.tile(np.arange(n), (end-start, 1))
        for li, gi in enumerate(range(start, end)):
            for j_idx in topk_idx[li]:
                j = int(j_idx)
                if j == gi: continue
                key = (min(gi, j), max(gi, j))
                if key in seen: continue
                seen.add(key)
                sim = float(sims[li, j])
                if sim < low_th: continue

                name_i = terms[key[0]]["id"]; name_j = terms[key[1]]["id"]
                label_i = terms[key[0]]["labels"][0] if terms[key[0]]["labels"] else "?"
                label_j = terms[key[1]]["labels"][0] if terms[key[1]]["labels"] else "?"

                # 数値風はスキップ
                if is_numeric_like(name_i) or is_numeric_like(name_j):
                    skipped_numeric += 1; continue
                # 真部分文字列ペアもスキップ (別概念の可能性高い)
                if is_substring_pair(name_i, name_j):
                    skipped_substr += 1; continue

                if sim >= high_th and label_i == label_j:
                    auto_merge.append((key[0], key[1], sim))
                elif sim >= low_th:
                    # type違いは閾値 >= high でも LLM judge へ送る
                    if sim >= high_th and label_i != label_j:
                        type_diff_to_llm += 1
                    llm_judge.append((key[0], key[1], sim))
    print(f"  filter: 数値除外={skipped_numeric}, 真部分文字列={skipped_substr}, type違い→LLM={type_diff_to_llm}")
    return auto_merge, llm_judge


# ─── 文脈取得 (LLM判定用) ──────────────────────────────────────────────

def fetch_context(graph, term_id: str, max_triples: int = 5,
                  max_chunk_chars: int = 200) -> dict:
    """Term の隣接triples と source chunk 1つを取得"""
    # 隣接triples (in/out 両方)
    rows = graph.query("""
        MATCH (n {id: $tid})-[r]-(o)
        WHERE NOT 'Document' IN labels(o)
          AND NOT 'ProcessedChunk' IN labels(o)
          AND type(r) <> 'MENTIONS'
        RETURN type(r) AS t, o.id AS oid, startNode(r).id AS sid
        LIMIT $lim
    """, params={"tid": term_id, "lim": max_triples})
    triples = []
    for r in rows:
        if r["sid"] == term_id:
            triples.append(f"-[{r['t']}]-> {r['oid']}")
        else:
            triples.append(f"<-[{r['t']}]- {r['oid']}")
    # source chunk 1つ
    snippet = ""
    try:
        rows = graph.query("""
            MATCH (n {id: $tid})<-[:MENTIONS]-(d:Document)
            RETURN substring(d.text, 0, $maxc) AS txt LIMIT 1
        """, params={"tid": term_id, "maxc": max_chunk_chars})
        if rows:
            snippet = rows[0]["txt"] or ""
    except Exception:
        pass
    return {"triples": triples, "snippet": snippet}


# ─── LLM 判定 ──────────────────────────────────────────────────────────

JUDGE_PROMPT = """次の2つのエンティティは「同一の実体」を指しているかを判定してください。
**判定が曖昧な場合は NO を選んでください**。over-merge は KG を破壊します。

Entity A
  名前: {a_name}
  type: {a_type}
  隣接triples: {a_triples}
  出現文脈: {a_snippet}

Entity B
  名前: {b_name}
  type: {b_type}
  隣接triples: {b_triples}
  出現文脈: {b_snippet}

YES とすべきケース (慎重に):
- 表記揺れ・略称・正式名で **完全に同じ実体** を指す (例: 労基法=労働基準法、富士通=富士通株式会社、PC=パソコン)
- type違いでも実体が同じ (例: Concept「部品」と Product「部品」が同じ物理的部品)

NO とすべきケース (積極的に NO):
- **役割・職種違い**: オーナー≠マネージャー、設計者≠開発者
- **容量・サイズ・型番違い**: 8GB≠8GB×2、システムA≠システムB、原理原則[12]≠原理原則[14]
- **数値・ID違い**: 番号・コード・連番が異なる (CCP≠CCP番号、要件1≠要件2)
- **意味の包含**: 上位概念≠下位概念 (要求≠要求事項、要求≠要望、ディスプレイ≠外部ディスプレイ、USB≠USB機器)
- **集合の単位違い**: 「数」「等」「率」などの単位/接尾辞違い (就職者数≠就職者等、件数≠件)
- **generic 名 + 具体名**: 4文字以下の generic 名 (品/物/者/部) と具体名は別 (「品」≠「本製品」)
- **文脈で明確に別物**: 隣接triplesや snippet から指示対象が異なる

必ず以下のJSON 1行のみで返答:
{{"same": true|false, "reason": "30字以内"}}"""


def llm_judge_pair(a: dict, b: dict, ctx_a: dict, ctx_b: dict, llm) -> dict:
    """1ペアの判定"""
    prompt = JUDGE_PROMPT.format(
        a_name=a["id"], a_type=a["labels"][0],
        a_triples=", ".join(ctx_a["triples"][:4]) or "(なし)",
        a_snippet=(ctx_a["snippet"] or "(なし)")[:150],
        b_name=b["id"], b_type=b["labels"][0],
        b_triples=", ".join(ctx_b["triples"][:4]) or "(なし)",
        b_snippet=(ctx_b["snippet"] or "(なし)")[:150],
    )
    try:
        resp = llm.invoke(prompt)
        raw = resp.content.strip()
        m = re.search(r"\{[^{}]*\"same\"[^{}]*\}", raw, re.DOTALL)
        if not m:
            return {"same": None, "reason": f"parse fail: {raw[:80]}"}
        data = json.loads(m.group(0))
        return {"same": bool(data.get("same")), "reason": str(data.get("reason", "")).strip()[:60]}
    except Exception as e:
        return {"same": None, "reason": f"err: {e}"}


# ─── Union-Find ─────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


# ─── Canonical 選定 ─────────────────────────────────────────────────────

def choose_canonical(cluster: list[dict]) -> dict:
    """cluster (Term dict のリスト) から代表を選ぶ

    優先順位: pagerank > mention_count > 名前の長さ (短い=正規形に近い想定)
    """
    return max(cluster, key=lambda t: (
        t.get("pr", 0.0),
        t.get("mc", 0),
        -len(t["id"]),
    ))


# ─── Neo4j マージ実行 ────────────────────────────────────────────────────

def merge_cluster(graph, canonical_id: str, duplicate_ids: list[str]) -> tuple[int, int]:
    """APOC で代表ノードに統合。戻り値: (移動エッジ数, 削除ノード数)"""
    if not duplicate_ids:
        return (0, 0)
    try:
        # APOC で in/out エッジを canonical に移動 → duplicate を削除
        rows = graph.query("""
            MATCH (canon {id: $cid})
            UNWIND $dups AS dup_id
            MATCH (dup {id: dup_id})
            WHERE elementId(dup) <> elementId(canon)
            // outgoing edges
            CALL {
                WITH canon, dup
                MATCH (dup)-[r]->(other)
                WHERE elementId(other) <> elementId(canon)
                CALL apoc.create.relationship(canon, type(r), properties(r), other)
                  YIELD rel
                DELETE r
                RETURN COUNT(rel) AS o
            }
            // incoming edges
            CALL {
                WITH canon, dup
                MATCH (other)-[r]->(dup)
                WHERE elementId(other) <> elementId(canon)
                CALL apoc.create.relationship(other, type(r), properties(r), canon)
                  YIELD rel
                DELETE r
                RETURN COUNT(rel) AS i
            }
            // canonical との残自己ループ削除 + dup削除
            WITH canon, dup, o, i
            OPTIONAL MATCH (canon)-[selfr]-(canon) DELETE selfr
            DETACH DELETE dup
            RETURN o + i AS moved
        """, params={"cid": canonical_id, "dups": duplicate_ids})
        moved = sum(r["moved"] for r in rows) if rows else 0
        # aliases プロパティに duplicate 名を記録
        graph.query("""
            MATCH (canon {id: $cid})
            SET canon.aliases = COALESCE(canon.aliases, []) + $dups
        """, params={"cid": canonical_id, "dups": duplicate_ids})
        return (moved, len(duplicate_ids))
    except Exception as e:
        print(f"  ⚠️  merge failed for canonical={canonical_id}: {e}")
        return (0, 0)


# ─── メイン ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--high", type=float, default=0.95, help="自動マージ閾値")
    ap.add_argument("--low", type=float, default=0.85, help="LLM判定 lower閾値")
    ap.add_argument("--top-k", type=int, default=20, help="各Term の検索近傍数")
    ap.add_argument("--concurrency", type=int, default=8, help="LLM並列度")
    ap.add_argument("--dry-run", action="store_true", help="Neo4j書込みしない")
    ap.add_argument("--max-judge", type=int, default=None,
                    help="LLM判定の最大件数 (デバッグ用)")
    ap.add_argument("--limit", type=int, default=None,
                    help="pagerank上位N Termのみ対象 (32K全部だと重い、5000推奨)")
    ap.add_argument("--output", default="_bench/entity_linking_report.json")
    args = ap.parse_args()

    from graphrag_core.config import reset_settings, get_settings
    from graphrag_core.llm.factory import create_chat_llm, create_embeddings
    from langchain_neo4j import Neo4jGraph
    reset_settings()
    s = get_settings()
    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw)

    print("=== Stage 1: Term node 取得 ===", flush=True)
    t0 = time.time()
    terms = fetch_terms(graph, limit=args.limit)
    print(f"  total: {len(terms)} terms", flush=True)
    if not terms:
        print("Term node なし"); return

    print("\n=== Stage A: 正規化 (NFKC + neologdn + 空白除去) ===")
    norm_clusters, norm_to_indices = normalize_clusters(terms)
    norm_pairs = []
    for cluster in norm_clusters:
        sorted_idxs = sorted(cluster)
        for i in sorted_idxs:
            for j in sorted_idxs:
                if i < j:
                    norm_pairs.append((i, j, 1.0))
    print(f"  正規化で自動マージできるクラスタ: {len(norm_clusters)}")
    print(f"  ペア数: {len(norm_pairs)}")
    if norm_clusters:
        print("  例:")
        for cluster in list(norm_clusters)[:3]:
            names = [terms[i]['id'] for i in sorted(cluster)][:3]
            print(f"    [{len(cluster)}] {names}")

    print("\n=== Stage B: Fuzzy match (Levenshtein <= 2, type内) ===")
    t_b = time.time()
    fuzzy_pair_list = fuzzy_pairs(terms, norm_to_indices, max_distance=2)
    print(f"  fuzzy ペア: {len(fuzzy_pair_list)} ({time.time()-t_b:.1f}s)")
    if fuzzy_pair_list:
        print("  例:")
        for i, j, sim in fuzzy_pair_list[:5]:
            print(f"    {terms[i]['id']:25s} ⟷ {terms[j]['id']:25s}  ratio={sim:.2f}")

    print("\n=== Stage C: ruri embedding (正規化後の name で) ===")
    t1 = time.time()
    embeddings = create_embeddings()
    vecs = embed_terms(terms, embeddings, use_normalized=True)
    print(f"  embedded ({time.time()-t1:.1f}s)")

    print("\n=== Stage D: embedding候補ペア抽出 ===")
    t2 = time.time()
    auto_pairs, llm_pairs = find_candidate_pairs(
        vecs, terms, args.low, args.high, args.top_k,
    )
    # Stage A/B でカバー済みのペアは embedding 段階から除外
    already = set()
    for i, j, _ in norm_pairs:
        already.add((min(i, j), max(i, j)))
    for i, j, _ in fuzzy_pair_list:
        already.add((min(i, j), max(i, j)))
    auto_pairs = [(i, j, s) for i, j, s in auto_pairs if (min(i, j), max(i, j)) not in already]
    llm_pairs = [(i, j, s) for i, j, s in llm_pairs if (min(i, j), max(i, j)) not in already]
    print(f"  embedding自動マージ候補 (>={args.high}): {len(auto_pairs)}")
    print(f"  LLM判定候補 ({args.low}-{args.high}): {len(llm_pairs)} ({time.time()-t2:.1f}s)")
    if auto_pairs[:5]:
        print("  embedding-only 自動マージ例:")
        for i, j, sim in auto_pairs[:3]:
            print(f"    {terms[i]['id']:25s} ⟷ {terms[j]['id']:25s}  sim={sim:.3f}")

    if args.max_judge:
        llm_pairs = llm_pairs[: args.max_judge]
        print(f"  --max-judge で {len(llm_pairs)} に制限")

    print("\n=== Stage 4: LLM判定 ===")
    llm = create_chat_llm(temperature=0)
    judge_results = {}   # (i, j) -> {"same": bool, "reason": str}

    def _judge_one(pair):
        i, j, sim = pair
        ca = fetch_context(graph, terms[i]["id"])
        cb = fetch_context(graph, terms[j]["id"])
        verdict = llm_judge_pair(terms[i], terms[j], ca, cb, llm)
        return (i, j, sim, verdict)

    t3 = time.time()
    if llm_pairs:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(_judge_one, p) for p in llm_pairs]
            done = 0
            for fut in as_completed(futs):
                i, j, sim, verdict = fut.result()
                judge_results[(i, j)] = verdict
                done += 1
                if done % 50 == 0 or done == len(llm_pairs):
                    avg = (time.time() - t3) / done
                    eta = avg * (len(llm_pairs) - done) / 60
                    yes_n = sum(1 for v in judge_results.values() if v.get("same"))
                    print(f"  {done}/{len(llm_pairs)}  avg={avg:.1f}s  ETA={eta:.0f}min  same={yes_n}")
    print(f"  完了 ({time.time()-t3:.1f}s)")
    confirmed_yes = [(i, j) for (i, j), v in judge_results.items() if v.get("same") is True]
    print(f"  LLM-confirmed同一: {len(confirmed_yes)}")

    print("\n=== Stage E: 代表選定クラスタリング (no transitive chain) ===")
    # 各Termについて、自分と「直接的に同一判定された」相手集合を作る
    # その集合内で pagerank最高のTermを canonical に選ぶ
    # 鎖でつながらない (A↔B、B↔C → A↔C ではない)
    direct_links: dict[int, set] = {i: set() for i in range(len(terms))}
    for i, j, _ in norm_pairs:
        direct_links[i].add(j); direct_links[j].add(i)
    for i, j, _ in fuzzy_pair_list:
        direct_links[i].add(j); direct_links[j].add(i)
    for i, j, _ in auto_pairs:
        direct_links[i].add(j); direct_links[j].add(i)
    for i, j in confirmed_yes:
        direct_links[i].add(j); direct_links[j].add(i)

    # 各Termの canonical: 自分 + 直接リンク集合の中で最高pagerankのTerm
    canonical_of: dict[int, int] = {}
    for i, t in enumerate(terms):
        candidates = list(direct_links[i]) + [i]
        canonical_of[i] = max(candidates, key=lambda x: (
            terms[x].get("pr", 0.0),
            terms[x].get("mc", 0),
            -len(terms[x]["id"]),
        ))

    # canonicalの transitive closure を最大1回だけ追う (代表の代表が違うケースを救済)
    for i in list(canonical_of):
        c = canonical_of[i]
        if canonical_of[c] != c:
            canonical_of[i] = canonical_of[c]

    # 同じ canonical を持つ Term をクラスタ化
    clusters: dict[int, list[int]] = {}
    for idx in range(len(terms)):
        clusters.setdefault(canonical_of[idx], []).append(idx)
    multi_clusters = {r: m for r, m in clusters.items() if len(m) > 1}
    # 大きすぎるクラスタは over-merging の疑いがあるので drop (前回10→今回5に厳格化)
    MAX_CLUSTER_SIZE = 5
    dropped = {r: m for r, m in multi_clusters.items() if len(m) > MAX_CLUSTER_SIZE}
    multi_clusters = {r: m for r, m in multi_clusters.items() if len(m) <= MAX_CLUSTER_SIZE}
    print(f"  クラスタ: {len(multi_clusters)} (合計Term: {sum(len(m) for m in multi_clusters.values())})")
    if dropped:
        print(f"  ⚠️ {MAX_CLUSTER_SIZE}超のクラスタを安全のためdrop: {len(dropped)} (合計Term: {sum(len(m) for m in dropped.values())})")
        for r, m in list(dropped.items())[:5]:
            print(f"    [{len(m)}] {terms[r]['id']} ← {[terms[i]['id'] for i in m[:5]]}...")

    print("\n=== Stage 6: canonical 選定 ===")
    merge_plan = []   # [(canonical_id, [dup_ids]), ...]
    for root, members in multi_clusters.items():
        cluster_terms = [terms[i] for i in members]
        canon = choose_canonical(cluster_terms)
        dups = [t["id"] for t in cluster_terms if t["id"] != canon["id"]]
        merge_plan.append({
            "canonical": canon["id"],
            "canonical_type": canon["labels"][0] if canon["labels"] else "",
            "canonical_pr": canon.get("pr", 0.0),
            "canonical_mc": canon.get("mc", 0),
            "duplicates": dups,
            "size": len(cluster_terms),
        })
    merge_plan.sort(key=lambda x: -x["size"])

    print(f"  クラスタ計画: {len(merge_plan)}")
    print("  Top 10 クラスタ:")
    for p in merge_plan[:10]:
        print(f"    [{p['size']}] {p['canonical']} ← {p['duplicates'][:3]}{'...' if len(p['duplicates'])>3 else ''}")

    # レポート出力
    report = {
        "config": {
            "high_threshold": args.high,
            "low_threshold": args.low,
            "top_k": args.top_k,
        },
        "stats": {
            "total_terms": len(terms),
            "stage_a_normalize_clusters": len(norm_clusters),
            "stage_a_pairs": len(norm_pairs),
            "stage_b_fuzzy_pairs": len(fuzzy_pair_list),
            "stage_c_auto_pairs": len(auto_pairs),
            "stage_c_llm_judged": len(judge_results),
            "stage_c_llm_yes": len(confirmed_yes),
            "final_clusters": len(merge_plan),
        },
        "merge_plan": merge_plan,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {args.output}")

    if args.dry_run:
        print("\n--dry-run のため Neo4j書込みスキップ")
        return

    print("\n=== Stage 7: Neo4jマージ実行 ===")
    total_moved, total_deleted = 0, 0
    t7 = time.time()
    for i, plan in enumerate(merge_plan):
        moved, deleted = merge_cluster(graph, plan["canonical"], plan["duplicates"])
        total_moved += moved
        total_deleted += deleted
        if (i + 1) % 100 == 0 or i + 1 == len(merge_plan):
            print(f"  {i+1}/{len(merge_plan)}  累計移動エッジ={total_moved}, 削除ノード={total_deleted}")
    print(f"\n完了 ({time.time()-t7:.1f}s)")
    print(f"  移動エッジ: {total_moved}")
    print(f"  削除ノード: {total_deleted}")

    # 最終チェック
    rows = graph.query("MATCH (n) WHERE NOT 'Document' IN labels(n) AND NOT 'ProcessedChunk' IN labels(n) RETURN COUNT(n) AS c")
    print(f"  残Term数: {rows[0]['c']}")


if __name__ == "__main__":
    main()
