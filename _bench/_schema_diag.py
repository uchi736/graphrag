"""現行KGのスキーマ実態診断（読み取りのみ）"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()

from langchain_neo4j import Neo4jGraph
g = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USER"),
               password=os.getenv("NEO4J_PW"), enhanced_schema=False)

ENT = ("n.id IS NOT NULL AND NOT n:ProcessedChunk AND NOT n:SchemaMeta "
       "AND NOT n.id =~ '[0-9a-f]{32,}'")

print("=== 1. 関係タイプ分布 (MENTIONS除く) ===")
rows = g.query(
    "MATCH ()-[r]->() WHERE type(r) <> 'MENTIONS' "
    "RETURN type(r) AS t, count(*) AS c ORDER BY c DESC"
)
total = sum(r["c"] for r in rows)
for r in rows:
    print(f"  {r['t']:22s} {r['c']:6d} ({100*r['c']/total:.1f}%)")
print(f"  TOTAL {total}")

print("\n=== 2. エンティティのラベル分布 (チャンク除外) ===")
rows = g.query(
    f"MATCH (n) WHERE {ENT} "
    "UNWIND labels(n) AS l RETURN l, count(*) AS c ORDER BY c DESC"
)
for r in rows:
    print(f"  {r['l']:15s} {r['c']:6d}")

print("\n=== 3. 数値/日付っぽいノード (value-as-node 問題) ===")
row = g.query(
    f"MATCH (n) WHERE {ENT} "
    "AND n.id =~ '^[0-9０-９,，.．%％~〜+±▲△一-九十百千万億兆年月日円人件台時分秒回歳割約-]+$' "
    "RETURN count(n) AS c"
)
print(f"  数値・日付・単位のみのノード: {row[0]['c']}")
rows = g.query(
    f"MATCH (n) WHERE {ENT} "
    "AND n.id =~ '^[0-9０-９,，.．%％~〜+±▲△一-九十百千万億兆年月日円人件台時分秒回歳割約-]+$' "
    "WITH n LIMIT 12 MATCH (a)-[r]->(n) WHERE type(r)<>'MENTIONS' "
    "RETURN a.id AS s, type(r) AS t, n.id AS o LIMIT 12"
)
for r in rows:
    print(f"    {r['s'][:30]} -[{r['t']}]-> {r['o'][:25]}")

print("\n=== 4. 同一idの複数ラベル分裂 ===")
row = g.query(
    f"MATCH (n) WHERE {ENT} "
    "WITH n.id AS id, count(*) AS c WHERE c > 1 "
    "RETURN count(*) AS dup_ids, sum(c) AS dup_nodes"
)
print(f"  分裂id数: {row[0]['dup_ids']}, 該当ノード数: {row[0]['dup_nodes']}")
rows = g.query(
    f"MATCH (n) WHERE {ENT} "
    "WITH n.id AS id, collect(DISTINCT labels(n)[0]) AS ls, count(*) AS c WHERE c > 1 "
    "RETURN id, ls, c ORDER BY c DESC LIMIT 8"
)
for r in rows:
    print(f"    {r['id'][:35]:35s} x{r['c']} {r['ls']}")

print("\n=== 5. ハブノード (次数上位, MENTIONS除く) ===")
rows = g.query(
    f"MATCH (n) WHERE {ENT} "
    "WITH n, COUNT { (n)-[r]-() WHERE type(r)<>'MENTIONS' } AS deg "
    "RETURN n.id AS id, labels(n)[0] AS l, deg ORDER BY deg DESC LIMIT 10"
)
for r in rows:
    print(f"    deg={r['deg']:5d} [{r['l']}] {r['id'][:40]}")

print("\n=== 6. 逆方向ペア関係の重複 (PART_OF vs HAS_PART 等) ===")
for a, b in [("PART_OF", "HAS_PART"), ("FOLLOWS", "PRECEDES")]:
    row = g.query(
        f"MATCH (x)-[r1:{a}]->(y) MATCH (y)-[r2:{b}]->(x) RETURN count(*) AS c"
    )
    print(f"  {a}/{b} 同一ペアで両方向: {row[0]['c']}")

print("\n=== 7. 孤立エンティティ (意味エッジ0本) ===")
row = g.query(
    f"MATCH (n) WHERE {ENT} "
    "AND NOT EXISTS { (n)-[r]-() WHERE type(r)<>'MENTIONS' } "
    "RETURN count(n) AS c"
)
print(f"  MENTIONS以外のエッジが無いノード: {row[0]['c']}")
