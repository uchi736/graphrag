"""実グラフに consolidate + enrich 再実行を適用する"""
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

from langchain_neo4j import Neo4jGraph
from graphrag_core.graph.consolidate import consolidate_post_build
from graphrag_core.graph.enrichment import enrich_post_build

g = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USER"),
               password=os.getenv("NEO4J_PW"), enhanced_schema=False)

t0 = time.time()
print("=== consolidate_post_build ===", flush=True)
stats = consolidate_post_build(g)
print(f"value_nodes_flagged: {stats['value_nodes_flagged']}")
print(f"duplicate_merge: {stats['duplicate_merge']}")
print(f"relation_normalize: {stats['relation_normalize']}")
print(f"consolidate done in {time.time()-t0:.0f}s", flush=True)

t1 = time.time()
print("\n=== enrich_post_build (再実行) ===", flush=True)
estats = enrich_post_build(g)
print(f"enrich: {estats}")
print(f"enrich done in {time.time()-t1:.0f}s")

# 統合後の規模
n = g.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
e = g.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
print(f"\nafter: nodes={n} edges={e}")
print("CONSOLIDATE COMPLETE")
