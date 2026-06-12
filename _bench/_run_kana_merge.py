"""実グラフに かな揺れマージ + enrich再実行 を適用"""
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
from graphrag_core.graph.consolidate import merge_kana_variant_nodes
from graphrag_core.graph.enrichment import enrich_post_build

g = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USER"),
               password=os.getenv("NEO4J_PW"), enhanced_schema=False)

t0 = time.time()
stats = merge_kana_variant_nodes(g)
print(f"kana merge: {stats} ({time.time()-t0:.0f}s)", flush=True)

t1 = time.time()
estats = enrich_post_build(g)
print(f"enrich: {estats} ({time.time()-t1:.0f}s)")

n = g.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
print(f"after: nodes={n}")
print("KANA MERGE COMPLETE")
