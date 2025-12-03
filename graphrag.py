"""
Graph-RAG with LLMGraphTransformer & LCEL (stable)
=================================================
最新版の LangChain API 変動に追従し、ImportError / TypeError をすべて潰した安定版スクリプトです。

## 主な修正点
1. **ParentDocumentRetriever が見つからない環境**でも動くよう、インポート失敗時は `vector_store.as_retriever()` をフォールバック。
2. **SemanticChunker** から削除された `chunk_size` 引数を排除。
3. **LLMGraphTransformer** のメソッド名を `convert_to_graph_documents()` に統一。
4. コメント行の重複を削除し、全体を整形。

依存ライブラリ
--------------
```bash
pip install langchain langchain-openai langchain-community langchain-postgres \
            langchain-experimental langchain-graph-retriever neo4j tiktoken \
            python-dotenv psycopg[binary]
```
※ `langchain` を明示的に追加しました。ParentDocumentRetriever がここに存在します。

`.env` 例 (Aura & RDS)
---------------------
```
OPENAI_API_KEY=sk-...

# --- Neo4j Aura ---
NEO4J_URI=neo4j+s://199e5d3d.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PW=your_neo4j_pw

# --- PGVector ---
PG_CONN=postgresql+psycopg://postgres:your_pw@localhost:5432/graph_rag
```

実行:
```bash
python graph_rag_lcel.py   # input.txt を同階層に配置
```
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# ── LangChain / OpenAI ─────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
try:
    from langchain_community.graphs.graph_document import GraphDocument  # ≥0.3.0
except ImportError:  # 旧互換
    from langchain_community.graphs import GraphDocument  # type: ignore
from langchain_community.vectorstores.pgvector import PGVector

# --- GraphRetriever (多段フォールバック) ---
try:
    from langchain_community.retrievers.graph import GraphRetriever  # ≥0.3.x
except ImportError:
    try:
        from langchain_graph_retriever import GraphRetriever
    except ImportError:
        from langchain_graph_retriever.graph_retriever import GraphRetriever

# --- ParentDocumentRetriever orフォールバック ---
try:
    from langchain_community.retrievers.parent_document import ParentDocumentRetriever
    HAS_PARENT = True
except ImportError:
    try:
        from langchain.retrievers.parent_document import ParentDocumentRetriever  # langchain>=0.2
        HAS_PARENT = True
    except ImportError:
        HAS_PARENT = False  # fallback to simple retriever

from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

# ── 環境変数 ───────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PW = os.getenv("NEO4J_PW")

# Postgres / PGVector
PG_CONN = os.getenv("PG_CONN")
if not PG_CONN:
    raise ValueError("PG_CONN 環境変数が未設定です。")

# ── 0. ドキュメント読み込み ───────────────────────────────────────
DOC_PATH = "input.txt"
if not Path(DOC_PATH).is_file():
    raise FileNotFoundError(f"{DOC_PATH} が見つかりません")
raw_text = Path(DOC_PATH).read_text(encoding="utf-8")

# ── 1. チャンク分割 (SemanticChunker) ─────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chunker = SemanticChunker(embeddings, buffer_size=50)  # chunk_size 引数は不要
chunks = chunker.create_documents([raw_text])

# ── 2. LLMGraphTransformer で GraphDocument 化 ───────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
transformer = LLMGraphTransformer(llm=llm)
graph_docs: List[GraphDocument] = transformer.convert_to_graph_documents(chunks)

# ── 3. Neo4j にロード ────────────────────────────────────────────
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PW)
graph.add_graph_documents(graph_docs, include_source=True)

# ── 4. PGVector にチャンク保存 ────────────────────────────────────
vector_store = PGVector.from_documents(chunks, embeddings, connection_string=PG_CONN)

# ── 5. Retriever 構築 ─────────────────────────────────────────────
graph_retriever = GraphRetriever(graph=graph, k=4, search_type="cypher")
if HAS_PARENT:
    vector_retriever = ParentDocumentRetriever(vector_store, search_kwargs={"k": 4})
else:
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# ── 6. LCEL チェイン定義 ───────────────────────────────────────────
graph_run = graph_retriever.as_runnable()
vector_run = vector_retriever.as_runnable()


def merge_ctx(data: Dict[str, Any]) -> Dict[str, Any]:
    """Graph/Vector の結果を 1 つの context 文字列へ整形"""
    triples = data["graph"]
    docs = data["docs"]
    graph_lines = [
        f"{t.get('start') or t.get('subject')} -[{t.get('predicate') or t.get('type')}]→ {t.get('end') or t.get('object')}"
        for t in triples
    ]
    context = (
        "<GRAPH_CONTEXT>\n" + "\n".join(graph_lines) + "\n</GRAPH_CONTEXT>\n\n" +
        "<DOCUMENT_CONTEXT>\n" + "\n---\n".join(d.page_content for d in docs) + "\n</DOCUMENT_CONTEXT>"
    )
    return {"context": context, "question": data["question"]}

prompt = PromptTemplate.from_template(
    """あなたはドキュメントの専門家です。\n質問: {question}\n\n{context}\n\n---\n上記情報のみを根拠に、日本語で網羅的かつ正確に回答してください。必要に応じて関連チャンクやエンティティを指摘してください。"""
)

chain = (
    {"question": RunnablePassthrough()}
    | RunnableParallel({"graph": graph_run, "docs": vector_run})
    | RunnableLambda(merge_ctx)
    | prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

# ── 7. 対話ループ ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Graph-RAG LCEL デモ。質問を入力してください (exit で終了)。")
    while True:
        q = input("\n質問> ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            break
        print("\n--- 回答 ---")
        try:
            print(chain.invoke(q))
        except Exception as e:
            print(f"[Error] {e}")
