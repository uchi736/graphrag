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
import sys
from pathlib import Path
from typing import List, Dict, Any
import hashlib

from dotenv import load_dotenv

# ── LangChain / OpenAI ─────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
try:
    # langchain>=0.2
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import (
        RunnableParallel,
        RunnablePassthrough,
        RunnableLambda,
    )
except ImportError:  # legacy (<0.2)
    from langchain.schema.output_parser import StrOutputParser  # type: ignore
    from langchain.schema.runnable import (  # type: ignore
        RunnableParallel,
        RunnablePassthrough,
        RunnableLambda,
    )

# ── 環境変数 ───────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Graph Backend選択
GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "networkx").lower()  # デフォルト: networkx

# Neo4j (GRAPH_BACKEND=neo4j の場合のみ必要)
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PW = os.getenv("NEO4J_PW")

# Postgres / PGVector
PG_CONN = os.getenv("PG_CONN")
if not PG_CONN:
    raise ValueError("PG_CONN 環境変数が未設定です。")

# バックエンド検証
if GRAPH_BACKEND not in ["neo4j", "networkx"]:
    raise ValueError(f"GRAPH_BACKEND は 'neo4j' または 'networkx' を指定してください。現在: {GRAPH_BACKEND}")

if GRAPH_BACKEND == "neo4j" and not all([NEO4J_URI, NEO4J_USER, NEO4J_PW]):
    raise ValueError("GRAPH_BACKEND=neo4j の場合、NEO4J_URI, NEO4J_USER, NEO4J_PW が必要です。")

# ── 0. ドキュメント読み込み ───────────────────────────────────────
DOC_PATH = "input.txt"
if not Path(DOC_PATH).is_file():
    raise FileNotFoundError(f"{DOC_PATH} が見つかりません")
raw_text = Path(DOC_PATH).read_text(encoding="utf-8")

# ── 1. チャンク分割 (RecursiveCharacterTextSplitter) ─────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chunker = RecursiveCharacterTextSplitter(
    chunk_size=500,           # 500文字ごとに分割
    chunk_overlap=100,        # 100文字オーバーラップ（文脈保持）
    separators=["\n\n", "\n", "。", "、", " ", ""],  # 日本語対応
    length_function=len
)
chunks = chunker.create_documents([raw_text])

# 重複チャンクを内容ハッシュで除去し、ハッシュをIDとして付与
deduped = []
seen_hashes = set()
for chunk in chunks:
    digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
    if digest in seen_hashes:
        continue
    seen_hashes.add(digest)
    chunk.metadata["id"] = digest
    deduped.append(chunk)
chunks = deduped


# --- ベクトルDBクリア機能（UI/CLI/ENVで利用） ---
class _DummyEmbeddings:
    """OpenAIを呼ばずにコレクション削除だけ行うためのダミー埋め込み"""

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def embed_query(self, _: str):
        return [0.0] * self.dim

    def embed_documents(self, texts):
        return [[0.0] * self.dim for _ in texts]


def clear_vector_store() -> None:
    """PGVector の既存コレクションを削除してクリーンにする"""
    store = PGVector(
        connection_string=PG_CONN,
        embedding_function=_DummyEmbeddings(),
        collection_name="graphrag",
        embedding_length=1536,
    )
    store.delete_collection()
    print("✅ Vector store collection 'graphrag' を削除しました")


def maybe_handle_control_mode() -> None:
    """UIメニュー/CLI引数/環境変数でクリア指示があれば即実行して終了"""
    args = [a.lower() for a in sys.argv[1:]]
    env_clear = os.getenv("CLEAR_VECTOR_STORE") == "1"
    use_ui = os.getenv("GRAPHRAG_UI") == "1" or any(a in {"ui", "menu", "manage"} for a in args)
    wants_clear = env_clear or any(a.strip("-") in {"clear", "reset", "cleanup"} for a in args)

    if use_ui:
        print("[GraphRAG 管理メニュー]")
        print("  1) Vector store を削除する")
        print("  2) 通常実行する")
        print("  3) 何もしないで終了")
        choice = input("選択番号を入力してください [1/2/3]: ").strip()
        if choice == "1":
            clear_vector_store()
            sys.exit(0)
        if choice == "3":
            sys.exit(0)
        # choice == "2" はそのまま続行

    if wants_clear:
        clear_vector_store()
        sys.exit(0)


maybe_handle_control_mode()

# ── 2. LLMGraphTransformer で GraphDocument 化 ───────────────────
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
transformer = LLMGraphTransformer(llm=llm)
graph_docs: List[GraphDocument] = transformer.convert_to_graph_documents(chunks)

# ── 3. グラフバックエンドにロード ────────────────────────────────────────────
if GRAPH_BACKEND == "neo4j":
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PW)
    graph.add_graph_documents(graph_docs, include_source=True)
    print(f"✅ Neo4jにグラフをロードしました (URI: {NEO4J_URI})")
else:  # networkx
    from networkx_graph import NetworkXGraph
    graph = NetworkXGraph(storage_path="graph.pkl", auto_save=True)
    graph.add_graph_documents(graph_docs, include_source=True)
    print(f"✅ NetworkXにグラフをロードしました (保存先: graph.pkl)")

# ── 4. PGVector にチャンク保存 ───────────────────────────────────────────
vector_store = PGVector.from_documents(
    chunks,
    embeddings,
    connection_string=PG_CONN,
    collection_name="graphrag",
    pre_delete_collection=True,  # 再実行時に既存コレクションを削除して重複を防止
    ids=[c.metadata["id"] for c in chunks],  # 同一IDの再登録を防ぐ
)

# ── 5. Retriever 構築 ─────────────────────────────────────────────
if GRAPH_BACKEND == "neo4j":
    graph_retriever = GraphRetriever(graph=graph, k=4, search_type="cypher")
else:  # networkx
    from networkx_graph import NetworkXGraphRetriever
    graph_retriever = NetworkXGraphRetriever(graph=graph, k=15, llm=llm)

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
    | ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    | StrOutputParser()
)

# ── 7. 対話ループ ────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Graph-RAG LCEL デモ (Backend: {GRAPH_BACKEND.upper()})")
    print("質問を入力してください (exit で終了)。")
    while True:
        q = input("\n質問> ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            break
        print("\n--- 回答 ---")
        try:
            print(chain.invoke(q))
        except Exception as e:
            print(f"[Error] {e}")
