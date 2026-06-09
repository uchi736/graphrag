"""
Streamlit UI for Graph-RAG
===========================
シンプルなGraph-RAG用のStreamlitアプリケーション
- PDF/テキストファイルアップロード
- 質問入力とRAG実行
- ナレッジグラフの可視化

Thin orchestrator: 全ロジックは graphrag_core パッケージに委譲。
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure the script directory is in Python path for local module imports
_script_dir = Path(__file__).parent.parent.resolve()
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import streamlit as st
from dotenv import load_dotenv
import tempfile
from typing import List, Dict, Any
import hashlib
import json
import uuid

# ── graphrag_core imports ─────────────────────────────────────────
from graphrag_core.config import get_settings
from graphrag_core.prompts import (
    QA_PROMPT,
    KG_SYSTEM_PROMPT,
    KG_USER_PROMPT,
    NL_TO_CYPHER_PROMPT,
)
from graphrag_core.llm.factory import create_chat_llm, get_llm_provider_info
from graphrag_core.llm.langfuse_utils import (
    observe,
    propagate_attributes,
    get_langfuse_callback,
    get_langfuse_config,
    is_langfuse_enabled,
)
from graphrag_core.db.utils import (
    normalize_pg_connection_string,
    ensure_tokenized_schema,
    ensure_hnsw_index,
    ensure_embedding_id_unique,
    ensure_schema_compatibility,
    add_connection_timeout,
    retry_on_timeout,
    batch_pgvector_from_documents,
    batch_update_tokenized,
)
from graphrag_core.text.japanese import get_japanese_processor, SUDACHI_AVAILABLE
from graphrag_core.text.chunking import create_markdown_chunks
from graphrag_core.retrieval.entity_vector import EntityVectorizer
from graphrag_core.retrieval.pipeline import retriever_and_merge as qa_retriever_and_merge
from graphrag_core.document.pdf import load_pdf_text, extract_pdf_text
from graphrag_core.graph.crud import (
    graph_add_node,
    graph_update_node,
    graph_delete_node,
    graph_get_node_info,
    graph_add_edge,
    graph_update_edge,
    graph_delete_edge,
    graph_get_edge_info,
    graph_get_data_for_cache,
)
from graphrag_core.ui.css import CUSTOM_CSS
from graphrag_core.ui.visualization import (
    visualize_graph_neo4j_viz,
    get_node_type,
    get_color_for_type,
)
from graphrag_core.ui.data_tables import display_data_tables as _display_data_tables
from graphrag_core.ui.dialogs import (
    edit_node_dialog as _edit_node_dialog,
    edit_edge_dialog as _edit_edge_dialog,
    confirm_delete_dialog,
)
from graphrag_core.ui.sidebar import render_sidebar

# ── LangChain imports ─────────────────────────────────────────────
from graphrag_core.llm.factory import create_embeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument
from langchain_community.document_loaders import TextLoader
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

try:
    from langchain_community.retrievers.parent_document import ParentDocumentRetriever
    HAS_PARENT = True
except ImportError:
    try:
        from langchain.retrievers.parent_document import ParentDocumentRetriever
        HAS_PARENT = True
    except ImportError:
        HAS_PARENT = False

# 環境変数読み込み
load_dotenv()

# =====================================================================
# Streamlit設定
# =====================================================================
st.set_page_config(
    page_title="Graph-RAG Demo",
    page_icon="🔗",
    layout="wide"
)

# カスタムCSS適用
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =====================================================================
# セッションステート早期初期化
# =====================================================================
if "max_nodes" not in st.session_state:
    st.session_state.max_nodes = 200
if "langfuse_session_id" not in st.session_state:
    st.session_state.langfuse_session_id = str(uuid.uuid4())

# タイトル
st.title("🔗 Graph-RAG with NEO4J & PGVector")


# =====================================================================
# サイドバー描画 (graphrag_core.ui.sidebar)
# =====================================================================
sidebar_config = render_sidebar(settings={
    "get_llm_provider_info": get_llm_provider_info,
    "neo4j_graph_class": Neo4jGraph,
    "sudachi_available": SUDACHI_AVAILABLE,
    "normalize_pg_connection_string": normalize_pg_connection_string,
})

# サイドバーから返った設定をローカル変数に展開
show_graph = sidebar_config.get("show_graph", True)
max_nodes = sidebar_config.get("max_nodes", 200)

# 環境変数をローカル変数へ (複数箇所で使用)
_settings = get_settings()
NEO4J_URI = _settings.neo4j_uri
NEO4J_USER = _settings.neo4j_user
NEO4J_PW = _settings.neo4j_pw
PG_CONN = _settings.pg_conn
PG_COLLECTION = _settings.pg_collection

# PGVECTOR_CONNECTION_STRING を設定（langchain-postgres が参照）
if PG_CONN and not os.getenv("PGVECTOR_CONNECTION_STRING"):
    os.environ["PGVECTOR_CONNECTION_STRING"] = PG_CONN

# =====================================================================
# セッションステート初期化
# =====================================================================
if "chain" not in st.session_state:
    st.session_state.chain = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "existing_graph_loaded" not in st.session_state:
    st.session_state.existing_graph_loaded = False
if "graph_data_cache" not in st.session_state:
    st.session_state.graph_data_cache = None


# =====================================================================
# ダイアログ wrapper (CRUD をバインド)
# =====================================================================
def _wrapped_edit_node_dialog(graph, node_info):
    _edit_node_dialog(
        graph, node_info,
        graph_update_node=graph_update_node,
        graph_add_node=graph_add_node,
        graph_get_data_for_cache=graph_get_data_for_cache,
    )


def _wrapped_edit_edge_dialog(graph, edge_info, all_nodes=None):
    _edit_edge_dialog(
        graph, edge_info, all_nodes,
        graph_update_edge=graph_update_edge,
        graph_add_edge=graph_add_edge,
        graph_get_data_for_cache=graph_get_data_for_cache,
    )


# =====================================================================
# ヘルパー関数
# =====================================================================
def create_vector_retriever(vector_store, top_k: int):
    """バージョン差異を吸収してRetrieverを構築する。"""
    if vector_store is None:
        return None
    if HAS_PARENT:
        try:
            return ParentDocumentRetriever(
                vectorstore=vector_store,
                search_kwargs={"k": top_k},
            )
        except Exception as e:
            print(f"[Retriever] ParentDocumentRetriever unavailable. fallback=vector_store.as_retriever ({e})")
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def check_existing_graph(graph) -> dict:
    """Neo4jに既存のグラフデータがあるかチェック"""
    try:
        result = graph.query("MATCH (n) RETURN count(n) AS node_count")
        node_count = result[0]['node_count'] if result else 0
        if node_count > 0:
            result_rel = graph.query("MATCH ()-[r]->() RETURN count(r) AS rel_count")
            rel_count = result_rel[0]['rel_count'] if result_rel else 0
            return {'exists': True, 'node_count': node_count, 'rel_count': rel_count}
        return {'exists': False, 'node_count': 0, 'rel_count': 0}
    except Exception as e:
        st.error(f"グラフ接続エラー: {e}")
        return {'exists': False, 'node_count': 0, 'rel_count': 0}


def _build_config_from_session_state() -> dict:
    """st.session_state からqa_pipeline用のconfig dictを構築"""
    log = st.session_state.get("path_rerank_log", [])
    st.session_state["path_rerank_log"] = log
    return {
        "graph_hop_count": st.session_state.get("graph_hop_count", 2),
        "retrieval_top_k": st.session_state.get("retrieval_top_k", 5),
        "enable_japanese_search": st.session_state.get("enable_japanese_search", True),
        "enable_rerank": st.session_state.get("enable_rerank", True),
        "enable_entity_vector": st.session_state.get("enable_entity_vector", False),
        "entity_similarity_threshold": st.session_state.get("entity_similarity_threshold", 0.7),
        "search_mode": st.session_state.get("search_mode", "hybrid"),
        "include_kg_source_chunks": st.session_state.get("include_kg_source_chunks", True),
        "path_max_candidates": get_settings().path_max_candidates,
        "_path_rerank_log": log,
    }


def _create_embeddings():
    return create_embeddings()


def _create_graph_instance():
    """Neo4jグラフインスタンスを作成"""
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PW,
        enhanced_schema=False,
    )


# =====================================================================
# グラフデータ取得関数
# =====================================================================
def get_enhanced_graph_data(graph, limit=200):
    """グラフバックエンドから拡張グラフデータを取得"""
    if hasattr(graph, 'get_graph_data'):
        return graph.get_graph_data(limit=limit)

    # Neo4jの場合はCypherクエリ
    query = f"""
    MATCH (n)-[r]->(m)
    WHERE type(r) <> 'MENTIONS'
    AND NOT n.id =~ '[0-9a-f]{{32,}}'
    AND NOT m.id =~ '[0-9a-f]{{32,}}'
    OPTIONAL MATCH (n)<-[:MENTIONS]-(doc_n:Document)
    OPTIONAL MATCH (m)<-[:MENTIONS]-(doc_m:Document)
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


def get_enhanced_subgraph_data(graph, center_nodes: List[str], hop: int = 1, limit: int = 500):
    """指定ノード群を中心にN-hop範囲のエッジを取得"""
    hop = max(1, min(int(hop), 3))
    query = f"""
    MATCH (c) WHERE c.id IN $entities
    MATCH (c)-[*1..{hop}]-(n)
    MATCH (n)-[r]->(m)
    WHERE type(r) <> 'MENTIONS'
      AND NOT n.id =~ '[0-9a-f]{{32,}}' AND NOT m.id =~ '[0-9a-f]{{32,}}'
    RETURN DISTINCT
      n.id AS source,
      COALESCE(labels(n)[0], 'Unknown') AS source_type,
      type(r) AS relation,
      m.id AS target,
      COALESCE(labels(m)[0], 'Unknown') AS target_type
    LIMIT {limit}
    """
    results = graph.query(query, params={'entities': center_nodes})
    return [
        {
            'source': r.get('source', ''),
            'source_type': r.get('source_type', 'Unknown'),
            'target': r.get('target', ''),
            'target_type': r.get('target_type', 'Unknown'),
            'relation': r.get('relation', 'RELATED'),
            'edge_key': 0,
            'source_degree': 0,
            'target_degree': 0,
            'source_docs': [],
            'target_docs': [],
        }
        for r in (results or [])
    ]


# =====================================================================
# Cypher クエリ関数
# =====================================================================
@observe(name="cypher_generation")
def natural_language_to_cypher(query: str) -> str:
    """自然言語クエリをCypherクエリに変換"""
    try:
        llm = create_chat_llm(temperature=0)
        prompt = NL_TO_CYPHER_PROMPT.format(query=query)
        response = llm.invoke(prompt, config=get_langfuse_callback())
        cypher_query = response.content.strip()
        if cypher_query.startswith("```"):
            lines = cypher_query.split("\n")
            cypher_query = "\n".join(lines[1:-1]) if len(lines) > 2 else cypher_query
        return cypher_query
    except Exception as e:
        st.error(f"Cypherクエリ変換エラー: {e}")
        return ""


def execute_cypher_and_visualize(cypher_query: str, graph):
    """Cypherクエリを実行して結果を返す"""
    try:
        dangerous_keywords = ['DELETE', 'DROP', 'CREATE', 'MERGE', 'SET', 'REMOVE', 'DETACH']
        upper_query = cypher_query.upper()
        for keyword in dangerous_keywords:
            if keyword in upper_query:
                st.error(f"⚠️ 危険なクエリが検出されました: {keyword} は使用できません")
                return None
        result = graph.query(cypher_query)
        if not result:
            st.warning("クエリ結果が空です")
            return None
        return result
    except Exception as e:
        st.error(f"クエリ実行エラー: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return None


# =====================================================================
# ドキュメント読み込み
# =====================================================================
def load_documents(uploaded_files) -> list:
    """アップロードされたファイルからテキストを抽出（ソースメタデータ付き）"""
    from langchain_core.documents import Document
    all_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            file_name = uploaded_file.name
            if uploaded_file.name.endswith('.pdf'):
                _s = get_settings()
                text_content = load_pdf_text(tmp_path)
                if _s.pdf_processor == "onprem" and text_content:
                    st.info(f"📄 オンプレ前処理で解析: {file_name}")
            elif uploaded_file.name.endswith('.txt') or uploaded_file.name.endswith('.md'):
                loader = TextLoader(tmp_path, encoding='utf-8')
                docs = loader.load()
                text_content = "\n".join([doc.page_content for doc in docs])
            else:
                text_content = uploaded_file.getvalue().decode('utf-8')

            all_docs.append(Document(
                page_content=text_content,
                metadata={"source": file_name}
            ))
        finally:
            os.unlink(tmp_path)

    return all_docs


def load_csv_edges(uploaded_file):
    """CSV( source,target,label ) を読み込みシンプルなエッジリストを返す"""
    if not uploaded_file:
        return []
    import csv
    import io

    text = uploaded_file.getvalue().decode("utf-8-sig", errors="ignore")

    reader = csv.DictReader(io.StringIO(text))
    edges = []
    for row in reader:
        if not row:
            continue
        normalized_row = {k.strip().lower() if k else k: v for k, v in row.items()}
        src = (normalized_row.get("source") or normalized_row.get("from") or normalized_row.get("src") or "").strip()
        tgt = (normalized_row.get("target") or normalized_row.get("to") or normalized_row.get("dst") or "").strip()
        rel = (normalized_row.get("label") or normalized_row.get("relation") or normalized_row.get("rel") or "RELATED_TO").strip()
        if not src or not tgt:
            continue
        edges.append({"source": src, "target": tgt, "label": rel})
    return edges


# =====================================================================
# 既存グラフからシステムを復元
# =====================================================================
def restore_from_existing_graph():
    """グラフバックエンドとPGVectorから既存データを使ってシステムを復元"""
    try:
        graph = _create_graph_instance()

        embeddings = _create_embeddings()
        pg_conn_with_timeout = add_connection_timeout(PG_CONN, timeout=30)

        def create_vector_store():
            return PGVector(
                connection=pg_conn_with_timeout,
                embeddings=embeddings,
                collection_name=PG_COLLECTION
            )

        vector_store = retry_on_timeout(create_vector_store, max_retries=3, delay=2.0)

        retrieval_top_k = st.session_state.get('retrieval_top_k', 5)
        vector_retriever = create_vector_retriever(vector_store, retrieval_top_k)

        llm = create_chat_llm(temperature=0)
        config = _build_config_from_session_state()

        def retriever_and_merge(question: str):
            return qa_retriever_and_merge(
                question, graph, llm, embeddings, vector_retriever,
                PG_CONN, PG_COLLECTION, config
            )

        prompt = PromptTemplate.from_template(QA_PROMPT)

        llm_chain = (
            prompt
            | create_chat_llm(temperature=0)
            | StrOutputParser()
        )

        @observe(name="answer_generation")
        def generate_with_sources(data):
            answer = llm_chain.invoke(
                {"question": data["question"], "context": data["context"]},
                config=get_langfuse_callback()
            )
            return {
                "answer": answer,
                "vector_sources": data["vector_sources"],
                "kg_source_chunks": data.get("kg_source_chunks", []),
                "graph_sources": data["graph_sources"],
                "graph_paths": data.get("graph_paths", []),
                "extracted_entities": data.get("extracted_entities", {})
            }

        chain = (
            RunnablePassthrough()
            | RunnableLambda(retriever_and_merge)
            | RunnableLambda(generate_with_sources)
        )

        return chain, graph
    except Exception as e:
        raise Exception(f"システム復元エラー: {e}")


# =====================================================================
# RAGシステム構築
# =====================================================================
def build_rag_system(source_docs: list, csv_edges: list | None = None):
    """RAGシステムの構築"""

    embeddings = _create_embeddings()

    # 2段階Markdownチャンキング
    all_chunks = create_markdown_chunks(source_docs, chunk_size=1024, chunk_overlap=100)
    chunks = all_chunks

    # チャンク重複除去（ハッシュベース）
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

    enable_knowledge_graph = st.session_state.get('enable_knowledge_graph', True)

    if enable_knowledge_graph:
        st.info("🕸️ ナレッジグラフ生成中...")

        llm_provider = _settings.llm_provider.lower()
        _kg_session_id = f"{st.session_state.langfuse_session_id}_kg_{datetime.now().strftime('%H%M%S')}"

        llm = create_chat_llm(temperature=0)
        if is_langfuse_enabled():
            from langfuse.langchain import CallbackHandler as LangfuseHandler
            llm.callbacks = [LangfuseHandler(trace_context={"name": "kg_building", "session_id": _kg_session_id})]

        # VLLM (ignore_tool_usage=True) ではカスタムpromptを渡さない。
        # LLMGraphTransformerのデフォルトpromptがJSON出力形式を指示するため、
        # カスタムpromptを渡すとJSON指示が欠落しパース失敗する。
        # ドメイン固有の指示はadditional_instructionsで渡す。
        _kg_additional = (
            "抽出する: 技術用語、概念、固有名詞、プロセス名、規格名。"
            "抽出しない: 一般的な名詞（「こと」「もの」「方法」）、代名詞、動詞。"
            "RELATED_TOは他に適切な関係がない場合の最終手段として使用。"
        )
        from graphrag_core.graph.schema import get_allowed_node_types, get_allowed_relations
        _kg_kwargs = dict(
            llm=llm,
            allowed_nodes=get_allowed_node_types(),
            allowed_relationships=get_allowed_relations(),
            strict_mode=False,
            ignore_tool_usage=(llm_provider == "vllm"),
        )
        if llm_provider == "vllm":
            # VLLMではデフォルトprompt（JSON出力指示付き）を使用
            _kg_kwargs["additional_instructions"] = _kg_additional
        else:
            # Azure OpenAI等ではカスタムprompt（tool use対応）
            kg_prompt = ChatPromptTemplate.from_messages([
                ("system", KG_SYSTEM_PROMPT),
                ("user", KG_USER_PROMPT)
            ])
            _kg_kwargs["prompt"] = kg_prompt

        transformer = LLMGraphTransformer(**_kg_kwargs)

        # グラフバックエンド初期化
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PW,
            enhanced_schema=True
        )
        try:
            processed = graph.query("MATCH (c:ProcessedChunk) RETURN c.hash AS hash")
            processed_hashes = {r['hash'] for r in processed} if processed else set()
        except Exception:
            processed_hashes = set()

        # 未処理チャンクをフィルタ
        pending_chunks = [c for c in chunks if c.metadata.get("id") not in processed_hashes]
        skipped_count = len(chunks) - len(pending_chunks)
        if skipped_count > 0:
            st.info(f"📋 処理対象: {len(pending_chunks)}/{len(chunks)} チャンク（{skipped_count}件は処理済みのためスキップ）")

        # チャンクごとに処理 + 即座にDB保存
        graph_docs = []
        if pending_chunks:
            progress_bar = st.progress(0, text="ナレッジグラフ生成中...")
            for i, chunk in enumerate(pending_chunks):
                progress_bar.progress((i + 1) / len(pending_chunks), text=f"ナレッジグラフ生成中... {i+1}/{len(pending_chunks)}")
                try:
                    chunk_docs = transformer.convert_to_graph_documents([chunk])
                    graph.add_graph_documents(chunk_docs, include_source=True)
                    graph_docs.extend(chunk_docs)

                    chunk_hash = chunk.metadata.get("id")
                    if chunk_hash:
                        graph.query(
                            "MERGE (c:ProcessedChunk {hash: $hash}) SET c.processed_at = datetime()",
                            {"hash": chunk_hash}
                        )
                except Exception as e:
                    st.warning(f"チャンク {i+1} の処理でエラー: {e}")
                    continue
            progress_bar.empty()
        else:
            st.success("✅ すべてのチャンクは処理済みです")

        # CSVエッジ取り込み
        if csv_edges:
            for edge in csv_edges:
                graph.query(
                    f"""
                    MERGE (s:CSVNode {{id: $src}})
                    MERGE (t:CSVNode {{id: $tgt}})
                    MERGE (s)-[r:`{edge['label']}`]->(t)
                    """,
                    params={"src": edge["source"], "tgt": edge["target"]}
                )

        # Documentノード（add_graph_documentsが作成）にsource名を付与
        for chunk in chunks:
            chunk_id = chunk.metadata.get("id")
            doc_name = chunk.metadata.get("source", "Unknown")
            if chunk_id:
                graph.query("""
                    MATCH (d:Document {id: $chunk_id})
                    SET d.source = $doc_name
                """, params={"chunk_id": chunk_id, "doc_name": doc_name})

        # エンティティベクトル化
        if st.session_state.get('enable_entity_vector', True):
            with st.spinner("エンティティをベクトル化中..."):
                try:
                    entity_vectorizer = EntityVectorizer(PG_CONN, embeddings)
                    entities = entity_vectorizer.extract_entities_from_graph(
                        graph
                    )
                    num_saved = entity_vectorizer.add_entities(entities, graph_docs)
                    if num_saved > 0:
                        st.success(f"✅ {num_saved}個のエンティティをベクトル化しました")
                except Exception as e:
                    st.warning(f"エンティティベクトル化エラー: {e}")
    else:
        st.info("⚡ ナレッジグラフをスキップし、ベクトル検索のみを使用します")
        llm = create_chat_llm(temperature=0)
        graph_docs = []

        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PW,
            enhanced_schema=True
        )

        # CSVエッジ取り込み（ナレッジグラフOFFでもCSVは処理）
        if csv_edges:
            st.info(f"🔗 CSVから{len(csv_edges)}件のエッジを追加中...")
            for edge in csv_edges:
                graph.query(
                    f"""
                    MERGE (s:CSVNode {{id: $src}})
                    MERGE (t:CSVNode {{id: $tgt}})
                    MERGE (s)-[r:`{edge['label']}`]->(t)
                    """,
                    params={"src": edge["source"], "tgt": edge["target"]}
                )
            st.success(f"✅ CSVから{len(csv_edges)}件のエッジを追加しました")

        # CSVエッジからのエンティティベクトル化
        if csv_edges and st.session_state.get('enable_entity_vector', True):
            with st.spinner("CSVエンティティをベクトル化中..."):
                try:
                    entity_vectorizer = EntityVectorizer(PG_CONN, embeddings)
                    entities = entity_vectorizer.extract_entities_from_graph(
                        graph
                    )
                    num_saved = entity_vectorizer.add_entities(entities, [])
                    if num_saved > 0:
                        st.success(f"✅ {num_saved}個のエンティティをベクトル化しました")
                except Exception as e:
                    st.warning(f"エンティティベクトル化エラー: {e}")

        # スキーマメタ情報を Neo4j に刻印（EDC連携時の追跡用）
        try:
            from graphrag_core.graph.schema import stamp_schema_metadata, describe_schema
            stamp_schema_metadata(graph)
            st.info(f"📋 スキーマ刻印: {describe_schema()}")
        except Exception as e:
            st.warning(f"⚠️ スキーマ刻印エラー: {e}")

    # 日本語トークン化
    japanese_processor = get_japanese_processor()
    if japanese_processor and st.session_state.get('enable_japanese_search', True):
        with st.spinner("日本語トークン化中..."):
            for chunk in chunks:
                try:
                    tokenized = japanese_processor.tokenize(chunk.page_content)
                    chunk.metadata['tokenized_content'] = tokenized
                except Exception as e:
                    st.warning(f"トークン化エラー（スキップ）: {e}")
                    chunk.metadata['tokenized_content'] = None

    # PGVector保存
    if not chunks:
        st.warning("チャンクが0件のためベクトルストア保存をスキップしました")
        vector_store = None
    else:
        ids = []
        for c in chunks:
            cid = c.metadata.get("id")
            if not cid:
                raise ValueError("Chunk metadata に id がありません")
            ids.append(cid)

        ensure_embedding_id_unique(PG_CONN)
        ensure_schema_compatibility(PG_CONN)
        ensure_hnsw_index(PG_CONN)
        pg_conn_with_timeout = add_connection_timeout(PG_CONN, timeout=30)

        vector_store = batch_pgvector_from_documents(
            chunks,
            embeddings,
            connection=pg_conn_with_timeout,
            collection_name=PG_COLLECTION,
            pre_delete_collection=True,
        )

    # トークン化データをDBに反映
    if vector_store and japanese_processor and st.session_state.get('enable_japanese_search', True):
        try:
            ensure_tokenized_schema(PG_CONN)
            batch_update_tokenized(PG_CONN, chunks)
        except Exception as e:
            st.warning(f"トークン化データのDB保存エラー: {e}")

    # QAチェーン構築
    retrieval_top_k = st.session_state.get('retrieval_top_k', 5)
    vector_retriever = create_vector_retriever(vector_store, retrieval_top_k)

    config = _build_config_from_session_state()

    def retriever_and_merge(question: str):
        return qa_retriever_and_merge(
            question, graph, llm, embeddings, vector_retriever,
            PG_CONN, PG_COLLECTION, config
        )

    prompt = PromptTemplate.from_template(QA_PROMPT)

    llm_chain = (
        prompt
        | create_chat_llm(temperature=0)
        | StrOutputParser()
    )

    @observe(name="answer_generation")
    def generate_with_sources(data):
        answer = llm_chain.invoke(
            {"question": data["question"], "context": data["context"]},
            config=get_langfuse_callback()
        )
        return {
            "answer": answer,
            "vector_sources": data["vector_sources"],
            "kg_source_chunks": data.get("kg_source_chunks", []),
            "graph_sources": data["graph_sources"],
            "extracted_entities": data.get("extracted_entities", {})
        }

    chain = (
        RunnablePassthrough()
        | RunnableLambda(retriever_and_merge)
        | RunnableLambda(generate_with_sources)
    )

    return chain, graph


def update_chunks_only(source_docs: list):
    """チャンクのみ更新（グラフ再構築スキップ）"""
    embeddings = _create_embeddings()

    all_chunks = create_markdown_chunks(source_docs, chunk_size=1024, chunk_overlap=100)

    deduped = []
    seen_hashes = set()
    for chunk in all_chunks:
        digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        chunk.metadata["id"] = digest
        deduped.append(chunk)
    chunks = deduped

    st.info(f"📄 {len(chunks)}個のチャンクを生成しました")

    japanese_processor = get_japanese_processor()
    if japanese_processor and st.session_state.get('enable_japanese_search', True):
        with st.spinner("日本語トークン化中..."):
            for chunk in chunks:
                try:
                    tokenized = japanese_processor.tokenize(chunk.page_content)
                    chunk.metadata['tokenized_content'] = tokenized
                except Exception:
                    chunk.metadata['tokenized_content'] = None

    if not chunks:
        st.warning("チャンクが0件のためスキップしました")
        return None

    ensure_embedding_id_unique(PG_CONN)
    ensure_schema_compatibility(PG_CONN)
    ensure_hnsw_index(PG_CONN)
    pg_conn_with_timeout = add_connection_timeout(PG_CONN, timeout=30)

    vector_store = batch_pgvector_from_documents(
        chunks,
        embeddings,
        connection=pg_conn_with_timeout,
        collection_name=PG_COLLECTION,
        pre_delete_collection=True,
    )

    if vector_store and japanese_processor and st.session_state.get('enable_japanese_search', True):
        try:
            ensure_tokenized_schema(PG_CONN)
            batch_update_tokenized(PG_CONN, chunks)
        except Exception as e:
            st.warning(f"トークン化データのDB保存エラー: {e}")

    st.success("✅ チャンクのみ更新完了（グラフは既存を維持）")
    st.warning("⚠️ KGソースチャンクは古いMENTIONSエッジを参照するため、完全再構築を推奨します")

    return vector_store


# =====================================================================
# メインUI
# =====================================================================
st.header("📁 ドキュメントアップロード")

# 既存グラフのチェック（初回のみ）
if not st.session_state.existing_graph_loaded and not st.session_state.initialized:
    try:
        temp_graph = _create_graph_instance()
        graph_info = check_existing_graph(temp_graph)

        if graph_info['exists']:
            st.info(f"📊 既存のナレッジグラフを発見しました: ノード {graph_info['node_count']}個、リレーションシップ {graph_info['rel_count']}本")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 既存グラフを読み込む", type="primary"):
                    with st.spinner("既存グラフからシステムを復元中..."):
                        try:
                            st.session_state.chain, st.session_state.graph = restore_from_existing_graph()
                            st.session_state.initialized = True
                            st.session_state.existing_graph_loaded = True
                            st.success("✅ 既存グラフから復元完了！すぐに質問できます。")
                            st.rerun()
                        except Exception as e:
                            st.error(f"復元エラー: {e}")

            with col2:
                if st.button("🗑️ 既存グラフをクリアして新規作成"):
                    with st.spinner("既存データをクリア中..."):
                        try:
                            temp_graph.query("MATCH (n) DETACH DELETE n")
                            st.session_state.existing_graph_loaded = True
                            st.success("✅ クリア完了。新しいドキュメントをアップロードしてください。")
                            st.rerun()
                        except Exception as e:
                            st.error(f"クリアエラー: {e}")

            st.markdown("---")
    except Exception:
        pass

uploaded_files = st.file_uploader(
    "PDF/テキスト/Markdownファイルをアップロード",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    help="複数ファイルをアップロード可能。Azure DI処理済みの_azure_di.mdファイルも再利用可能"
)
csv_edges_file = st.file_uploader(
    "edges.csv (source,target,label)",
    type=["csv"],
    accept_multiple_files=False,
    help="シンプルなノード・エッジ関係をCSVで追加する場合に指定してください"
)
has_docs = bool(uploaded_files)
has_csv = bool(csv_edges_file)

if has_docs:
    st.success(f"✅ {len(uploaded_files)} ファイルがアップロードされました")
    with st.expander("📄 アップロード済みファイル"):
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size} bytes)")

if has_csv:
    st.info(f"🔗 edges.csv を受信: {csv_edges_file.name}")

# ナレッジグラフ構築ボタン
if has_docs or has_csv:
    col1, col2 = st.columns(2)
    with col1:
        new_build = st.button("🚀 新規構築", type="primary", help="処理済みデータをクリアして最初から構築")
    with col2:
        resume_build = st.button("▶️ 続きから再開", help="処理済みチャンクをスキップして続きから構築")

    st.caption("⚡ 高速オプション")
    chunks_only = st.button("📄 チャンクのみ更新", help="グラフ再構築をスキップしてPGVectorのチャンクのみ更新（高速）")

    if new_build or resume_build:
        if new_build:
            try:
                temp_graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PW)
                temp_graph.query("MATCH (c:ProcessedChunk) DELETE c")
                st.info("🗑️ 処理済みデータをクリアしました")
            except Exception as e:
                st.warning(f"クリア処理でエラー（続行します）: {e}")

        source_docs = []
        if has_docs:
            with st.spinner("ドキュメント読み込み中..."):
                try:
                    source_docs = load_documents(uploaded_files)
                    total_chars = sum(len(doc.page_content) for doc in source_docs)
                    st.info(f"📄 {len(source_docs)} ファイル読み込み完了（総文字数: {total_chars:,} 文字）")
                except Exception as e:
                    st.error(f"ファイル読み込みエラー: {e}")
                    st.stop()

        with st.spinner("ナレッジグラフ構築中... (数分かかる場合があります)"):
            try:
                csv_edges = load_csv_edges(csv_edges_file) if has_csv else []
                st.session_state.chain, st.session_state.graph = build_rag_system(source_docs, csv_edges)
                st.session_state.initialized = True
                st.session_state.uploaded_files = [f.name for f in uploaded_files] if has_docs else []
                st.session_state.graph_data_cache = None
                if 'all_node_list' in st.session_state:
                    st.session_state.all_node_list = None
                st.success("✅ ナレッジグラフ構築完了!")
            except Exception as e:
                st.error(f"構築エラー: {e}")
                import traceback
                st.code(traceback.format_exc())

    if chunks_only:
        if not has_docs:
            st.error("ドキュメントをアップロードしてください")
        else:
            with st.spinner("ドキュメント読み込み中..."):
                try:
                    source_docs = load_documents(uploaded_files)
                    total_chars = sum(len(doc.page_content) for doc in source_docs)
                    st.info(f"📄 {len(source_docs)} ファイル読み込み完了（総文字数: {total_chars:,} 文字）")
                except Exception as e:
                    st.error(f"ファイル読み込みエラー: {e}")
                    st.stop()

            with st.spinner("チャンクを更新中..."):
                try:
                    vector_store = update_chunks_only(source_docs)
                    if vector_store:
                        st.session_state.uploaded_files = [f.name for f in uploaded_files]
                except Exception as e:
                    st.error(f"チャンク更新エラー: {e}")
                    import traceback
                    st.code(traceback.format_exc())

st.markdown("---")


# =====================================================================
# Langfuse トレース用ルートラッパー
# =====================================================================
@observe(name="graphrag_qa")
def _run_traced_question(chain, question, langfuse_session_id):
    """1質問=1トレースとして階層化"""
    _lf_sid = f"{langfuse_session_id}_q{hash(question) % 10000:04d}"
    with propagate_attributes(session_id=_lf_sid):
        return chain.invoke(question)


# =====================================================================
# タブ形式UI
# =====================================================================
tab1, tab2, tab3 = st.tabs(["💬 質問応答", "🕸️ グラフ探索", "📄 登録ドキュメント"])

# ─── タブ1: 質問応答 ───────────────────────────────────────────
with tab1:
    st.header("💬 質問応答")

    if st.session_state.initialized:
        question = st.text_area("質問を入力してください:", height=150, key="question_input")

        if st.button("🔍 質問する", type="primary"):
            if question:
                with st.spinner("回答生成中..."):
                    try:
                        result = _run_traced_question(
                            st.session_state.chain, question,
                            st.session_state.langfuse_session_id
                        )

                        st.markdown("### 📝 回答")
                        st.markdown(result["answer"])

                        search_mode = st.session_state.get('search_mode', 'hybrid')
                        mode_labels = {
                            'hybrid': 'ハイブリッド検索',
                            'vector': 'ベクトル検索',
                            'keyword': 'キーワード検索'
                        }
                        doc_label = mode_labels.get(search_mode, 'ベクトル検索')

                        with st.expander(f"📚 参照ドキュメント ({doc_label})", expanded=False):
                            vector_sources = result.get("vector_sources", [])
                            if vector_sources:
                                for i, doc in enumerate(vector_sources, 1):
                                    st.markdown(f"**チャンク {i}:**")
                                    source = doc.metadata.get('source', '')
                                    if source:
                                        st.caption(f"出典: {source}")
                                    st.text(doc.page_content)
                                    if i < len(vector_sources):
                                        st.divider()
                            else:
                                st.info("ドキュメント検索結果なし")

                        with st.expander("📄 KGソースチャンク (Graph RAG)", expanded=False):
                            kg_chunks = result.get("kg_source_chunks", [])
                            if kg_chunks:
                                for i, doc in enumerate(kg_chunks, 1):
                                    st.markdown(f"**チャンク {i}:**")
                                    source = doc.metadata.get('source', 'KG')
                                    st.caption(f"出典: {source}")
                                    st.text(doc.page_content)
                                    if i < len(kg_chunks):
                                        st.divider()
                            else:
                                st.info("KGからの追加チャンクなし")

                        with st.expander("🕸️ ナレッジグラフ (Graph RAG)", expanded=False):
                            graph_paths = result.get("graph_paths", [])
                            graph_sources = result.get("graph_sources", [])
                            if graph_paths:
                                for i, p in enumerate(graph_paths, 1):
                                    st.markdown(f"**推論パス{i}:** {p.get('path_text', '')}")
                            elif graph_sources:
                                for triple in graph_sources:
                                    st.markdown(f"- `{triple.get('start')}` -[{triple.get('type')}]→ `{triple.get('end')}`")
                            else:
                                st.info("グラフ検索結果なし")

                        rerank_log = st.session_state.get("path_rerank_log", [])
                        if rerank_log:
                            with st.expander("📊 パスリランキングログ", expanded=False):
                                for line in rerank_log:
                                    st.markdown(line)
                                log_text = "\n".join(line.replace("`", "").replace("**", "") for line in rerank_log)
                                st.download_button(
                                    "ログをダウンロード",
                                    data=log_text,
                                    file_name=f"path_rerank_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                                    mime="text/plain",
                                )

                        with st.expander("🔍 抽出されたエンティティ", expanded=False):
                            extracted = result.get("extracted_entities", {})
                            if extracted:
                                llm_ents = extracted.get("llm_entities", [])
                                if llm_ents:
                                    st.markdown("**LLM抽出:**")
                                    st.write(", ".join(llm_ents))
                                vector_ents = extracted.get("vector_entities", [])
                                if vector_ents:
                                    st.markdown("**ベクトル/キーワード検索:**")
                                    for eid, score in vector_ents[:10]:
                                        st.write(f"- {eid} (score: {score:.3f})")
                                merged_ents = extracted.get("merged_entities", [])
                                if merged_ents:
                                    st.markdown("**グラフ検索に使用:**")
                                    st.write(", ".join(merged_ents[:15]))
                            else:
                                st.info("エンティティ情報なし")

                    except Exception as e:
                        st.error(f"エラー: {e}")
            else:
                st.warning("質問を入力してください")
    else:
        st.info("まずRAGシステムを初期化してください")

# ─── タブ2: グラフ探索 ────────────────────────────────────────
with tab2:
    st.header("🕸️ グラフ探索")

    if st.session_state.initialized:
        display_mode = st.radio(
            "表示モード",
            ["🕸️ グラフ可視化", "📊 データテーブル", "🔍 Cypherクエリ検索"],
            horizontal=True
        )

        st.markdown("---")

        # モード1: グラフ可視化
        if display_mode == "🕸️ グラフ可視化":
            if not show_graph:
                st.warning("サイドバーで「ナレッジグラフを表示」をONにしてください")
            else:
                viz_scope = st.radio(
                    "📊 可視化範囲",
                    ["全体表示", "部分表示（検索）"],
                    horizontal=True,
                    help="大規模グラフの場合は部分表示を推奨します"
                )

                if viz_scope == "部分表示（検索）":
                    st.markdown("### 🔍 ノード検索")

                    if 'center_nodes' not in st.session_state:
                        st.session_state.center_nodes = []

                    if 'all_node_list' not in st.session_state:
                        if st.session_state.graph_data_cache:
                            graph_data = st.session_state.graph_data_cache
                            all_nodes = list(set(
                                [item['source'] for item in graph_data] +
                                [item['target'] for item in graph_data]
                            ))
                            st.session_state.all_node_list = sorted(all_nodes)
                        else:
                            with st.spinner("ノードリスト取得中..."):
                                try:
                                    graph_data = get_enhanced_graph_data(st.session_state.graph, limit=max_nodes)
                                    st.session_state.graph_data_cache = graph_data
                                    all_nodes = list(set(
                                        [item['source'] for item in graph_data] +
                                        [item['target'] for item in graph_data]
                                    ))
                                    st.session_state.all_node_list = sorted(all_nodes)
                                except Exception as e:
                                    st.error(f"エラー: {e}")
                                    st.session_state.all_node_list = []

                    if st.session_state.all_node_list:
                        search_query = st.text_input(
                            "🔍 ノード検索（部分一致）",
                            placeholder="例: API",
                            help="検索したノードとその周辺を表示します"
                        )

                        if search_query:
                            matched_nodes = [n for n in st.session_state.all_node_list
                                            if search_query.lower() in n.lower()]
                            st.caption(f"🔍 検索結果: {len(matched_nodes)}件")

                            if matched_nodes:
                                selected_node = st.selectbox(
                                    "ノードを選択",
                                    options=[""] + matched_nodes,
                                    index=0,
                                    key="node_select_box",
                                    help="リストから1つ選んで追加してください"
                                )

                                def _add_center_node():
                                    node = st.session_state.node_select_box
                                    if node and node not in st.session_state.center_nodes:
                                        st.session_state.center_nodes.append(node)

                                def _reset_center_nodes():
                                    st.session_state.center_nodes = []

                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    st.button("➕ 中心ノードに追加", on_click=_add_center_node,
                                              disabled=not selected_node)
                                with col2:
                                    if st.session_state.center_nodes:
                                        st.button("🗑️ リセット", on_click=_reset_center_nodes)
                            else:
                                st.warning(f"「{search_query}」に一致するノードが見つかりませんでした")
                        else:
                            st.info("💡 ノード名を入力して検索してください")

                        if st.session_state.center_nodes:
                            st.markdown("---")
                            st.write("**中心ノード:**", ", ".join(st.session_state.center_nodes))

                            hop_distance = st.slider(
                                "周辺表示範囲（Hop数）",
                                min_value=1,
                                max_value=3,
                                value=2,
                                help="選択ノードから何Hop先まで表示するか"
                            )

                            if st.button("📊 サブグラフを表示", type="primary"):
                                with st.spinner("サブグラフ取得中..."):
                                    try:
                                        subgraph_data = get_enhanced_subgraph_data(
                                            st.session_state.graph,
                                            st.session_state.center_nodes,
                                            hop_distance,
                                            limit=500
                                        )

                                        if subgraph_data:
                                            unique_nodes = set()
                                            for item in subgraph_data:
                                                unique_nodes.add(item['source'])
                                                unique_nodes.add(item['target'])

                                            st.success(f"✅ サブグラフ取得完了")
                                            st.info(f"📊 表示: ノード {len(unique_nodes)}個 / エッジ {len(subgraph_data)}本")

                                            html = visualize_graph_neo4j_viz(subgraph_data)
                                            if html:
                                                st.components.v1.html(html, height=700)
                                            else:
                                                st.warning("可視化に失敗しました。")
                                        else:
                                            st.warning("選択したノードのサブグラフが見つかりませんでした")
                                    except Exception as e:
                                        st.error(f"エラー: {e}")
                                        import traceback
                                        st.code(traceback.format_exc())
                        else:
                            st.info("👆 検索してノードを追加してください")
                    else:
                        st.warning("ノードリストが取得できませんでした。先に「全体表示」でグラフを読み込んでください。")

                else:
                    # 全体表示モード
                    if st.session_state.graph_data_cache is None:
                        if st.button("📊 グラフを読み込む", type="primary"):
                            with st.spinner("グラフデータ取得中..."):
                                try:
                                    graph_data = get_enhanced_graph_data(st.session_state.graph, limit=max_nodes)
                                    st.session_state.graph_data_cache = graph_data
                                    st.success(f"✅ {len(graph_data)}件のエッジを読み込みました")
                                except Exception as e:
                                    st.error(f"エラー: {e}")

                    if st.session_state.graph_data_cache:
                        try:
                            graph_data = st.session_state.graph_data_cache

                            if not graph_data:
                                st.warning("グラフデータがありません")
                            else:
                                unique_nodes = set()
                                for item in graph_data:
                                    unique_nodes.add(item['source'])
                                    unique_nodes.add(item['target'])

                                st.info(f"📊 表示中: ノード {len(unique_nodes)}個 / エッジ {len(graph_data)}本")

                                html = visualize_graph_neo4j_viz(graph_data)
                                if html:
                                    st.components.v1.html(html, height=700)
                                else:
                                    st.warning("可視化に失敗しました。")

                            def _reload_graph():
                                st.session_state.graph_data_cache = None
                                if 'all_node_list' in st.session_state:
                                    del st.session_state['all_node_list']

                            st.button("🔄 グラフを再読み込み", on_click=_reload_graph)

                        except Exception as e:
                            st.error(f"エラー: {e}")
                            import traceback
                            st.code(traceback.format_exc())

        # モード2: データテーブル
        elif display_mode == "📊 データテーブル":
            if st.session_state.graph_data_cache is None:
                if st.button("📊 データを読み込む", type="primary", key="load_data_table"):
                    with st.spinner("データ取得中..."):
                        try:
                            graph_data = get_enhanced_graph_data(st.session_state.graph, limit=max_nodes)
                            st.session_state.graph_data_cache = graph_data
                            st.success(f"✅ {len(graph_data)}件のエッジを読み込みました")
                        except Exception as e:
                            st.error(f"エラー: {e}")

            if st.session_state.graph_data_cache:
                try:
                    graph_data = st.session_state.graph_data_cache
                    if graph_data:
                        _display_data_tables(
                            graph_data,
                            graph=st.session_state.graph,
                            enable_edit=True,
                            graph_add_node=graph_add_node,
                            graph_update_node=graph_update_node,
                            graph_delete_node=graph_delete_node,
                            graph_get_node_info=graph_get_node_info,
                            graph_add_edge=graph_add_edge,
                            graph_update_edge=graph_update_edge,
                            graph_delete_edge=graph_delete_edge,
                            graph_get_edge_info=graph_get_edge_info,
                            graph_get_data_for_cache=graph_get_data_for_cache,
                            edit_node_dialog_fn=_wrapped_edit_node_dialog,
                            edit_edge_dialog_fn=_wrapped_edit_edge_dialog,
                        )
                    else:
                        st.warning("グラフデータがありません")
                except Exception as e:
                    st.error(f"エラー: {e}")

        # モード3: Cypherクエリ検索
        elif display_mode == "🔍 Cypherクエリ検索":
            st.markdown("### 自然言語でグラフを検索")
            st.info("例: 「APIに関するグラフを見たい」「認証と関係のあるエンティティを表示」")

            with st.expander("📋 クエリテンプレート"):
                template = st.selectbox(
                    "よく使うクエリ",
                    [
                        "カスタム（自分で入力）",
                        "特定エンティティに関連するすべての関係を表示",
                        "最も接続数が多いノードTop10を表示",
                        "すべてのリレーションシップタイプを表示"
                    ]
                )

                if template == "特定エンティティに関連するすべての関係を表示":
                    entity_name = st.text_input("エンティティ名を入力:", placeholder="例: API")
                    nl_query = f"{entity_name}に関連するすべての関係を表示" if entity_name else ""
                elif template == "最も接続数が多いノードTop10を表示":
                    nl_query = "最も接続数が多いノードTop10を表示"
                elif template == "すべてのリレーションシップタイプを表示":
                    nl_query = "すべてのリレーションシップタイプとその数を表示"
                else:
                    nl_query = ""

            user_query = st.text_area(
                "自然言語クエリ:",
                value=nl_query,
                height=100,
                placeholder="例: APIに関するグラフを見たい"
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                convert_button = st.button("🔄 Cypherに変換", type="primary")

            if "generated_cypher" not in st.session_state:
                st.session_state.generated_cypher = ""

            if convert_button and user_query:
                with st.spinner("Cypherクエリを生成中..."):
                    cypher_query = natural_language_to_cypher(user_query)
                    st.session_state.generated_cypher = cypher_query

            if st.session_state.generated_cypher:
                st.markdown("### 📝 生成されたCypherクエリ")
                edited_cypher = st.text_area(
                    "Cypherクエリ（編集可能）:",
                    value=st.session_state.generated_cypher,
                    height=150,
                    key="cypher_editor"
                )

                def _clear_cypher():
                    st.session_state.generated_cypher = ""

                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    execute_button = st.button("▶️ 実行", type="primary")
                with col2:
                    st.button("🗑️ クリア", on_click=_clear_cypher)

                if execute_button and edited_cypher:
                    with st.spinner("クエリ実行中..."):
                        result = execute_cypher_and_visualize(edited_cypher, st.session_state.graph)

                        if result:
                            st.success(f"✅ {len(result)}件の結果を取得しました")

                            st.markdown("### 📊 クエリ結果")
                            import pandas as pd
                            df = pd.DataFrame(result)
                            for col in df.columns:
                                if df[col].apply(type).nunique() > 1:
                                    df[col] = df[col].astype(str)
                            st.dataframe(df, use_container_width=True)

                            if len(result) > 0 and 'source' in result[0] and 'target' in result[0] and 'relation' in result[0]:
                                st.markdown("### 🕸️ グラフ可視化")
                                html = visualize_graph_neo4j_viz(result)
                                if html:
                                    st.components.v1.html(html, height=700)

    else:
        st.info("まずRAGシステムを初期化してください")

# ─── タブ3: 登録ドキュメント ───────────────────────────────────
with tab3:
    st.header("📄 登録ドキュメント")

    if PG_CONN:
        try:
            import psycopg
            raw_conn = normalize_pg_connection_string(PG_CONN)
            with psycopg.connect(raw_conn) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT
                            COALESCE(e.cmetadata->>'source', '(unknown)') as source,
                            COUNT(*) as chunk_count
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                        WHERE c.name = %s
                        GROUP BY e.cmetadata->>'source'
                        ORDER BY chunk_count DESC
                    """, (PG_COLLECTION,))
                    rows = cur.fetchall()

            if rows:
                total_chunks = sum(r[1] for r in rows)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("総チャンク数", f"{total_chunks:,}")
                with col2:
                    st.metric("ドキュメント数", len(rows))

                st.markdown("### ソースファイル一覧")
                import pandas as pd
                df = pd.DataFrame(rows, columns=["ソースファイル", "チャンク数"])
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.info("登録されたドキュメントはありません")
        except Exception as e:
            st.error(f"DB接続エラー: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning("PG_CONNが設定されていません")

# フッター
st.markdown("---")
st.markdown("**Graph-RAG Demo** | Powered by LangChain, Neo4j & PGVector")
