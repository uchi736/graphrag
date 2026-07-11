"""
graphrag_core/ui/system.py
Non-tab system assembly: build/restore the QA chain + graph, document/CSV loading,
chunks-only update. Each function takes ctx:UIContext as its first arg and reads
env/config off ctx. The two generate_with_sources / retriever_and_merge closures
stay nested and DISTINCT per function (restore includes graph_paths; build omits it).
"""
import os
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from graphrag_core.config import get_settings, build_pipeline_config
from graphrag_core.prompts import QA_PROMPT, KG_SYSTEM_PROMPT, KG_USER_PROMPT
from graphrag_core.llm.factory import create_chat_llm
from graphrag_core.llm.langfuse_utils import (
    observe,
    get_langfuse_callback,
    is_langfuse_enabled,
)
from graphrag_core.db.utils import (
    ensure_tokenized_schema,
    ensure_hnsw_index,
    ensure_embedding_id_unique,
    ensure_schema_compatibility,
    add_connection_timeout,
    retry_on_timeout,
    batch_pgvector_from_documents,
    batch_update_tokenized,
)
from graphrag_core.text.japanese import get_japanese_processor
from graphrag_core.text.chunking import create_markdown_chunks
from graphrag_core.retrieval.entity_vector import EntityVectorizer
from graphrag_core.retrieval.pipeline import retriever_and_merge as qa_retriever_and_merge
from graphrag_core.graph.provenance import stamp_graph_provenance, graph_matches_collection
from graphrag_core.graph.incremental import make_chunk_id
from graphrag_core.document.pdf import load_pdf_text
from graphrag_core.services.qa import QADeps, make_qa_chain
from graphrag_core.services.retrievers import create_vector_retriever

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import TextLoader
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


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
def restore_from_existing_graph(ctx):
    """グラフバックエンドとPGVectorから既存データを使ってシステムを復元"""
    try:
        graph = ctx.create_graph_instance()

        embeddings = ctx.create_embeddings()
        pg_conn_with_timeout = add_connection_timeout(ctx.pg_conn, timeout=30)

        def create_vector_store():
            return PGVector(
                connection=pg_conn_with_timeout,
                embeddings=embeddings,
                collection_name=ctx.pg_collection
            )

        vector_store = retry_on_timeout(create_vector_store, max_retries=3, delay=2.0)

        llm = create_chat_llm(temperature=0)

        # 検索+回答生成は services.qa に一元化（KGゲート/graph_paths含む）。
        # config_provider=ctx.build_config で「設定変更が再構築なしで即反映」を維持。
        deps = QADeps(
            graph=graph, llm=llm, embeddings=embeddings,
            vector_store=vector_store,
            pg_conn=ctx.pg_conn, pg_collection=ctx.pg_collection,
        )
        chain = make_qa_chain(deps, ctx.build_config)

        return chain, graph
    except Exception as e:
        raise Exception(f"システム復元エラー: {e}")


# =====================================================================
# RAGシステム構築
# =====================================================================
def build_rag_system(ctx, source_docs: list, csv_edges: list | None = None):
    """RAGシステムの構築"""

    # 構築結果の統計（呼び出し側が成功/部分成功/失敗を判定するのに使う）。
    # KGパスのチャンクループで上書きされる。ベクトルのみ構築では 0/0/0 のまま＝成功扱い。
    st.session_state.last_build_stats = {"ok": 0, "err": 0, "total": 0}

    embeddings = ctx.create_embeddings()

    # 2段階Markdownチャンキング
    all_chunks = create_markdown_chunks(source_docs, chunk_size=1024, chunk_overlap=100)
    chunks = all_chunks

    # チャンク重複除去（ハッシュベース）
    deduped = []
    seen_hashes = set()
    for chunk in chunks:
        # ID = sha256(doc_id + 本文): 増分更新(incremental.py)・build_kg.py と同一体系
        digest = make_chunk_id(chunk.metadata.get("source", ""), chunk.page_content)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        chunk.metadata["id"] = digest
        deduped.append(chunk)
    chunks = deduped

    enable_knowledge_graph = st.session_state.get('enable_knowledge_graph', True)

    if enable_knowledge_graph:
        st.info("🕸️ ナレッジグラフ生成中...")

        llm_provider = ctx.settings.llm_provider.lower()
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
            url=ctx.neo4j_uri,
            username=ctx.neo4j_user,
            password=ctx.neo4j_pw,
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

        # チャンクごとに処理 + 即座にDB保存（成否をカウントして成功表示をゲートする）
        graph_docs = []
        ok_count = 0
        err_count = 0
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
                    ok_count += 1
                except Exception as e:
                    err_count += 1
                    st.warning(f"チャンク {i+1} の処理でエラー: {e}")
                    continue
            progress_bar.empty()
        else:
            st.info("📋 すべてのチャンクは処理済みです")
        st.session_state.last_build_stats = {"ok": ok_count, "err": err_count, "total": len(pending_chunks)}

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
                    entity_vectorizer = EntityVectorizer(ctx.pg_conn, embeddings)
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
            url=ctx.neo4j_uri,
            username=ctx.neo4j_user,
            password=ctx.neo4j_pw,
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
                    entity_vectorizer = EntityVectorizer(ctx.pg_conn, embeddings)
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

        ensure_embedding_id_unique(ctx.pg_conn)
        ensure_schema_compatibility(ctx.pg_conn)
        ensure_hnsw_index(ctx.pg_conn)
        pg_conn_with_timeout = add_connection_timeout(ctx.pg_conn, timeout=30)

        vector_store = batch_pgvector_from_documents(
            chunks,
            embeddings,
            connection=pg_conn_with_timeout,
            collection_name=ctx.pg_collection,
            pre_delete_collection=True,
        )

    # トークン化データをDBに反映
    if vector_store and japanese_processor and st.session_state.get('enable_japanese_search', True):
        try:
            ensure_tokenized_schema(ctx.pg_conn)
            batch_update_tokenized(ctx.pg_conn, chunks)
        except Exception as e:
            st.warning(f"トークン化データのDB保存エラー: {e}")

    # グラフ↔コレクション整合性のため、このグラフの出自コレクションを刻印する。
    # 以後アプリ実行時にこの出自と現 PG_COLLECTION を照合し、不整合なら KG をスキップする。
    stamp_graph_provenance(graph, ctx.pg_collection, doc_count=len(chunks))

    # QAチェーン構築（services.qa に一元化。restore 側と完全に同一実装）
    deps = QADeps(
        graph=graph, llm=llm, embeddings=embeddings,
        vector_store=vector_store,
        pg_conn=ctx.pg_conn, pg_collection=ctx.pg_collection,
    )
    chain = make_qa_chain(deps, ctx.build_config)

    return chain, graph


def update_chunks_only(ctx, source_docs: list):
    """チャンクのみ更新（グラフ再構築スキップ）"""
    embeddings = ctx.create_embeddings()

    all_chunks = create_markdown_chunks(source_docs, chunk_size=1024, chunk_overlap=100)

    deduped = []
    seen_hashes = set()
    for chunk in all_chunks:
        # ID = sha256(doc_id + 本文): 増分更新(incremental.py)・build_kg.py と同一体系
        digest = make_chunk_id(chunk.metadata.get("source", ""), chunk.page_content)
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

    ensure_embedding_id_unique(ctx.pg_conn)
    ensure_schema_compatibility(ctx.pg_conn)
    ensure_hnsw_index(ctx.pg_conn)
    pg_conn_with_timeout = add_connection_timeout(ctx.pg_conn, timeout=30)

    vector_store = batch_pgvector_from_documents(
        chunks,
        embeddings,
        connection=pg_conn_with_timeout,
        collection_name=ctx.pg_collection,
        pre_delete_collection=True,
    )

    if vector_store and japanese_processor and st.session_state.get('enable_japanese_search', True):
        try:
            ensure_tokenized_schema(ctx.pg_conn)
            batch_update_tokenized(ctx.pg_conn, chunks)
        except Exception as e:
            st.warning(f"トークン化データのDB保存エラー: {e}")

    st.success("✅ チャンクのみ更新完了（グラフは既存を維持）")
    st.warning("⚠️ KGソースチャンクは古いMENTIONSエッジを参照するため、完全再構築を推奨します")

    return vector_store
