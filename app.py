"""
Streamlit UI for Graph-RAG
===========================
シンプルなGraph-RAG用のStreamlitアプリケーション
- PDF/テキストファイルアップロード
- 質問入力とRAG実行
- ナレッジグラフの可視化
"""
import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import tempfile
from typing import List
import hashlib
import fitz  # PyMuPDF
import json

# LangChain imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
# LLM Factory for provider selection
from llm_factory import create_chat_llm, get_llm_provider_info
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import TextLoader
try:
    from langchain_community.graphs.graph_document import GraphDocument
except ImportError:
    from langchain_community.graphs import GraphDocument
from langchain_postgres import PGVector

# 日本語ハイブリッド検索
from japanese_text_processor import get_japanese_processor, SUDACHI_AVAILABLE
from hybrid_retriever import HybridRetriever
from db_utils import normalize_pg_connection_string, ensure_tokenized_schema, ensure_hnsw_index

# エンティティベクトル化
from entity_vectorizer import EntityVectorizer

try:
    from langchain_community.retrievers.graph import GraphRetriever
except ImportError:
    try:
        from langchain_graph_retriever import GraphRetriever
    except ImportError:
        from langchain_graph_retriever.graph_retriever import GraphRetriever

try:
    from langchain_community.retrievers.parent_document import ParentDocumentRetriever
    HAS_PARENT = True
except ImportError:
    try:
        from langchain.retrievers.parent_document import ParentDocumentRetriever
        HAS_PARENT = True
    except ImportError:
        HAS_PARENT = False

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

# 環境変数読み込み
load_dotenv()

# Streamlit設定
st.set_page_config(
    page_title="Graph-RAG Demo",
    page_icon="🔗",
    layout="wide"
)

# セッションステートでバックエンド管理（早期初期化）
if "graph_backend" not in st.session_state:
    st.session_state.graph_backend = os.getenv("GRAPH_BACKEND", "networkx").lower()

# タイトルをバックエンドに応じて動的に変更
st.title(f"🔗 Graph-RAG with {st.session_state.graph_backend.upper()} & PGVector")

# サイドバー: 環境設定確認
with st.sidebar:
    st.header("⚙️ 設定")

    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PW = os.getenv("NEO4J_PW")
    PG_CONN = os.getenv("PG_CONN")
    if PG_CONN and not os.getenv("PGVECTOR_CONNECTION_STRING"):
        # Keep PGVector's expected env var in sync with the existing PG_CONN setting
        os.environ["PGVECTOR_CONNECTION_STRING"] = PG_CONN

    # 必須環境変数チェック（OpenAI, PGVector）
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, PG_CONN]):
        st.error("環境変数が不足しています。.envファイルを確認してください。")
        st.stop()

    # LLM Provider Status
    st.markdown("---")
    st.markdown("### 🤖 LLM Provider")
    llm_info = get_llm_provider_info()
    st.info(f"{llm_info['status']}\n\nProvider: {llm_info['provider']}\nModel: {llm_info['model']}")

    st.markdown("---")
    st.markdown("### 🗄️ グラフバックエンド")

    # バックエンド選択UI
    backend_options = {
        "NetworkX (軽量・Neo4j不要)": "networkx",
        "Neo4j (高性能・大規模)": "neo4j"
    }

    current_backend_label = [k for k, v in backend_options.items()
                              if v == st.session_state.graph_backend][0]

    selected_backend = st.radio(
        "バックエンド選択",
        list(backend_options.keys()),
        index=list(backend_options.values()).index(st.session_state.graph_backend),
        help="NetworkX: 即座に使用可能、小〜中規模データ / Neo4j: 大規模データ・高度なクエリ",
        label_visibility="collapsed"
    )

    # バックエンド切り替え検出
    new_backend = backend_options[selected_backend]
    if new_backend != st.session_state.graph_backend:
        st.warning("⚠️ バックエンドを切り替えると、既存のグラフデータはクリアされます。")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 切り替える", type="primary", use_container_width=True, key="switch_backend"):
                # データクリア
                st.session_state.chain = None
                st.session_state.graph = None
                st.session_state.initialized = False
                st.session_state.uploaded_files = []
                st.session_state.existing_graph_loaded = False
                st.session_state.graph_data_cache = None
                st.session_state.graph_backend = new_backend
                st.success(f"✅ {new_backend.upper()}に切り替えました")
                st.rerun()
        with col2:
            if st.button("❌ キャンセル", use_container_width=True, key="cancel_switch"):
                st.rerun()
        st.stop()  # 切り替え確認中は以降の処理を停止

    # Neo4j使用時のみNeo4j設定を必須化
    if st.session_state.graph_backend == "neo4j":
        if not all([NEO4J_URI, NEO4J_USER, NEO4J_PW]):
            st.error("❌ Neo4jを使用するには NEO4J_URI, NEO4J_USER, NEO4J_PW が必要です。")
            st.info("💡 NetworkXに切り替えると、Neo4j設定なしで使用できます。")
            if st.button("NetworkXに切り替え", key="fallback_to_networkx"):
                st.session_state.graph_backend = "networkx"
                st.rerun()
            st.stop()

        # Neo4j接続テスト
        try:
            with st.spinner("Neo4j接続確認中..."):
                test_graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PW)
                del test_graph
            st.success(f"✅ Neo4j接続成功")
        except Exception as e:
            st.error(f"❌ Neo4j接続エラー: {str(e)[:100]}")
            st.info("💡 NetworkXに切り替えますか？")
            if st.button("NetworkXに切り替え", key="fallback_on_error"):
                st.session_state.graph_backend = "networkx"
                st.rerun()
            st.stop()
    else:
        st.success(f"✅ NetworkXモード (Neo4j設定不要)")

    st.markdown("---")
    st.markdown("### 📊 グラフ可視化設定")

    viz_engine = st.radio(
        "可視化エンジン",
        ["Pyvis (推奨)", "Streamlit-Agraph"],
        index=0,
        help="Pyvisは高度な物理演算とリッチなビジュアル、Agraphは軽量でシンプル"
    )

    show_graph = st.checkbox("ナレッジグラフを表示", value=True)

    if show_graph:
        max_nodes = st.slider("最大表示ノード数", 50, 500, 200, 50)

    st.markdown("---")
    st.markdown("### 🔍 検索設定")

    # TopK設定（検索結果数）
    retrieval_top_k = st.slider(
        "検索結果数 (Top-K)",
        min_value=1,
        max_value=20,
        value=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        step=1,
        help="RAG検索で取得するチャンク数。多いほど文脈が豊富になりますが、処理時間が増加します。"
    )
    st.session_state.retrieval_top_k = retrieval_top_k

    # ナレッジグラフ機能設定
    st.markdown("---")
    st.markdown("### 🕸️ ナレッジグラフ")

    enable_knowledge_graph = st.checkbox(
        "ナレッジグラフ生成を有効化",
        value=os.getenv("ENABLE_KNOWLEDGE_GRAPH", "true").lower() == "true",
        help="テキストからエンティティと関係性を抽出してグラフ構造を生成します。処理時間が増加しますが、より高度な質問応答が可能になります。"
    )
    st.session_state.enable_knowledge_graph = enable_knowledge_graph

    if enable_knowledge_graph:
        st.info("🔍 ナレッジグラフ: 有効\nエンティティと関係性を抽出し、グラフベースの推論を行います")

        # グラフ探索ホップ数設定
        graph_hop_count = st.slider(
            "グラフ探索ホップ数",
            min_value=1,
            max_value=3,
            value=int(os.getenv("GRAPH_HOP_COUNT", "1")),
            step=1,
            help="1hop=直接関係のみ、2hop=友達の友達まで、3hop=さらに間接的な関係まで探索"
        )
        st.session_state.graph_hop_count = graph_hop_count

        # エンティティベクトル検索設定
        enable_entity_vector = st.checkbox(
            "エンティティベクトル検索",
            value=os.getenv("ENABLE_ENTITY_VECTOR_SEARCH", "true").lower() == "true",
            help="エンティティの類似度検索を有効化。類義語や関連語も検索可能になります。"
        )
        st.session_state.enable_entity_vector = enable_entity_vector

        if enable_entity_vector:
            entity_similarity_threshold = st.slider(
                "エンティティ類似度閾値",
                min_value=0.5,
                max_value=1.0,
                value=float(os.getenv("ENTITY_SIMILARITY_THRESHOLD", "0.7")),
                step=0.05,
                help="エンティティ検索の類似度閾値。低いほど幅広く検索します。"
            )
            st.session_state.entity_similarity_threshold = entity_similarity_threshold
    else:
        st.warning("⚡ ナレッジグラフ: 無効\nベクトル検索のみ使用（高速モード）")

    # 日本語ハイブリッド検索設定
    if SUDACHI_AVAILABLE:
        enable_jp_search = st.checkbox(
            "日本語ハイブリッド検索",
            value=os.getenv("ENABLE_JAPANESE_SEARCH", "true").lower() == "true",
            help="ベクトル検索とキーワード検索を組み合わせます（精度向上）"
        )

        if enable_jp_search:
            search_mode = st.radio(
                "検索モード",
                ["ハイブリッド (推奨)", "ベクトルのみ", "キーワードのみ"],
                help="ハイブリッド: RRFでスコア統合 / ベクトル: 意味検索 / キーワード: 全文検索"
            )

            # 検索モードをセッションステートに保存
            mode_map = {
                "ハイブリッド (推奨)": "hybrid",
                "ベクトルのみ": "vector",
                "キーワードのみ": "keyword"
            }
            st.session_state.search_mode = mode_map[search_mode]
            st.session_state.enable_japanese_search = True
        else:
            st.session_state.search_mode = "vector"
            st.session_state.enable_japanese_search = False
    else:
        st.warning("⚠️ sudachipy未インストール")
        st.caption("ベクトル検索のみ使用します")
        with st.expander("インストール方法"):
            st.code("pip install sudachipy sudachidict_core")
        st.session_state.search_mode = "vector"
        st.session_state.enable_japanese_search = False

    st.markdown("---")
    st.markdown("### 🗑️ データベース管理")

    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = False

    if not st.session_state.confirm_delete:
        if st.button("🗑️ データベースをクリア", use_container_width=True):
            st.session_state.confirm_delete = True
            st.rerun()
    else:
        st.warning("⚠️ 本当にすべてのデータを削除しますか？")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ はい、削除", type="primary", use_container_width=True):
                with st.spinner("データベースをクリア中..."):
                    try:
                        # グラフバックエンドクリア
                        if st.session_state.graph_backend == "neo4j":
                            temp_graph = Neo4jGraph(
                                url=NEO4J_URI,
                                username=NEO4J_USER,
                                password=NEO4J_PW,
                                enhanced_schema=True
                            )
                            temp_graph.query("MATCH (n) DETACH DELETE n")
                        else:  # networkx
                            from networkx_graph import NetworkXGraph
                            temp_graph = NetworkXGraph(storage_path="graph.pkl", auto_save=True)
                            temp_graph.graph.clear()
                            temp_graph.node_metadata.clear()
                            temp_graph.edge_metadata.clear()
                            temp_graph.save()

                        # PGVectorクリア
                        from langchain_community.vectorstores import PGVector
                        try:
                            # PGVectorのテーブルを削除
                            import psycopg2
                            conn = psycopg2.connect(PG_CONN)
                            cur = conn.cursor()
                            cur.execute("DROP TABLE IF EXISTS langchain_pg_collection CASCADE")
                            cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE")
                            conn.commit()
                            cur.close()
                            conn.close()
                        except Exception as e:
                            st.warning(f"PGVectorクリアで警告: {e}")

                        # セッションステートリセット
                        st.session_state.chain = None
                        st.session_state.graph = None
                        st.session_state.initialized = False
                        st.session_state.uploaded_files = []
                        st.session_state.existing_graph_loaded = False
                        st.session_state.graph_data_cache = None
                        st.session_state.confirm_delete = False

                        st.success("✅ データベースをクリアしました")
                        st.rerun()
                    except Exception as e:
                        st.error(f"クリアエラー: {e}")
                        st.session_state.confirm_delete = False
        with col2:
            if st.button("❌ キャンセル", use_container_width=True):
                st.session_state.confirm_delete = False
                st.rerun()

# セッションステート初期化
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

# 既存データチェック関数（バックエンド共通）
def check_existing_graph(graph, backend: str) -> dict:
    """グラフバックエンドに既存のグラフデータがあるかチェック"""
    try:
        if backend == "neo4j":
            query = """
            MATCH (n)
            RETURN count(n) AS node_count
            """
            result = graph.query(query)
            node_count = result[0]['node_count'] if result else 0

            if node_count > 0:
                query_rel = """
                MATCH ()-[r]->()
                RETURN count(r) AS rel_count
                """
                result_rel = graph.query(query_rel)
                rel_count = result_rel[0]['rel_count'] if result_rel else 0

                return {
                    'exists': True,
                    'node_count': node_count,
                    'rel_count': rel_count
                }
        else:  # networkx
            node_count = graph.graph.number_of_nodes()
            rel_count = graph.graph.number_of_edges()

            if node_count > 0:
                return {
                    'exists': True,
                    'node_count': node_count,
                    'rel_count': rel_count
                }

        return {'exists': False, 'node_count': 0, 'rel_count': 0}
    except Exception as e:
        st.error(f"グラフ接続エラー: {e}")
        return {'exists': False, 'node_count': 0, 'rel_count': 0}

# 既存グラフからシステムを復元
def restore_from_existing_graph():
    """グラフバックエンドとPGVectorから既存データを使ってシステムを復元"""
    try:
        # グラフ接続
        if st.session_state.graph_backend == "neo4j":
            graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PW,
                enhanced_schema=False  # APOC不要
            )
        else:  # networkx
            from networkx_graph import NetworkXGraph
            graph = NetworkXGraph(storage_path="graph.pkl", auto_save=True)

        # PGVector接続
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
        vector_store = PGVector(
            connection=PG_CONN,
            embeddings=embeddings
        )

        # Vector Retriever構築
        # TopK値を取得（デフォルト: 5）
        retrieval_top_k = st.session_state.get('retrieval_top_k', 5)

        if HAS_PARENT:
            vector_retriever = ParentDocumentRetriever(vector_store, search_kwargs={"k": retrieval_top_k})
        else:
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": retrieval_top_k})

        # エンティティ抽出関数（ハイブリッド版）
        def extract_entities_from_question(question: str) -> List[str]:
            """LLMとベクトル検索を使って質問からエンティティを抽出"""
            entities = []

            # 1. LLMによるエンティティ抽出
            extraction_prompt = f"""以下の質問文から、固有名詞や重要なエンティティ（人物、場所、物）を抽出してください。
エンティティのみをカンマ区切りで出力してください。説明は不要です。

質問: {question}

エンティティ:"""
            try:
                llm = create_chat_llm(temperature=0)
                response = llm.invoke(extraction_prompt)
                llm_entities = [e.strip() for e in response.content.split(',') if e.strip()]
                entities.extend(llm_entities)
            except Exception:
                # フォールバック: 簡易的なキーワード抽出
                entities.extend([w for w in question.split() if len(w) > 1])

            # 2. ベクトル検索によるエンティティ抽出（有効な場合）
            if st.session_state.get('enable_entity_vector', False):
                try:
                    entity_vectorizer = EntityVectorizer(PG_CONN, embeddings)

                    # 質問のベクトルで類似エンティティを検索
                    similarity_threshold = st.session_state.get('entity_similarity_threshold', 0.7)
                    similar_entities = entity_vectorizer.search_similar_entities(
                        question,
                        k=10,
                        score_threshold=similarity_threshold
                    )

                    # 検索結果をログ出力
                    if similar_entities:
                        print(f"[Entity Vector Search] Found {len(similar_entities)} similar entities")
                        for eid, score in similar_entities[:3]:
                            print(f"  - {eid}: {score:.3f}")

                    # エンティティIDのみを追加（重複排除）
                    for entity_id, score in similar_entities:
                        if entity_id not in entities:
                            entities.append(entity_id)

                except Exception as e:
                    # ベクトル検索が失敗してもLLM結果を使用
                    print(f"[Entity Vector Search Error] {e}")

            return entities

        def rank_relations_by_relevance(question: str, relations: list, top_k: int = 15) -> list:
            """LLMを使って関係性の質問への関連度をスコアリング"""
            if not relations:
                return []

            # 関係性リストをテキスト化
            relations_text = "\n".join([
                f"{i+1}. {r['start']} -[{r['type']}]-> {r['end']}"
                for i, r in enumerate(relations)
            ])

            ranking_prompt = f"""以下の質問に対して、各グラフ関係性の関連度を0-10でスコアリングしてください。

【質問】
{question}

【グラフ関係性】
{relations_text}

【指示】
- 各行の番号と関連度スコア（0-10）を「番号:スコア」形式で出力
- 質問に直接関連する関係性は高スコア（8-10）
- 間接的に関連する関係性は中スコア（4-7）
- 無関係な関係性は低スコア（0-3）
- 説明不要、スコアのみ出力

【出力例】
1:9
2:3
3:7

【出力】"""

            try:
                llm = create_chat_llm(temperature=0)
                response = llm.invoke(ranking_prompt)

                # スコアをパース
                scores = {}
                for line in response.content.strip().split('\n'):
                    if ':' in line:
                        try:
                            idx, score = line.split(':')
                            scores[int(idx.strip())] = float(score.strip())
                        except:
                            continue

                # スコアでソートして上位top_k件を返す
                ranked_relations = []
                for i, relation in enumerate(relations, 1):
                    score = scores.get(i, 0)
                    ranked_relations.append((score, relation))

                ranked_relations.sort(reverse=True, key=lambda x: x[0])
                return [rel for score, rel in ranked_relations[:top_k]]

            except Exception as e:
                # LLMリランキング失敗時は元のリストをそのまま返す
                return relations[:top_k]

        # グラフ検索関数（N-hopトラバーサル対応）
        def get_graph_context(question: str) -> list:
            """質問からエンティティを抽出し、N-hopトラバーサルでサブグラフを取得"""
            # 1. エンティティ抽出
            entities = extract_entities_from_question(question)
            if not entities:
                return []

            # 2. ホップ数を取得
            hop_count = st.session_state.get('graph_hop_count', 1)

            # 3. ホップ数に応じたクエリを実行
            if hop_count == 1:
                # 1-hop: 直接関係のみ
                query = """
                UNWIND $entities AS entity
                MATCH (n)
                WHERE n.id CONTAINS entity
                AND NOT n.id =~ '[0-9a-f]{32}'
                WITH collect(DISTINCT n) AS matched_nodes

                UNWIND matched_nodes AS start_node
                MATCH (start_node)-[r]-(connected_node)
                WHERE type(r) <> 'MENTIONS'
                AND NOT connected_node.id =~ '[0-9a-f]{32}'

                WITH r, startNode(r) AS actual_start, endNode(r) AS actual_end
                RETURN DISTINCT actual_start.id AS start, type(r) AS type, actual_end.id AS end
                LIMIT 30
                """
                top_k = 15
            elif hop_count == 2:
                # 2-hop: 可変長パス [*1..2]
                query = """
                UNWIND $entities AS entity
                MATCH (n)
                WHERE n.id CONTAINS entity
                AND NOT n.id =~ '[0-9a-f]{32}'
                WITH collect(DISTINCT n) AS matched_nodes

                UNWIND matched_nodes AS start_node
                MATCH path = (start_node)-[*1..2]-(end_node)
                WHERE ALL(r IN relationships(path) WHERE type(r) <> 'MENTIONS')
                AND ALL(node IN nodes(path) WHERE NOT node.id =~ '[0-9a-f]{32}')
                AND start_node <> end_node

                WITH relationships(path) AS rels
                UNWIND range(0, size(rels)-1) AS i
                WITH rels[i] AS r, startNode(rels[i]) AS s, endNode(rels[i]) AS e
                RETURN DISTINCT s.id AS start, type(r) AS type, e.id AS end
                LIMIT 50
                """
                top_k = 20
            else:  # hop_count == 3
                # 3-hop: 可変長パス [*1..3]
                query = """
                UNWIND $entities AS entity
                MATCH (n)
                WHERE n.id CONTAINS entity
                AND NOT n.id =~ '[0-9a-f]{32}'
                WITH collect(DISTINCT n) AS matched_nodes

                UNWIND matched_nodes AS start_node
                MATCH path = (start_node)-[*1..3]-(end_node)
                WHERE ALL(r IN relationships(path) WHERE type(r) <> 'MENTIONS')
                AND ALL(node IN nodes(path) WHERE NOT node.id =~ '[0-9a-f]{32}')
                AND start_node <> end_node

                WITH relationships(path) AS rels
                UNWIND range(0, size(rels)-1) AS i
                WITH rels[i] AS r, startNode(rels[i]) AS s, endNode(rels[i]) AS e
                RETURN DISTINCT s.id AS start, type(r) AS type, e.id AS end
                LIMIT 80
                """
                top_k = 25

            try:
                result = graph.query(query, params={"entities": entities})
                if result:
                    # 4. LLMリランキングで関連度の高い関係性のみに絞る
                    result = rank_relations_by_relevance(question, result, top_k=top_k)
                return result if result else []
            except Exception as e:
                # フォールバック: 単純な1-hopマッチング
                fallback_query = """
                MATCH (n)-[r]->(m)
                WHERE (
                    ANY(entity IN $entities WHERE n.id CONTAINS entity OR m.id CONTAINS entity)
                )
                AND type(r) <> 'MENTIONS'
                AND NOT n.id =~ '[0-9a-f]{32}'
                AND NOT m.id =~ '[0-9a-f]{32}'
                RETURN DISTINCT n.id AS start, type(r) AS type, m.id AS end
                LIMIT 20
                """
                try:
                    result = graph.query(fallback_query, params={"entities": entities})
                    if result:
                        result = rank_relations_by_relevance(question, result, top_k=15)
                    return result if result else []
                except Exception:
                    return []

        # チェイン構築（Graph-First Retrieval）
        def retriever_and_merge(question: str):
            # 1. ナレッジグラフが有効な場合のみグラフ検索を実行
            triples = []
            enable_knowledge_graph = st.session_state.get('enable_knowledge_graph', True)

            if enable_knowledge_graph:
                triples = get_graph_context(question)

            # 2. グラフ検索結果があればそれを使用、なければベクトル検索を補助的に使用
            docs = []
            if triples:
                # グラフから関連エンティティを取得し、それに関連するドキュメントチャンクを取得
                entity_names = list(set([t.get('start') for t in triples] + [t.get('end') for t in triples]))

                # エンティティに関連するチャンクを取得
                if entity_names:
                    chunk_query = """
                    UNWIND $entity_names AS entity_name
                    MATCH (e {id: entity_name})<-[:MENTIONS]-(chunk)
                    WHERE chunk.id =~ '[0-9a-f]{32}'
                    RETURN DISTINCT chunk.id AS chunk_id, chunk.text AS text
                    LIMIT 5
                    """
                    try:
                        chunk_results = graph.query(chunk_query, params={"entity_names": entity_names})
                        if chunk_results:
                            # グラフから取得したチャンクをドキュメントとして追加
                            from langchain_core.documents import Document
                            docs = [Document(page_content=r.get('text', ''), metadata={'id': r.get('chunk_id')})
                                   for r in chunk_results if r.get('text')]
                    except Exception:
                        pass

            # 3. グラフからドキュメントが取得できない場合はベクトル検索を使用
            if not docs:
                # ハイブリッド検索を使用（有効な場合）
                if st.session_state.get('enable_japanese_search', False) and SUDACHI_AVAILABLE:
                    try:
                        hybrid_retriever = HybridRetriever(PG_CONN, collection_name="graphrag")
                        query_embedding = embeddings.embed_query(question)
                        search_type = st.session_state.get('search_mode', 'hybrid')

                        # TopK値を取得
                        retrieval_top_k = st.session_state.get('retrieval_top_k', 5)

                        hybrid_results = hybrid_retriever.search(
                            query_text=question,
                            query_vector=query_embedding,
                            k=retrieval_top_k,
                            search_type=search_type
                        )

                        # LangChain Document形式に変換
                        from langchain_core.documents import Document
                        docs = [
                            Document(
                                page_content=r['text'],
                                metadata=r['metadata']
                            ) for r in hybrid_results
                        ]
                    except Exception as e:
                        st.warning(f"ハイブリッド検索エラー（ベクトル検索にフォールバック）: {e}")
                        docs = vector_retriever.invoke(question)
                else:
                    # 従来のベクトル検索
                    docs = vector_retriever.invoke(question)

            graph_lines = [
                f"{t.get('start')} -[{t.get('type')}]→ {t.get('end')}"
                for t in triples
            ] if triples else ["(グラフデータなし)"]

            context = (
                "<GRAPH_CONTEXT>\n" + "\n".join(graph_lines) + "\n</GRAPH_CONTEXT>\n\n" +
                "<DOCUMENT_CONTEXT>\n" + "\n---\n".join(d.page_content for d in docs) + "\n</DOCUMENT_CONTEXT>"
            )
            return {
                "context": context,
                "question": question,
                "vector_sources": docs,
                "graph_sources": triples
            }

        prompt = PromptTemplate.from_template(
            """あなたはドキュメントの専門家です。\n質問: {question}\n\n{context}\n\n---\n上記情報のみを根拠に、日本語で網羅的かつ正確に回答してください。"""
        )

        # LLM呼び出し部分
        llm_chain = (
            prompt
            | create_chat_llm(temperature=0)
            | StrOutputParser()
        )

        # ソース情報を保持する関数
        def generate_with_sources(data):
            answer = llm_chain.invoke({"question": data["question"], "context": data["context"]})
            return {
                "answer": answer,
                "vector_sources": data["vector_sources"],
                "graph_sources": data["graph_sources"]
            }

        chain = (
            RunnablePassthrough()
            | RunnableLambda(retriever_and_merge)
            | RunnableLambda(generate_with_sources)
        )

        return chain, graph

    except Exception as e:
        raise Exception(f"システム復元エラー: {e}")

# ドキュメント読み込み関数
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
                # PyMuPDF (fitz) で高精度抽出
                pdf_doc = fitz.open(tmp_path)
                text_parts = []
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    # レイアウト保持・ソート付きでテキスト抽出
                    text = page.get_text("text", sort=True)
                    if text.strip():  # 空ページをスキップ
                        text_parts.append(text)
                pdf_doc.close()
                text_content = "\n\n".join(text_parts)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_path, encoding='utf-8')
                docs = loader.load()
                text_content = "\n".join([doc.page_content for doc in docs])
            else:
                # その他のテキストファイル
                text_content = uploaded_file.getvalue().decode('utf-8')

            # メタデータ付きドキュメント作成
            all_docs.append(Document(
                page_content=text_content,
                metadata={"source": file_name}
            ))
        finally:
            os.unlink(tmp_path)

    return all_docs

# 初期化関数
def load_csv_edges(uploaded_file):
    """CSV( source,target,label ) を読み込みシンプルなエッジリストを返す"""
    if not uploaded_file:
        return []
    import csv
    import io

    # UTF-8-sig (BOM付き) にも対応
    try:
        text = uploaded_file.getvalue().decode("utf-8-sig")
    except Exception:
        try:
            text = uploaded_file.getvalue().decode("utf-8")
        except Exception:
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

    reader = csv.DictReader(io.StringIO(text))
    edges = []

    for row in reader:
        if not row:
            continue

        # ヘッダーの空白も考慮してキーを正規化（小文字化・空白除去）
        normalized_row = {k.strip().lower() if k else k: v for k, v in row.items()}

        src = (normalized_row.get("source") or normalized_row.get("from") or normalized_row.get("src") or "").strip()
        tgt = (normalized_row.get("target") or normalized_row.get("to") or normalized_row.get("dst") or "").strip()
        rel = (normalized_row.get("label") or normalized_row.get("relation") or normalized_row.get("rel") or "RELATED_TO").strip()

        if not src or not tgt:
            continue
        edges.append({"source": src, "target": tgt, "label": rel})

    return edges


def build_rag_system(source_docs: list, csv_edges: list | None = None):
    """RAGシステムの構築"""

    # チャンク分割（RecursiveCharacterTextSplitter: 重複を防ぐ）
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY
    )
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=500,           # 500文字ごとに分割
        chunk_overlap=100,        # 100文字オーバーラップ（文脈保持）
        separators=["\n\n", "\n", "。", "、", " ", ""],  # 日本語対応
        length_function=len
    )

    # ドキュメントごとにチャンク分割し、メタデータを保持
    all_chunks = []
    for doc in source_docs:
        doc_chunks = chunker.create_documents([doc.page_content])
        # 各チャンクにソースメタデータを付与
        for chunk in doc_chunks:
            chunk.metadata.update(doc.metadata)
        all_chunks.extend(doc_chunks)

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

    # ナレッジグラフ機能のチェック
    enable_knowledge_graph = st.session_state.get('enable_knowledge_graph', True)

    if enable_knowledge_graph:
        st.info("🕸️ ナレッジグラフ生成中...")

        # GraphDocument化
        llm = create_chat_llm(temperature=0)

        # カスタムKG抽出プロンプト（専門用語＋包括的な関係タイプ）
        kg_system_prompt = """
あなたはテキストから専門用語とその関係性を抽出する専門家です。
以下のルールに従って、テキストから専門用語ノードと関係性を抽出してください。

【ノード抽出ルール】
- 専門用語（Term）のみを抽出してください
- 専門用語の例:
  - 技術用語: API、データベース、アルゴリズム、機械学習
  - 医療用語: 疾患名、薬剤名、治療法
  - 法律用語: 法令名、契約条項、法的概念
  - ビジネス用語: KPI、ROI、サプライチェーン
  - 学術用語: 理論名、方法論、概念
  - プロセス・手順: 工程名、ステップ、フェーズ
- 一般的な名詞や動詞は無視してください（「人」「物」「する」「行う」など）
- 固有名詞は専門用語として扱ってください

【表記ゆれの統一】
- 同じ概念を指す異なる表記は同一ノードとして扱ってください
  例: 「AI」「人工知能」「Artificial Intelligence」→「AI」
  例: 「DB」「データベース」→「データベース」
  例: 「ML」「機械学習」「Machine Learning」→「機械学習」

【リレーションシップ抽出ルール】
以下のカテゴリの関係性を抽出してください：

**1. 階層・分類関係**
- IS_A: 上位下位関係（具体→抽象）
  例: 「MySQL」-[IS_A]->「データベース」
- BELONGS_TO_CATEGORY: カテゴリ所属
  例: 「決算書」-[BELONGS_TO_CATEGORY]->「財務書類」
- PART_OF: 部分構成関係
  例: 「エンジン」-[PART_OF]->「自動車」
- HAS_STEP: プロセスのステップ
  例: 「要件定義」-[HAS_STEP]->「システム開発」

**2. 属性・特性関係**
- HAS_ATTRIBUTE: 属性保持
  例: 「データベース」-[HAS_ATTRIBUTE]->「ACID特性」
- RELATED_TO: 一般的な関連性
  例: 「セキュリティ」-[RELATED_TO]->「認証」

**3. 因果・依存関係**
- AFFECTS: 影響関係
  例: 「金利」-[AFFECTS]->「住宅ローン」
- CAUSES: 原因結果
  例: 「メモリリーク」-[CAUSES]->「システムダウン」
- DEPENDS_ON: 依存関係
  例: 「デプロイ」-[DEPENDS_ON]->「テスト完了」

**4. 適用・制約関係**
- APPLIES_TO: 適用対象
  例: 「GDPR」-[APPLIES_TO]->「個人情報」
- APPLIES_WHEN: 適用条件
  例: 「緊急対応手順」-[APPLIES_WHEN]->「障害発生時」
- REQUIRES_QUALITY_GATE: 品質ゲート要求
  例: 「本番リリース」-[REQUIRES_QUALITY_GATE]->「セキュリティ監査」
- REQUIRES_APPROVAL_FROM: 承認要求
  例: 「予算執行」-[REQUIRES_APPROVAL_FROM]->「取締役会」

**5. 所有・責任関係**
- OWNED_BY: 所有者
  例: 「認証サービス」-[OWNED_BY]->「セキュリティチーム」

**6. 同義語関係**
- SAME_AS: 完全同義
  例: 「AI」-[SAME_AS]->「人工知能」
- ALIAS_OF: エイリアス・略称
  例: 「DB」-[ALIAS_OF]->「データベース」

【重要な注意事項】
- 明確な関係性のみを抽出し、推測や曖昧な関係は含めないでください
- 関係の方向性に注意してください（特にIS_A、PART_OFなど）
- テキスト中に明示されている関係を優先してください
"""

        kg_user_prompt = """
以下のテキストから、上記ルールに従って専門用語とその関係性を抽出してください。

テキスト:
{input}
"""

        kg_prompt = ChatPromptTemplate.from_messages([
            ("system", kg_system_prompt),
            ("user", kg_user_prompt)
        ])

        transformer = LLMGraphTransformer(
            llm=llm,
            prompt=kg_prompt,
            allowed_nodes=["Term"],
            allowed_relationships=[
                # 階層・分類関係
                "IS_A", "BELONGS_TO_CATEGORY", "PART_OF", "HAS_STEP",
                # 属性・特性関係
                "HAS_ATTRIBUTE", "RELATED_TO",
                # 因果・依存関係
                "AFFECTS", "CAUSES", "DEPENDS_ON",
                # 適用・制約関係
                "APPLIES_TO", "APPLIES_WHEN", "REQUIRES_QUALITY_GATE", "REQUIRES_APPROVAL_FROM",
                # 所有・責任関係
                "OWNED_BY",
                # 同義語関係
                "SAME_AS", "ALIAS_OF"
            ],
            strict_mode=True
        )
        graph_docs = transformer.convert_to_graph_documents(chunks)

        # グラフバックエンドにロード
        if st.session_state.graph_backend == "neo4j":
            graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PW,
                enhanced_schema=True  # APOCを使用
            )
            graph.add_graph_documents(graph_docs, include_source=True)
        else:  # networkx
            from networkx_graph import NetworkXGraph
            graph = NetworkXGraph(storage_path="graph.pkl", auto_save=True)
            graph.add_graph_documents(graph_docs, include_source=True)

        # CSVエッジ取り込み（source,target,label のシンプル形式）
        if csv_edges:
            if st.session_state.graph_backend == "neo4j":
                for edge in csv_edges:
                    graph.query(
                        f"""
                        MERGE (s:CSVNode {{id: $src}})
                        MERGE (t:CSVNode {{id: $tgt}})
                        MERGE (s)-[r:`{edge['label']}`]->(t)
                        """,
                        params={"src": edge["source"], "tgt": edge["target"]}
                    )
            else:
                # NetworkXGraphはCypher MERGEをサポートしないので手動追加
                for edge in csv_edges:
                    src = edge["source"]
                    tgt = edge["target"]
                    rel = edge["label"]
                    graph.add_node_manual(src, node_type="CSVNode")
                    graph.add_node_manual(tgt, node_type="CSVNode")
                    graph.add_edge_manual(src, tgt, rel_type=rel)
                if getattr(graph, "auto_save", False):
                    graph.save()

        # Documentノードを作成してChunkとリンク
        for doc in source_docs:
            doc_name = doc.metadata.get("source", "Unknown")
            # Documentノードを作成
            graph.query("""
                MERGE (d:Document {name: $doc_name})
                SET d.created = timestamp()
            """, params={"doc_name": doc_name})

        # 各ChunkをDocumentにリンク
        for chunk in chunks:
            chunk_id = chunk.metadata.get("id")
            doc_name = chunk.metadata.get("source", "Unknown")
            if chunk_id:
                graph.query("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (d:Document {name: $doc_name})
                    MERGE (c)-[:FROM_DOCUMENT]->(d)
                """, params={"chunk_id": chunk_id, "doc_name": doc_name})

        # クロスドキュメント推論: 共通する専門用語を持つドキュメント間にリレーションを作成
        cross_doc_query = """
        MATCH (d1:Document)<-[:FROM_DOCUMENT]-(c1:Chunk)-[:MENTIONS]->(term:Term)
        MATCH (d2:Document)<-[:FROM_DOCUMENT]-(c2:Chunk)-[:MENTIONS]->(term)
        WHERE d1.name <> d2.name
        WITH d1, d2, COUNT(DISTINCT term) AS common_terms
        WHERE common_terms >= 2
        MERGE (d1)-[r:SHARES_TOPICS_WITH]->(d2)
        SET r.common_term_count = common_terms
        """
        try:
            graph.query(cross_doc_query)
        except Exception as e:
            # クロスドキュメント推論が失敗しても続行
            pass

        # エンティティベクトル化（有効な場合）
        if st.session_state.get('enable_entity_vector', True):
            with st.spinner("エンティティをベクトル化中..."):
                try:
                    entity_vectorizer = EntityVectorizer(PG_CONN, embeddings)

                    # グラフからエンティティを抽出
                    entities = entity_vectorizer.extract_entities_from_graph(
                        graph,
                        graph_backend=st.session_state.graph_backend
                    )

                    # エンティティをベクトル化して保存
                    num_saved = entity_vectorizer.add_entities(entities, graph_docs)

                    if num_saved > 0:
                        st.success(f"✅ {num_saved}個のエンティティをベクトル化しました")

                except Exception as e:
                    st.warning(f"エンティティベクトル化エラー: {e}")
    else:
        st.info("⚡ ナレッジグラフをスキップし、ベクトル検索のみを使用します")
        # LLMはチェーン構築で必要
        llm = create_chat_llm(temperature=0)
        graph_docs = []  # ナレッジグラフOFFの場合は空

        # グラフオブジェクトは作成（CSVエッジ用）
        if st.session_state.graph_backend == "neo4j":
            graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PW,
                enhanced_schema=True
            )
        else:  # networkx
            from networkx_graph import NetworkXGraph
            graph = NetworkXGraph(storage_path="graph.pkl", auto_save=True)

        # CSVエッジ取り込み（ナレッジグラフOFFでもCSVは処理）
        if csv_edges:
            st.info(f"🔗 CSVから{len(csv_edges)}件のエッジを追加中...")
            if st.session_state.graph_backend == "neo4j":
                for edge in csv_edges:
                    graph.query(
                        f"""
                        MERGE (s:CSVNode {{id: $src}})
                        MERGE (t:CSVNode {{id: $tgt}})
                        MERGE (s)-[r:`{edge['label']}`]->(t)
                        """,
                        params={"src": edge["source"], "tgt": edge["target"]}
                    )
            else:
                for edge in csv_edges:
                    src = edge["source"]
                    tgt = edge["target"]
                    rel = edge["label"]
                    graph.add_node_manual(src, node_type="CSVNode")
                    graph.add_node_manual(tgt, node_type="CSVNode")
                    graph.add_edge_manual(src, tgt, rel_type=rel)
                if getattr(graph, "auto_save", False):
                    graph.save()
            st.success(f"✅ CSVから{len(csv_edges)}件のエッジを追加しました")

        # CSVエッジからのエンティティベクトル化（有効な場合）
        if csv_edges and st.session_state.get('enable_entity_vector', True):
            with st.spinner("CSVエンティティをベクトル化中..."):
                try:
                    entity_vectorizer = EntityVectorizer(PG_CONN, embeddings)

                    # グラフからエンティティを抽出
                    entities = entity_vectorizer.extract_entities_from_graph(
                        graph,
                        graph_backend=st.session_state.graph_backend
                    )

                    # エンティティをベクトル化して保存
                    num_saved = entity_vectorizer.add_entities(entities, [])

                    if num_saved > 0:
                        st.success(f"✅ {num_saved}個のエンティティをベクトル化しました")

                except Exception as e:
                    st.warning(f"エンティティベクトル化エラー: {e}")

    # 日本語トークン化（有効な場合）
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

    # PGVector保存（重複防止設定付き）
    # チャンクが0件の場合はスキップ（CSVのみの場合など）
    if not chunks:
        st.warning("チャンクが0件のためベクトルストア保存をスキップしました")
        vector_store = None
    else:
        # IDのNULLチェック
        ids = []
        for c in chunks:
            cid = c.metadata.get("id")
            if not cid:
                raise ValueError("Chunk metadata に id がありません")
            ids.append(cid)

        ensure_hnsw_index(PG_CONN)
        vector_store = PGVector.from_documents(
            chunks,
            embeddings,
            connection=PG_CONN,
            collection_name="graphrag",
            pre_delete_collection=True,  # 既存コレクション削除
            ids=ids,  # ID指定で重複防止
            use_jsonb=True,
        )

    # トークン化データをDBに反映
    if vector_store and japanese_processor and st.session_state.get('enable_japanese_search', True):
        try:
            ensure_tokenized_schema(PG_CONN)
            import psycopg
            raw_pg_conn = normalize_pg_connection_string(PG_CONN)
            with psycopg.connect(raw_pg_conn) as conn:
                with conn.cursor() as cur:
                    for chunk in chunks:
                        tokenized = chunk.metadata.get('tokenized_content')
                        if tokenized:
                            cur.execute("""
                                UPDATE langchain_pg_embedding
                                SET tokenized_content = %s
                                WHERE cmetadata->>'id' = %s
                            """, (tokenized, chunk.metadata['id']))
                conn.commit()
        except Exception as e:
            st.warning(f"トークン化データのDB保存エラー: {e}")

    # Vector Retriever構築
    # TopK値を取得（デフォルト: 5）
    retrieval_top_k = st.session_state.get('retrieval_top_k', 5)

    # vector_storeがNone（CSVのみ）の場合はretrieverもNone
    if vector_store is None:
        vector_retriever = None
    elif HAS_PARENT:
        vector_retriever = ParentDocumentRetriever(vector_store, search_kwargs={"k": retrieval_top_k})
    else:
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": retrieval_top_k})

    # エンティティ抽出関数（ハイブリッド版）
    def extract_entities_from_question(question: str) -> List[str]:
        """LLMとベクトル検索を使って質問からエンティティを抽出"""
        entities = []

        # 1. LLMによるエンティティ抽出
        extraction_prompt = f"""以下の質問文から、固有名詞や重要なエンティティ（人物、場所、物）を抽出してください。
エンティティのみをカンマ区切りで出力してください。説明は不要です。

質問: {question}

エンティティ:"""
        try:
            response = llm.invoke(extraction_prompt)
            llm_entities = [e.strip() for e in response.content.split(',') if e.strip()]
            entities.extend(llm_entities)
        except Exception:
            # フォールバック: 簡易的なキーワード抽出
            entities.extend([w for w in question.split() if len(w) > 1])

        # 2. ベクトル検索によるエンティティ抽出（有効な場合）
        if st.session_state.get('enable_entity_vector', False):
            try:
                entity_vectorizer = EntityVectorizer(PG_CONN, embeddings)

                # 質問のベクトルで類似エンティティを検索
                similarity_threshold = st.session_state.get('entity_similarity_threshold', 0.7)
                similar_entities = entity_vectorizer.search_similar_entities(
                    question,
                    k=10,
                    score_threshold=similarity_threshold
                )

                # 検索結果をログ出力
                if similar_entities:
                    print(f"[Entity Vector Search] Found {len(similar_entities)} similar entities")
                    for eid, score in similar_entities[:3]:
                        print(f"  - {eid}: {score:.3f}")

                # エンティティIDのみを追加（重複排除）
                for entity_id, score in similar_entities:
                    if entity_id not in entities:
                        entities.append(entity_id)

            except Exception as e:
                # ベクトル検索が失敗してもLLM結果を使用
                print(f"[Entity Vector Search Error] {e}")

        return entities

    def rank_relations_by_relevance(question: str, relations: list, top_k: int = 15) -> list:
        """LLMを使って関係性の質問への関連度をスコアリング"""
        if not relations:
            return []

        # 関係性リストをテキスト化
        relations_text = "\n".join([
            f"{i+1}. {r['start']} -[{r['type']}]-> {r['end']}"
            for i, r in enumerate(relations)
        ])

        ranking_prompt = f"""以下の質問に対して、各グラフ関係性の関連度を0-10でスコアリングしてください。

【質問】
{question}

【グラフ関係性】
{relations_text}

【指示】
- 各行の番号と関連度スコア（0-10）を「番号:スコア」形式で出力
- 質問に直接関連する関係性は高スコア（8-10）
- 間接的に関連する関係性は中スコア（4-7）
- 無関係な関係性は低スコア（0-3）
- 説明不要、スコアのみ出力

【出力例】
1:9
2:3
3:7

【出力】"""

        try:
            response = llm.invoke(ranking_prompt)

            # スコアをパース
            scores = {}
            for line in response.content.strip().split('\n'):
                if ':' in line:
                    try:
                        idx, score = line.split(':')
                        scores[int(idx.strip())] = float(score.strip())
                    except:
                        continue

            # スコアでソートして上位top_k件を返す
            ranked_relations = []
            for i, relation in enumerate(relations, 1):
                score = scores.get(i, 0)
                ranked_relations.append((score, relation))

            ranked_relations.sort(reverse=True, key=lambda x: x[0])
            return [rel for score, rel in ranked_relations[:top_k]]

        except Exception as e:
            # LLMリランキング失敗時は元のリストをそのまま返す
            return relations[:top_k]

    # グラフ検索関数（N-hopトラバーサル対応）
    def get_graph_context(question: str) -> list:
        """質問からエンティティを抽出し、N-hopトラバーサルでサブグラフを取得"""
        # 1. エンティティ抽出
        entities = extract_entities_from_question(question)
        if not entities:
            return []

        # 2. ホップ数を取得
        hop_count = st.session_state.get('graph_hop_count', 1)

        # 3. ホップ数に応じたクエリを実行
        if hop_count == 1:
            # 1-hop: 直接関係のみ
            query = """
            UNWIND $entities AS entity
            MATCH (n)
            WHERE n.id CONTAINS entity
            AND NOT n.id =~ '[0-9a-f]{32}'
            WITH collect(DISTINCT n) AS matched_nodes

            UNWIND matched_nodes AS start_node
            MATCH (start_node)-[r]-(connected_node)
            WHERE type(r) <> 'MENTIONS'
            AND NOT connected_node.id =~ '[0-9a-f]{32}'

            WITH r, startNode(r) AS actual_start, endNode(r) AS actual_end
            RETURN DISTINCT actual_start.id AS start, type(r) AS type, actual_end.id AS end
            LIMIT 30
            """
            top_k = 15
        elif hop_count == 2:
            # 2-hop: 可変長パス [*1..2]
            query = """
            UNWIND $entities AS entity
            MATCH (n)
            WHERE n.id CONTAINS entity
            AND NOT n.id =~ '[0-9a-f]{32}'
            WITH collect(DISTINCT n) AS matched_nodes

            UNWIND matched_nodes AS start_node
            MATCH path = (start_node)-[*1..2]-(end_node)
            WHERE ALL(r IN relationships(path) WHERE type(r) <> 'MENTIONS')
            AND ALL(node IN nodes(path) WHERE NOT node.id =~ '[0-9a-f]{32}')
            AND start_node <> end_node

            WITH relationships(path) AS rels
            UNWIND range(0, size(rels)-1) AS i
            WITH rels[i] AS r, startNode(rels[i]) AS s, endNode(rels[i]) AS e
            RETURN DISTINCT s.id AS start, type(r) AS type, e.id AS end
            LIMIT 50
            """
            top_k = 20
        else:  # hop_count == 3
            # 3-hop: 可変長パス [*1..3]
            query = """
            UNWIND $entities AS entity
            MATCH (n)
            WHERE n.id CONTAINS entity
            AND NOT n.id =~ '[0-9a-f]{32}'
            WITH collect(DISTINCT n) AS matched_nodes

            UNWIND matched_nodes AS start_node
            MATCH path = (start_node)-[*1..3]-(end_node)
            WHERE ALL(r IN relationships(path) WHERE type(r) <> 'MENTIONS')
            AND ALL(node IN nodes(path) WHERE NOT node.id =~ '[0-9a-f]{32}')
            AND start_node <> end_node

            WITH relationships(path) AS rels
            UNWIND range(0, size(rels)-1) AS i
            WITH rels[i] AS r, startNode(rels[i]) AS s, endNode(rels[i]) AS e
            RETURN DISTINCT s.id AS start, type(r) AS type, e.id AS end
            LIMIT 80
            """
            top_k = 25

        try:
            result = graph.query(query, params={"entities": entities})
            if result:
                # 4. LLMリランキングで関連度の高い関係性のみに絞る
                result = rank_relations_by_relevance(question, result, top_k=top_k)
            return result if result else []
        except Exception as e:
            # フォールバック: 単純な1-hopマッチング
            fallback_query = """
            MATCH (n)-[r]->(m)
            WHERE (
                ANY(entity IN $entities WHERE n.id CONTAINS entity OR m.id CONTAINS entity)
            )
            AND type(r) <> 'MENTIONS'
            AND NOT n.id =~ '[0-9a-f]{32}'
            AND NOT m.id =~ '[0-9a-f]{32}'
            RETURN DISTINCT n.id AS start, type(r) AS type, m.id AS end
            LIMIT 20
            """
            try:
                result = graph.query(fallback_query, params={"entities": entities})
                if result:
                    result = rank_relations_by_relevance(question, result, top_k=15)
                return result if result else []
            except Exception:
                return []

    # LCELチェイン構築（Graph-First Retrieval）
    def retriever_and_merge(question: str):
        """グラフ検索を優先し、補助的にベクトル検索を使用"""
        # 1. ナレッジグラフが有効な場合のみグラフ検索を実行
        triples = []
        enable_knowledge_graph = st.session_state.get('enable_knowledge_graph', True)

        if enable_knowledge_graph:
            triples = get_graph_context(question)

        # 2. グラフ検索結果があればそれを使用、なければベクトル検索を補助的に使用
        docs = []
        if triples:
            # グラフから関連エンティティを取得し、それに関連するドキュメントチャンクを取得
            entity_names = list(set([t.get('start') for t in triples] + [t.get('end') for t in triples]))

            # エンティティに関連するチャンクを取得（ドキュメント情報付き）
            if entity_names:
                chunk_query = """
                UNWIND $entity_names AS entity_name
                MATCH (e {id: entity_name})<-[:MENTIONS]-(chunk)
                WHERE chunk.id =~ '[0-9a-f]{32}'
                OPTIONAL MATCH (chunk)-[:FROM_DOCUMENT]->(doc:Document)
                RETURN DISTINCT chunk.id AS chunk_id, chunk.text AS text, doc.name AS source
                LIMIT 5
                """
                try:
                    chunk_results = graph.query(chunk_query, params={"entity_names": entity_names})
                    if chunk_results:
                        # グラフから取得したチャンクをドキュメントとして追加（ソース情報付き）
                        from langchain_core.documents import Document
                        docs = [Document(
                            page_content=r.get('text', ''),
                            metadata={
                                'id': r.get('chunk_id'),
                                'source': r.get('source', 'Unknown')
                            })
                            for r in chunk_results if r.get('text')]
                except Exception:
                    pass

        # 3. グラフからドキュメントが取得できない場合はベクトル検索を使用
        if not docs:
            # ハイブリッド検索を使用（有効な場合）
            if st.session_state.get('enable_japanese_search', False) and SUDACHI_AVAILABLE:
                try:
                    hybrid_retriever = HybridRetriever(PG_CONN, collection_name="graphrag")
                    query_embedding = embeddings.embed_query(question)
                    search_type = st.session_state.get('search_mode', 'hybrid')

                    # TopK値を取得
                    retrieval_top_k = st.session_state.get('retrieval_top_k', 5)

                    hybrid_results = hybrid_retriever.search(
                        query_text=question,
                        query_vector=query_embedding,
                        k=retrieval_top_k,
                        search_type=search_type
                    )

                    # LangChain Document形式に変換
                    from langchain_core.documents import Document
                    docs = [
                        Document(
                            page_content=r['text'],
                            metadata=r['metadata']
                        ) for r in hybrid_results
                    ]
                except Exception as e:
                    st.warning(f"ハイブリッド検索エラー（ベクトル検索にフォールバック）: {e}")
                    docs = vector_retriever.invoke(question)
            else:
                # 従来のベクトル検索
                docs = vector_retriever.invoke(question)

        graph_lines = [
            f"{t.get('start')} -[{t.get('type')}]→ {t.get('end')}"
            for t in triples
        ] if triples else ["(グラフデータなし)"]

        # ドキュメントコンテキストにソース情報を含める
        doc_contexts = []
        for d in docs:
            source = d.metadata.get('source', 'Unknown')
            doc_contexts.append(f"[出典: {source}]\n{d.page_content}")

        context = (
            "<GRAPH_CONTEXT>\n" + "\n".join(graph_lines) + "\n</GRAPH_CONTEXT>\n\n" +
            "<DOCUMENT_CONTEXT>\n" + "\n---\n".join(doc_contexts) + "\n</DOCUMENT_CONTEXT>"
        )
        return {
            "context": context,
            "question": question,
            "vector_sources": docs,
            "graph_sources": triples
        }

    prompt = PromptTemplate.from_template(
        """あなたはドキュメントの専門家です。\n質問: {question}\n\n{context}\n\n---\n上記情報のみを根拠に、日本語で網羅的かつ正確に回答してください。
複数のドキュメントから情報を取得した場合は、それぞれの出典を明示してください。"""
    )

    # LLM呼び出し部分
    llm_chain = (
        prompt
        | create_chat_llm(temperature=0)
        | StrOutputParser()
    )

    # ソース情報を保持する関数
    def generate_with_sources(data):
        answer = llm_chain.invoke({"question": data["question"], "context": data["context"]})
        return {
            "answer": answer,
            "vector_sources": data["vector_sources"],
            "graph_sources": data["graph_sources"]
        }

    chain = (
        RunnablePassthrough()
        | RunnableLambda(retriever_and_merge)
        | RunnableLambda(generate_with_sources)
    )

    return chain, graph

# グラフ取得関数（改善版・バックエンド共通）
def get_enhanced_graph_data(graph, limit=200):
    """グラフバックエンドから拡張グラフデータを取得（チャンクID除外、MENTIONS関係除外、ドキュメント情報付与）"""
    # NetworkXの場合は専用メソッドを使用
    if hasattr(graph, 'get_graph_data'):
        # NetworkXGraph の場合
        return graph.get_graph_data(limit=limit)

    # Neo4jの場合は既存のCypherクエリ
    query = f"""
    MATCH (n)-[r]->(m)
    WHERE type(r) <> 'MENTIONS'
    AND NOT n.id =~ '[0-9a-f]{{32}}'
    AND NOT m.id =~ '[0-9a-f]{{32}}'
    OPTIONAL MATCH (n)<-[:MENTIONS]-(chunk_n)-[:FROM_DOCUMENT]->(doc_n:Document)
    OPTIONAL MATCH (m)<-[:MENTIONS]-(chunk_m)-[:FROM_DOCUMENT]->(doc_m:Document)
    WITH n, r, m, labels(n) as source_labels, labels(m) as target_labels,
         COLLECT(DISTINCT doc_n.name) AS source_docs,
         COLLECT(DISTINCT doc_m.name) AS target_docs
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
    result = graph.query(query)
    return result

# 後方互換性のため
def get_graph_data(graph):
    """Neo4jからグラフデータを取得（シンプル版）"""
    return get_enhanced_graph_data(graph, limit=100)


def get_enhanced_subgraph_data(graph, center_nodes: List[str], hop: int = 1, limit: int = 500):
    """サブグラフデータ取得（バックエンド判定付き）"""
    backend = st.session_state.graph_backend

    if backend == "networkx":
        # NetworkXの場合は専用メソッド使用
        if hasattr(graph, 'get_subgraph_data'):
            return graph.get_subgraph_data(center_nodes, hop, limit)
        else:
            # フォールバック: get_graph_data()で全取得
            return graph.get_graph_data(limit=limit)
    elif backend == "neo4j":
        # Neo4jの場合はエンティティ検索使用
        results = graph.query(params={'entities': center_nodes})
        # 簡易変換（Neo4jのqueryメソッドの出力を想定）
        graph_data = []
        for r in results:
            graph_data.append({
                'source': r.get('start', ''),
                'source_type': 'Unknown',
                'target': r.get('end', ''),
                'target_type': 'Unknown',
                'relation': r.get('type', 'RELATED'),
                'edge_key': 0,
                'source_degree': 0,
                'target_degree': 0,
                'source_docs': [],
                'target_docs': []
            })
        return graph_data[:limit]

    return []

# ノードタイプ推論関数
def get_node_type(node_name: str, node_label: str = None) -> str:
    """ノード名やラベルからタイプを推論"""
    if node_label and node_label != 'Unknown':
        return node_label

    # 人物判定
    person_keywords = ['太郎', '姫', '爺', '婆', '王', '侍', '人', '者']
    if any(kw in node_name for kw in person_keywords):
        return 'Person'

    # 場所判定
    place_keywords = ['山', '川', '島', '村', '城', '国', '都', '里']
    if any(kw in node_name for kw in place_keywords):
        return 'Place'

    # イベント判定
    event_keywords = ['戦', '旅', '退治', '発見', '誕生', '出会']
    if any(kw in node_name for kw in event_keywords):
        return 'Event'

    # 物判定
    object_keywords = ['宝', '刀', '船', '玉', '箱', '鏡']
    if any(kw in node_name for kw in object_keywords):
        return 'Object'

    return 'Other'

# タイプごとの色を返す
def get_color_for_type(node_type: str) -> str:
    """ノードタイプに応じた色を返す"""
    color_map = {
        'Person': '#FF6B6B',      # 赤系（人物）
        'Place': '#4ECDC4',       # 青緑系（場所）
        'Event': '#95E1D3',       # 緑系（イベント）
        'Object': '#FFE66D',      # 黄色系（物）
        'Organization': '#A8E6CF', # 薄緑（組織）
        'Other': '#95A5A6',       # グレー（その他）
        'Unknown': '#7F8C8D'      # 濃いグレー（不明）
    }
    return color_map.get(node_type, '#95A5A6')

# Streamlit-Agraph可視化関数
def visualize_graph_agraph(graph_data):
    """Streamlit-Agraphでインタラクティブに可視化"""
    try:
        from streamlit_agraph import agraph, Node, Edge, Config

        # データ検証
        if not graph_data:
            st.warning("⚠️ グラフデータが空です（Agraph）")
            return None

        nodes = []
        edges = []
        node_dict = {}

        # ノード収集とタイプ判定
        for item in graph_data:
            # 必須キーの検証
            if 'source' not in item or 'target' not in item or 'relation' not in item:
                st.warning(f"⚠️ 不正なデータ形式をスキップ: {item}")
                continue

            source_type = get_node_type(item['source'], item.get('source_type'))
            target_type = get_node_type(item['target'], item.get('target_type'))

            source_degree = item.get('source_degree', 1)
            target_degree = item.get('target_degree', 1)

            if item['source'] not in node_dict:
                node_dict[item['source']] = {
                    'type': source_type,
                    'degree': source_degree
                }

            if item['target'] not in node_dict:
                node_dict[item['target']] = {
                    'type': target_type,
                    'degree': target_degree
                }

        # ノード作成（サイズを接続数に応じて控えめに調整）
        for node_id, node_info in node_dict.items():
            size = 8 + min(node_info['degree'] * 1.5, 20)  # 最小8、最大28（控えめ）
            color = get_color_for_type(node_info['type'])
            nodes.append(
                Node(
                    id=node_id,
                    label=node_id,
                    size=size,
                    color=color,
                    title=f"{node_id} ({node_info['type']}, 接続数: {node_info['degree']})"
                )
            )

        # エッジ作成
        for item in graph_data:
            if 'source' in item and 'target' in item and 'relation' in item:
                edges.append(
                    Edge(
                        source=item['source'],
                        target=item['target'],
                        label=item['relation'],
                        color="#888888"
                    )
                )

        # ノードまたはエッジが空の場合
        if not nodes or not edges:
            st.warning(f"⚠️ Agraphデータ不足: ノード{len(nodes)}個、エッジ{len(edges)}本")
            return None

        # 設定
        config = Config(
            width="100%",
            height=700,
            directed=True,
            nodeHighlightBehavior=True,
            highlightColor="#F7B731",
            collapsible=True,
            node={'labelProperty': 'label'},
            link={'labelProperty': 'label', 'renderLabel': True}
        )

        agraph(nodes=nodes, edges=edges, config=config)
        return True  # 成功時はTrueを返す

    except ImportError:
        st.info("ℹ️ streamlit-agraphがインストールされていません")
        return None
    except Exception as e:
        st.warning(f"⚠️ Agraph可視化エラー: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return None

# Pyvis強化版可視化関数
def visualize_graph_pyvis_enhanced(graph_data):
    """Pyvisで強化されたグラフを可視化"""
    try:
        from pyvis.network import Network

        # データ検証
        if not graph_data:
            st.warning("⚠️ グラフデータが空です（Pyvis）")
            return None

        net = Network(
            height="700px",
            width="100%",
            bgcolor="#1a1a1a",
            font_color="white",
            notebook=False
        )

        # 物理エンジン設定
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000,
              "springLength": 250,
              "springConstant": 0.001,
              "damping": 0.5
            },
            "minVelocity": 0.75
          },
          "nodes": {
            "font": {"size": 14, "face": "arial"},
            "borderWidth": 2,
            "borderWidthSelected": 4
          },
          "edges": {
            "color": {"inherit": "from"},
            "smooth": {"type": "continuous"},
            "font": {"size": 12, "align": "middle"}
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)

        node_dict = {}

        # ノード情報収集
        for item in graph_data:
            # 必須キーの検証
            if 'source' not in item or 'target' not in item or 'relation' not in item:
                st.warning(f"⚠️ 不正なデータ形式をスキップ: {item}")
                continue

            source_type = get_node_type(item['source'], item.get('source_type'))
            target_type = get_node_type(item['target'], item.get('target_type'))

            source_degree = item.get('source_degree', 1)
            target_degree = item.get('target_degree', 1)
            source_docs = item.get('source_docs', [])
            target_docs = item.get('target_docs', [])

            if item['source'] not in node_dict:
                node_dict[item['source']] = {
                    'type': source_type,
                    'degree': source_degree,
                    'color': get_color_for_type(source_type),
                    'docs': source_docs
                }

            if item['target'] not in node_dict:
                node_dict[item['target']] = {
                    'type': target_type,
                    'degree': target_degree,
                    'color': get_color_for_type(target_type),
                    'docs': target_docs
                }

        # ノード追加（サイズを控えめに調整）
        for node_id, node_info in node_dict.items():
            size = 12 + min(node_info['degree'] * 1, 18)  # 最小12、最大30（控えめ）
            docs_str = "<br>出典: " + ", ".join(node_info['docs']) if node_info.get('docs') else ""
            net.add_node(
                node_id,
                label=node_id,
                color=node_info['color'],
                size=size,
                title=f"<b>{node_id}</b><br>タイプ: {node_info['type']}<br>接続数: {node_info['degree']}{docs_str}",
                borderWidth=2
            )

        # エッジ追加
        for item in graph_data:
            if 'source' in item and 'target' in item and 'relation' in item:
                net.add_edge(
                    item['source'],
                    item['target'],
                    label=item['relation'],
                    title=item['relation'],
                    arrows='to',
                    color='#666666'
                )

        # ノードまたはエッジが空の場合
        if len(node_dict) == 0:
            st.warning("⚠️ Pyvisデータ不足: ノードが0個です")
            return None

        net.save_graph("graph_enhanced.html")
        with open("graph_enhanced.html", "r", encoding="utf-8") as f:
            html = f.read()
        return html

    except ImportError:
        st.info("ℹ️ pyvisがインストールされていません")
        return None
    except Exception as e:
        st.warning(f"⚠️ Pyvis可視化エラー: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return None

# 旧グラフ可視化関数（後方互換性）
def visualize_graph(graph_data):
    """pyvisでグラフを可視化（シンプル版）"""
    return visualize_graph_pyvis_enhanced(graph_data)

# 自然言語→Cypherクエリ変換関数
def natural_language_to_cypher(query: str) -> str:
    """自然言語クエリをCypherクエリに変換"""
    try:
        llm = create_chat_llm(temperature=0)

        prompt = f"""あなたはNeo4jのCypherクエリエキスパートです。
以下の自然言語をCypherクエリに変換してください。

【グラフスキーマ情報】
- ノード: プロパティは `id` (エンティティ名を格納)
- リレーションシップ: 動的（MENTIONS以外のすべての関係タイプ）
- 除外条件: チャンクノード（id =~ '[0-9a-f]{{32}}'）は除外すること
- MENTIONS関係は除外すること

【クエリ作成ルール】
1. RETURN句で必ず以下を返すこと:
   - ノード間の関係の場合: n.id AS source, type(r) AS relation, m.id AS target
   - ノードのみの場合: n.id AS node_id, labels(n) AS labels
2. チャンクノードを除外: WHERE NOT n.id =~ '[0-9a-f]{{32}}'
3. MENTIONS関係を除外: WHERE type(r) <> 'MENTIONS'
4. LIMIT句を必ず付与（デフォルト50）

自然言語クエリ: {query}

Cypherクエリ（クエリのみ出力、説明不要）:"""

        response = llm.invoke(prompt)
        cypher_query = response.content.strip()

        # コードブロックを除去（```cypher ``` で囲まれている場合）
        if cypher_query.startswith("```"):
            lines = cypher_query.split("\n")
            cypher_query = "\n".join(lines[1:-1]) if len(lines) > 2 else cypher_query

        return cypher_query

    except Exception as e:
        st.error(f"Cypherクエリ変換エラー: {e}")
        return ""

# Cypherクエリ実行&可視化関数
def execute_cypher_and_visualize(cypher_query: str, graph):
    """Cypherクエリを実行して結果を返す"""
    try:
        # 危険なクエリを検出
        dangerous_keywords = ['DELETE', 'DROP', 'CREATE', 'MERGE', 'SET', 'REMOVE', 'DETACH']
        upper_query = cypher_query.upper()

        for keyword in dangerous_keywords:
            if keyword in upper_query:
                st.error(f"⚠️ 危険なクエリが検出されました: {keyword} は使用できません")
                return None

        # クエリ実行
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

# テーブル表示関数（編集機能付き）
def display_data_tables(graph_data, graph=None, enable_edit=False):
    """ノードとエッジをテーブル形式で表示（編集機能付き）"""
    import pandas as pd

    # ノードデータの集計
    nodes_dict = {}
    for item in graph_data:
        # ソースノード
        if item['source'] not in nodes_dict:
            source_type = get_node_type(item['source'], item.get('source_type'))
            nodes_dict[item['source']] = {
                'ノードID': item['source'],
                'タイプ': source_type,
                '接続数': item.get('source_degree', 0),
                '色': get_color_for_type(source_type)
            }

        # ターゲットノード
        if item['target'] not in nodes_dict:
            target_type = get_node_type(item['target'], item.get('target_type'))
            nodes_dict[item['target']] = {
                'ノードID': item['target'],
                'タイプ': target_type,
                '接続数': item.get('target_degree', 0),
                '色': get_color_for_type(target_type)
            }

    # エッジデータの作成
    edges_list = []
    for item in graph_data:
        edges_list.append({
            '始点': item['source'],
            'リレーション': item['relation'],
            '終点': item['target'],
            'edge_key': item.get('edge_key', 0)
        })

    # ノードテーブル
    st.subheader("📍 ノード一覧")

    # 編集機能が有効な場合は編集ボタンを追加
    if enable_edit and graph:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("➕ 新規ノード追加", key="add_node_btn"):
                st.session_state.edit_mode = "add_node"

        # 編集モードの処理
        if st.session_state.get('edit_mode') == 'add_node':
            with st.expander("➕ 新規ノード追加", expanded=True):
                edit_node_dialog(graph, None)
                if st.button("閉じる"):
                    st.session_state.edit_mode = None
                    st.rerun()

    nodes_df = pd.DataFrame(list(nodes_dict.values()))
    st.dataframe(
        nodes_df.sort_values('接続数', ascending=False),
        width='stretch',
        hide_index=True
    )

    # 編集機能: ノード個別編集・削除
    if enable_edit and graph:
        st.caption("ノードを編集・削除する場合は以下から選択してください")
        selected_node = st.selectbox(
            "ノードを選択",
            options=[""] + list(nodes_dict.keys()),
            key="selected_node"
        )

        if selected_node:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✏️ 編集", key=f"edit_node_{selected_node}"):
                    st.session_state.editing_node = selected_node

            # 編集モード時は常にダイアログ表示
            if st.session_state.get('editing_node') == selected_node:
                node_info = graph.get_node_info(selected_node)
                if node_info:
                    with st.expander(f"✏️ ノード編集: {selected_node}", expanded=True):
                        edit_node_dialog(graph, node_info)
            with col2:
                if st.button("🗑️ 削除", key=f"delete_node_{selected_node}"):
                    if st.session_state.get(f'confirm_delete_node_{selected_node}'):
                        success = graph.delete_node(selected_node)
                        if success:
                            st.success(f"✅ ノード '{selected_node}' を削除しました")
                            # キャッシュクリア + 即座に再取得
                            graph_data = graph.get_graph_data(limit=200)
                            st.session_state.graph_data_cache = graph_data
                            st.rerun()
                        else:
                            st.error("削除に失敗しました")
                        st.session_state[f'confirm_delete_node_{selected_node}'] = False
                    else:
                        st.session_state[f'confirm_delete_node_{selected_node}'] = True
                        st.warning(f"⚠️ ノード '{selected_node}' を削除しますか？もう一度削除ボタンを押してください。")

    # CSVダウンロード
    csv_nodes = nodes_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 ノードをCSVでダウンロード",
        data=csv_nodes,
        file_name="nodes.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # エッジテーブル
    st.subheader("🔗 エッジ一覧")

    # 編集機能が有効な場合は追加ボタンを表示
    if enable_edit and graph:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("➕ 新規エッジ追加", key="add_edge_btn"):
                st.session_state.edit_mode = "add_edge"

        # 編集モードの処理
        if st.session_state.get('edit_mode') == 'add_edge':
            all_node_ids = list(nodes_dict.keys())
            with st.expander("➕ 新規エッジ追加", expanded=True):
                edit_edge_dialog(graph, None, all_node_ids)
                if st.button("閉じる", key="close_add_edge"):
                    st.session_state.edit_mode = None
                    st.rerun()

    edges_df = pd.DataFrame(edges_list)
    st.dataframe(
        edges_df,
        width='stretch',
        hide_index=True
    )

    # 編集機能: エッジ個別編集・削除
    if enable_edit and graph:
        st.caption("エッジを編集・削除する場合は以下から選択してください")

        # エッジ選択肢を作成
        edge_options = [""] + [f"{e['始点']} → {e['終点']} ({e['リレーション']})" for e in edges_list]
        selected_edge_str = st.selectbox(
            "エッジを選択",
            options=edge_options,
            key="selected_edge"
        )

        if selected_edge_str:
            # 選択されたエッジを解析
            selected_idx = edge_options.index(selected_edge_str) - 1
            if selected_idx >= 0:
                selected_edge_data = edges_list[selected_idx]
                source = selected_edge_data['始点']
                target = selected_edge_data['終点']
                rel_type = selected_edge_data['リレーション']
                edge_key = selected_edge_data.get('edge_key', 0)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✏️ 編集", key=f"edit_edge_{selected_idx}"):
                        st.session_state.editing_edge = selected_idx

                # 編集モード時は常にダイアログ表示
                if st.session_state.get('editing_edge') == selected_idx:
                    edge_info = graph.get_edge_info(source, target, edge_key)
                    if edge_info:
                        with st.expander(f"✏️ エッジ編集: {source} → {target}", expanded=True):
                            edit_edge_dialog(graph, edge_info)
                with col2:
                    if st.button("🗑️ 削除", key=f"delete_edge_{selected_idx}"):
                        if st.session_state.get(f'confirm_delete_edge_{selected_idx}'):
                            success = graph.delete_edge(source, target, edge_key)
                            if success:
                                st.success(f"✅ エッジ '{source} → {target}' を削除しました")
                                # キャッシュクリア + 即座に再取得
                                graph_data = graph.get_graph_data(limit=200)
                                st.session_state.graph_data_cache = graph_data
                                st.rerun()
                            else:
                                st.error("削除に失敗しました")
                            st.session_state[f'confirm_delete_edge_{selected_idx}'] = False
                        else:
                            st.session_state[f'confirm_delete_edge_{selected_idx}'] = True
                            st.warning(f"⚠️ エッジ '{source} → {target}' を削除しますか？もう一度削除ボタンを押してください。")

    # CSVダウンロード
    csv_edges = edges_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 エッジをCSVでダウンロード",
        data=csv_edges,
        file_name="edges.csv",
        mime="text/csv"
    )

    # 統計情報
    st.markdown("---")
    st.subheader("📊 統計情報")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("総ノード数", len(nodes_dict))
    with col2:
        st.metric("総エッジ数", len(edges_list))
    with col3:
        avg_degree = sum(n['接続数'] for n in nodes_dict.values()) / len(nodes_dict) if nodes_dict else 0
        st.metric("平均接続数", f"{avg_degree:.1f}")


# グラフ編集用ヘルパー関数
def edit_node_dialog(graph, node_info=None):
    """ノード編集ダイアログ"""
    st.subheader("✏️ ノード編集" if node_info else "➕ 新規ノード追加")

    with st.form("node_form"):
        if node_info:
            node_id = st.text_input("ノードID", value=node_info['id'], disabled=True)
            node_type = st.text_input("タイプ", value=node_info.get('type', 'Unknown'))
            properties_str = st.text_area(
                "プロパティ (JSON形式)",
                value=json.dumps(node_info.get('properties', {}), ensure_ascii=False, indent=2)
            )
        else:
            node_id = st.text_input("ノードID", placeholder="例: 桃太郎")
            node_type = st.text_input("タイプ", value="Unknown", placeholder="例: Person")
            properties_str = st.text_area("プロパティ (JSON形式)", value="{}")

        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("💾 保存", type="primary")
        with col2:
            cancel = st.form_submit_button("❌ キャンセル")

        if submit:
            try:
                properties = json.loads(properties_str) if properties_str.strip() else {}

                if node_info:
                    # 更新
                    success = graph.update_node(node_id, node_type, properties)
                    if success:
                        st.success(f"✅ ノード '{node_id}' を更新しました")
                        # 編集状態をクリア
                        st.session_state.editing_node = None
                        # キャッシュクリア + 即座に再取得
                        graph_data = graph.get_graph_data(limit=200)
                        st.session_state.graph_data_cache = graph_data
                        st.rerun()
                    else:
                        st.error("更新に失敗しました")
                else:
                    # 新規追加
                    if not node_id:
                        st.error("ノードIDを入力してください")
                    else:
                        success = graph.add_node_manual(node_id, node_type, properties)
                        if success:
                            st.success(f"✅ ノード '{node_id}' を追加しました")
                            # 編集状態をクリア（新規追加の場合は該当なし）
                            st.session_state.edit_mode = None
                            # キャッシュクリア + 即座に再取得
                            graph_data = graph.get_graph_data(limit=200)
                            st.session_state.graph_data_cache = graph_data
                            st.rerun()
                        else:
                            st.error("追加に失敗しました")
            except json.JSONDecodeError:
                st.error("プロパティのJSON形式が不正です")

        if cancel:
            # 編集状態をクリア
            st.session_state.editing_node = None
            st.session_state.edit_mode = None
            st.rerun()


def edit_edge_dialog(graph, edge_info=None, all_nodes=None):
    """エッジ編集ダイアログ"""
    st.subheader("✏️ エッジ編集" if edge_info else "➕ 新規エッジ追加")

    if all_nodes is None:
        all_nodes = []

    with st.form("edge_form"):
        if edge_info:
            source = st.text_input("始点ノード", value=edge_info['source'], disabled=True)
            target = st.text_input("終点ノード", value=edge_info['target'], disabled=True)
            edge_key = edge_info.get('edge_key', 0)
            rel_type = st.text_input("リレーションタイプ", value=edge_info.get('type', 'RELATED'))
            properties_str = st.text_area(
                "プロパティ (JSON形式)",
                value=json.dumps(edge_info.get('properties', {}), ensure_ascii=False, indent=2)
            )
        else:
            if all_nodes:
                source = st.selectbox("始点ノード", options=all_nodes)
                target = st.selectbox("終点ノード", options=all_nodes)
            else:
                source = st.text_input("始点ノード", placeholder="例: 桃太郎")
                target = st.text_input("終点ノード", placeholder="例: 鬼")
            edge_key = 0
            rel_type = st.text_input("リレーションタイプ", value="RELATED", placeholder="例: 倒した")
            properties_str = st.text_area("プロパティ (JSON形式)", value="{}")

        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("💾 保存", type="primary")
        with col2:
            cancel = st.form_submit_button("❌ キャンセル")

        if submit:
            try:
                properties = json.loads(properties_str) if properties_str.strip() else {}

                if edge_info:
                    # 更新
                    success = graph.update_edge(source, target, edge_key, rel_type, properties)
                    if success:
                        st.success(f"✅ エッジ '{source} -> {target}' を更新しました")
                        # 編集状態をクリア
                        st.session_state.editing_edge = None
                        # キャッシュクリア + 即座に再取得
                        graph_data = graph.get_graph_data(limit=200)
                        st.session_state.graph_data_cache = graph_data
                        st.rerun()
                    else:
                        st.error("更新に失敗しました")
                else:
                    # 新規追加
                    if not source or not target:
                        st.error("始点と終点を指定してください")
                    else:
                        edge_key = graph.add_edge_manual(source, target, rel_type, properties)
                        if edge_key is not None:
                            st.success(f"✅ エッジ '{source} -> {target}' を追加しました")
                            # 編集状態をクリア（新規追加の場合は該当なし）
                            st.session_state.edit_mode = None
                            # キャッシュクリア + 即座に再取得
                            graph_data = graph.get_graph_data(limit=200)
                            st.session_state.graph_data_cache = graph_data
                            st.rerun()
                        else:
                            st.error("追加に失敗しました")
            except json.JSONDecodeError:
                st.error("プロパティのJSON形式が不正です")

        if cancel:
            # 編集状態をクリア
            st.session_state.editing_edge = None
            st.session_state.edit_mode = None
            st.rerun()


def confirm_delete_dialog(item_type, item_name, callback):
    """削除確認ダイアログ"""
    st.warning(f"⚠️ {item_type} '{item_name}' を削除しますか？")
    st.caption("この操作は取り消せません。")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 削除する", type="primary"):
            if callback():
                st.success(f"✅ {item_type} '{item_name}' を削除しました")
                st.session_state.graph_data_cache = None  # キャッシュクリア
                st.rerun()
            else:
                st.error("削除に失敗しました")
    with col2:
        if st.button("❌ キャンセル"):
            st.rerun()

# メインUI
st.header("📁 ドキュメントアップロード")

# 既存グラフのチェック（初回のみ）
if not st.session_state.existing_graph_loaded and not st.session_state.initialized:
    try:
        # グラフバックエンド接続
        if st.session_state.graph_backend == "neo4j":
            temp_graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PW,
                enhanced_schema=False  # APOC不要
            )
        else:  # networkx
            from networkx_graph import NetworkXGraph
            temp_graph = NetworkXGraph(storage_path="graph.pkl", auto_save=True)

        graph_info = check_existing_graph(temp_graph, st.session_state.graph_backend)

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
    except Exception as e:
        # Neo4j接続エラーは無視（後続の処理で対応）
        pass

uploaded_files = st.file_uploader(
    "PDF/テキストファイルをアップロード",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="複数ファイルをアップロード可能です"
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

# ナレッジグラフ構築ボタン（ドキュメントまたはCSVがあれば表示）
if has_docs or has_csv:
    if st.button("🚀 ナレッジグラフを構築", type="primary"):
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
                # 新しいグラフに合わせてキャッシュをクリア
                st.session_state.graph_data_cache = None
                if 'all_node_list' in st.session_state:
                    st.session_state.all_node_list = None
                st.success("✅ ナレッジグラフ構築完了!")
            except Exception as e:
                st.error(f"構築エラー: {e}")
                import traceback
                st.code(traceback.format_exc())

st.markdown("---")

# タブ形式UI
tab1, tab2 = st.tabs(["💬 質問応答", "🕸️ グラフ探索"])

with tab1:
    st.header("💬 質問応答")

    # 質問入力
    if st.session_state.initialized:
        question = st.text_area("質問を入力してください:", height=150, key="question_input")

        if st.button("🔍 質問する", type="primary"):
            if question:
                with st.spinner("回答生成中..."):
                    try:
                        result = st.session_state.chain.invoke(question)

                        # 回答表示
                        st.markdown("### 📝 回答")
                        st.markdown(result["answer"])

                        # 引用元: Vector RAG
                        with st.expander("📚 参照ドキュメント (Vector RAG)", expanded=False):
                            vector_sources = result.get("vector_sources", [])
                            if vector_sources:
                                for i, doc in enumerate(vector_sources, 1):
                                    st.markdown(f"**チャンク {i}:**")
                                    st.text(doc.page_content)
                                    if i < len(vector_sources):
                                        st.divider()
                            else:
                                st.info("ベクトル検索結果なし")

                        # 引用元: Graph RAG
                        with st.expander("🕸️ ナレッジグラフ (Graph RAG)", expanded=False):
                            graph_sources = result.get("graph_sources", [])
                            if graph_sources:
                                for triple in graph_sources:
                                    st.markdown(f"- `{triple.get('start')}` -[{triple.get('type')}]→ `{triple.get('end')}`")
                            else:
                                st.info("グラフ検索結果なし")

                    except Exception as e:
                        st.error(f"エラー: {e}")
            else:
                st.warning("質問を入力してください")
    else:
        st.info("まずRAGシステムを初期化してください")

with tab2:
    st.header("🕸️ グラフ探索")

    if st.session_state.initialized:
        # 表示モード選択
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
                # 可視化範囲選択
                viz_scope = st.radio(
                    "📊 可視化範囲",
                    ["全体表示", "部分表示（検索）"],
                    horizontal=True,
                    help="大規模グラフの場合は部分表示を推奨します"
                )

                if viz_scope == "部分表示（検索）":
                    # 部分可視化モード
                    st.markdown("### 🔍 ノード検索")

                    # セッションステート初期化
                    if 'center_nodes' not in st.session_state:
                        st.session_state.center_nodes = []

                    # 全ノードリスト取得（初回のみ）
                    if 'all_node_list' not in st.session_state:
                        if st.session_state.graph_data_cache:
                            graph_data = st.session_state.graph_data_cache
                            all_nodes = list(set([item['source'] for item in graph_data] + [item['target'] for item in graph_data]))
                            st.session_state.all_node_list = sorted(all_nodes)
                        else:
                            # キャッシュがない場合は一度取得
                            with st.spinner("ノードリスト取得中..."):
                                try:
                                    graph_data = get_enhanced_graph_data(st.session_state.graph, limit=max_nodes)
                                    st.session_state.graph_data_cache = graph_data
                                    all_nodes = list(set([item['source'] for item in graph_data] + [item['target'] for item in graph_data]))
                                    st.session_state.all_node_list = sorted(all_nodes)
                                except Exception as e:
                                    st.error(f"エラー: {e}")
                                    st.session_state.all_node_list = []

                    if st.session_state.all_node_list:
                        # 検索ボックス
                        search_query = st.text_input(
                            "🔍 ノード検索（部分一致）",
                            placeholder="例: 桃太郎",
                            help="検索したノードとその周辺を表示します"
                        )

                        if search_query:
                            # 検索実行
                            matched_nodes = [n for n in st.session_state.all_node_list
                                            if search_query.lower() in n.lower()]

                            st.caption(f"🔍 検索結果: {len(matched_nodes)}件")

                            if matched_nodes:
                                # selectboxで1つ選択
                                selected_node = st.selectbox(
                                    "ノードを選択",
                                    options=[""] + matched_nodes,
                                    index=0,
                                    help="リストから1つ選んで追加してください"
                                )

                                # ボタン配置
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    if selected_node and st.button("➕ 中心ノードに追加"):
                                        if selected_node not in st.session_state.center_nodes:
                                            st.session_state.center_nodes.append(selected_node)
                                            st.rerun()
                                        else:
                                            st.warning("既に追加されています")

                                with col2:
                                    if st.session_state.center_nodes and st.button("🗑️ リセット"):
                                        st.session_state.center_nodes = []
                                        st.rerun()
                            else:
                                st.warning(f"「{search_query}」に一致するノードが見つかりませんでした")
                        else:
                            st.info("💡 ノード名を入力して検索してください")

                        # 選択済み中心ノード表示
                        if st.session_state.center_nodes:
                            st.markdown("---")
                            st.write("**中心ノード:**", ", ".join(st.session_state.center_nodes))

                            # Hop数選択
                            hop_distance = st.slider(
                                "周辺表示範囲（Hop数）",
                                min_value=1,
                                max_value=3,
                                value=2,
                                help="選択ノードから何Hop先まで表示するか"
                            )

                            # サブグラフ取得＆表示
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
                                            # 統計情報
                                            unique_nodes = set()
                                            for item in subgraph_data:
                                                unique_nodes.add(item['source'])
                                                unique_nodes.add(item['target'])

                                            st.success(f"✅ サブグラフ取得完了")
                                            st.info(f"📊 表示: ノード {len(unique_nodes)}個 / エッジ {len(subgraph_data)}本")

                                            # 可視化
                                            if "Agraph" in viz_engine:
                                                result = visualize_graph_agraph(subgraph_data)
                                                if not result:
                                                    st.warning("⚠️ Streamlit-Agraphが利用できません。Pyvisにフォールバックします。")
                                                    html = visualize_graph_pyvis_enhanced(subgraph_data)
                                                    if html:
                                                        st.components.v1.html(html, height=700)
                                            else:
                                                html = visualize_graph_pyvis_enhanced(subgraph_data)
                                                if html:
                                                    st.components.v1.html(html, height=700)
                                                else:
                                                    st.warning("可視化ライブラリが利用できません。")
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
                    # 全体表示モード（既存処理）
                    # 初回グラフ読み込み
                    if st.session_state.graph_data_cache is None:
                        if st.button("📊 グラフを読み込む", type="primary"):
                            with st.spinner("グラフデータ取得中..."):
                                try:
                                    graph_data = get_enhanced_graph_data(st.session_state.graph, limit=max_nodes)
                                    st.session_state.graph_data_cache = graph_data
                                    st.success(f"✅ {len(graph_data)}件のエッジを読み込みました")
                                except Exception as e:
                                    st.error(f"エラー: {e}")

                    # グラフデータがある場合はリアルタイム表示
                    if st.session_state.graph_data_cache:
                        try:
                            graph_data = st.session_state.graph_data_cache

                            if not graph_data:
                                st.warning("グラフデータがありません")
                            else:
                                # 統計情報表示
                                unique_nodes = set()
                                for item in graph_data:
                                    unique_nodes.add(item['source'])
                                    unique_nodes.add(item['target'])

                                st.info(f"📊 表示中: ノード {len(unique_nodes)}個 / エッジ {len(graph_data)}本")

                                # 可視化エンジン選択
                                if "Agraph" in viz_engine:
                                    # Streamlit-Agraph可視化
                                    result = visualize_graph_agraph(graph_data)
                                    if not result:
                                        # Agraphが失敗した場合のみPyvisにフォールバック
                                        st.warning("⚠️ Streamlit-Agraphが利用できません。Pyvisにフォールバックします。")
                                        html = visualize_graph_pyvis_enhanced(graph_data)
                                        if html:
                                            st.components.v1.html(html, height=700)
                                else:
                                    # Pyvis可視化
                                    html = visualize_graph_pyvis_enhanced(graph_data)
                                    if html:
                                        st.components.v1.html(html, height=700)
                                    else:
                                        st.warning("可視化ライブラリが利用できません。")

                            # グラフをリセットするボタン
                            if st.button("🔄 グラフを再読み込み"):
                                st.session_state.graph_data_cache = None
                                st.session_state.all_node_list = None
                                st.rerun()

                        except Exception as e:
                            st.error(f"エラー: {e}")
                            import traceback
                            st.code(traceback.format_exc())

        # モード2: データテーブル
        elif display_mode == "📊 データテーブル":
            # グラフデータがない場合は読み込みボタン
            if st.session_state.graph_data_cache is None:
                if st.button("📊 データを読み込む", type="primary", key="load_data_table"):
                    with st.spinner("データ取得中..."):
                        try:
                            graph_data = get_enhanced_graph_data(st.session_state.graph, limit=max_nodes)
                            st.session_state.graph_data_cache = graph_data
                            st.success(f"✅ {len(graph_data)}件のエッジを読み込みました")
                        except Exception as e:
                            st.error(f"エラー: {e}")

            # データがある場合は表示
            if st.session_state.graph_data_cache:
                try:
                    graph_data = st.session_state.graph_data_cache

                    if graph_data:
                        # NetworkXバックエンドの場合のみ編集機能を有効化
                        enable_edit = (st.session_state.graph_backend == "networkx")
                        display_data_tables(
                            graph_data,
                            graph=st.session_state.graph if enable_edit else None,
                            enable_edit=enable_edit
                        )
                    else:
                        st.warning("グラフデータがありません")

                except Exception as e:
                    st.error(f"エラー: {e}")

        # モード3: Cypherクエリ検索
        elif display_mode == "🔍 Cypherクエリ検索":
            st.markdown("### 自然言語でグラフを検索")
            st.info("例: 「桃太郎に関するグラフを見たい」「おじいさんと関係のあるエンティティを表示」")

            # クエリテンプレート選択（オプション）
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
                    entity_name = st.text_input("エンティティ名を入力:", placeholder="例: 桃太郎")
                    if entity_name:
                        nl_query = f"{entity_name}に関連するすべての関係を表示"
                    else:
                        nl_query = ""
                elif template == "最も接続数が多いノードTop10を表示":
                    nl_query = "最も接続数が多いノードTop10を表示"
                elif template == "すべてのリレーションシップタイプを表示":
                    nl_query = "すべてのリレーションシップタイプとその数を表示"
                else:
                    nl_query = ""

            # 自然言語クエリ入力
            user_query = st.text_area(
                "自然言語クエリ:",
                value=nl_query,
                height=100,
                placeholder="例: 桃太郎に関するグラフを見たい"
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                convert_button = st.button("🔄 Cypherに変換", type="primary")

            # Cypherクエリ生成
            if "generated_cypher" not in st.session_state:
                st.session_state.generated_cypher = ""

            if convert_button and user_query:
                with st.spinner("Cypherクエリを生成中..."):
                    cypher_query = natural_language_to_cypher(user_query)
                    st.session_state.generated_cypher = cypher_query

            # 生成されたCypherクエリ表示（編集可能）
            if st.session_state.generated_cypher:
                st.markdown("### 📝 生成されたCypherクエリ")
                edited_cypher = st.text_area(
                    "Cypherクエリ（編集可能）:",
                    value=st.session_state.generated_cypher,
                    height=150,
                    key="cypher_editor"
                )

                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    execute_button = st.button("▶️ 実行", type="primary")
                with col2:
                    clear_button = st.button("🗑️ クリア")

                if clear_button:
                    st.session_state.generated_cypher = ""
                    st.rerun()

                # クエリ実行
                if execute_button and edited_cypher:
                    with st.spinner("クエリ実行中..."):
                        result = execute_cypher_and_visualize(edited_cypher, st.session_state.graph)

                        if result:
                            st.success(f"✅ {len(result)}件の結果を取得しました")

                            # 結果をテーブル表示
                            st.markdown("### 📊 クエリ結果")
                            import pandas as pd
                            df = pd.DataFrame(result)
                            st.dataframe(df, width='stretch')

                            # 可視化（source, relation, targetがある場合）
                            if len(result) > 0 and 'source' in result[0] and 'target' in result[0] and 'relation' in result[0]:
                                st.markdown("### 🕸️ グラフ可視化")

                                viz_choice = st.radio(
                                    "可視化エンジン",
                                    ["Pyvis", "Streamlit-Agraph"],
                                    horizontal=True,
                                    key="cypher_viz_engine"
                                )

                                if "Pyvis" in viz_choice:
                                    html = visualize_graph_pyvis_enhanced(result)
                                    if html:
                                        st.components.v1.html(html, height=700)
                                else:
                                    viz_result = visualize_graph_agraph(result)
                                    if not viz_result:
                                        st.warning("⚠️ Agraphで表示できません。Pyvisにフォールバック")
                                        html = visualize_graph_pyvis_enhanced(result)
                                        if html:
                                            st.components.v1.html(html, height=700)

    else:
        st.info("まずRAGシステムを初期化してください")

# フッター
st.markdown("---")
st.markdown("**Graph-RAG Demo** | Powered by LangChain, Neo4j & PGVector")
