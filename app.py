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

# LangChain imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import PyPDFLoader, TextLoader
try:
    from langchain_community.graphs.graph_document import GraphDocument
except ImportError:
    from langchain_community.graphs import GraphDocument
from langchain_community.vectorstores.pgvector import PGVector

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

from langchain_core.prompts import PromptTemplate
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

st.title("🔗 Graph-RAG with Neo4j & PGVector")

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

    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, NEO4J_URI, NEO4J_USER, NEO4J_PW, PG_CONN]):
        st.error("環境変数が不足しています。.envファイルを確認してください。")
        st.stop()

    st.success("✅ 環境変数読み込み完了")

    st.markdown("---")
    st.markdown("### 📊 グラフ可視化設定")

    viz_engine = st.radio(
        "可視化エンジン",
        ["Streamlit-Agraph (推奨)", "Pyvis (詳細)"],
        index=0,
        help="Agraphは軽量でインタラクティブ、Pyvisはより詳細な設定が可能"
    )

    show_graph = st.checkbox("ナレッジグラフを表示", value=True)

    if show_graph:
        max_nodes = st.slider("最大表示ノード数", 50, 500, 200, 50)

        st.markdown("**ノードタイプフィルター**")
        filter_person = st.checkbox("👤 人物 (Person)", value=True)
        filter_place = st.checkbox("🏞️ 場所 (Place)", value=True)
        filter_event = st.checkbox("⚡ イベント (Event)", value=True)
        filter_object = st.checkbox("📦 物 (Object)", value=True)
        filter_other = st.checkbox("❓ その他 (Other)", value=True)

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

# Neo4j既存データチェック関数
def check_existing_graph(graph) -> dict:
    """Neo4jに既存のグラフデータがあるかチェック"""
    try:
        query = """
        MATCH (n)
        RETURN count(n) AS node_count
        """
        result = graph.query(query)
        node_count = result[0]['node_count'] if result else 0

        if node_count > 0:
            # リレーションシップもカウント
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
        return {'exists': False, 'node_count': 0, 'rel_count': 0}
    except Exception as e:
        st.error(f"Neo4j接続エラー: {e}")
        return {'exists': False, 'node_count': 0, 'rel_count': 0}

# 既存グラフからシステムを復元
def restore_from_existing_graph():
    """Neo4jとPGVectorから既存データを使ってシステムを復元"""
    try:
        # Neo4j接続
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PW)

        # PGVector接続
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
        vector_store = PGVector(
            connection_string=PG_CONN,
            embedding_function=embeddings
        )

        # Vector Retriever構築
        if HAS_PARENT:
            vector_retriever = ParentDocumentRetriever(vector_store, search_kwargs={"k": 4})
        else:
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        # グラフ検索関数
        def get_graph_context(question: str) -> list:
            query = """
            MATCH (n)-[r]->(m)
            RETURN n.id AS start, type(r) AS type, m.id AS end
            LIMIT 10
            """
            try:
                result = graph.query(query)
                return result if result else []
            except Exception:
                return []

        # チェイン構築
        def retriever_and_merge(question: str):
            docs = vector_retriever.invoke(question)
            triples = get_graph_context(question)

            graph_lines = [
                f"{t.get('start')} -[{t.get('type')}]→ {t.get('end')}"
                for t in triples
            ] if triples else ["(グラフデータなし)"]

            context = (
                "<GRAPH_CONTEXT>\n" + "\n".join(graph_lines) + "\n</GRAPH_CONTEXT>\n\n" +
                "<DOCUMENT_CONTEXT>\n" + "\n---\n".join(d.page_content for d in docs) + "\n</DOCUMENT_CONTEXT>"
            )
            return {"context": context, "question": question}

        prompt = PromptTemplate.from_template(
            """あなたはドキュメントの専門家です。\n質問: {question}\n\n{context}\n\n---\n上記情報のみを根拠に、日本語で網羅的かつ正確に回答してください。"""
        )

        chain = (
            RunnablePassthrough()
            | RunnableLambda(retriever_and_merge)
            | prompt
            | AzureChatOpenAI(
                azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
                openai_api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                temperature=0
            )
            | StrOutputParser()
        )

        return chain, graph

    except Exception as e:
        raise Exception(f"システム復元エラー: {e}")

# ドキュメント読み込み関数
def load_documents(uploaded_files) -> str:
    """アップロードされたファイルからテキストを抽出"""
    all_text = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                all_text.append("\n".join([doc.page_content for doc in docs]))
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_path, encoding='utf-8')
                docs = loader.load()
                all_text.append("\n".join([doc.page_content for doc in docs]))
            else:
                # その他のテキストファイル
                text = uploaded_file.getvalue().decode('utf-8')
                all_text.append(text)
        finally:
            os.unlink(tmp_path)

    return "\n\n".join(all_text)

# 初期化関数
def build_rag_system(text_content: str):
    """RAGシステムの構築"""

    # チャンク分割
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY
    )
    chunker = SemanticChunker(embeddings, buffer_size=50)
    chunks = chunker.create_documents([text_content])

    # GraphDocument化
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        temperature=0
    )
    transformer = LLMGraphTransformer(llm=llm)
    graph_docs = transformer.convert_to_graph_documents(chunks)

    # Neo4jロード
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PW)
    graph.add_graph_documents(graph_docs, include_source=True)

    # PGVector保存
    vector_store = PGVector.from_documents(chunks, embeddings, connection_string=PG_CONN)

    # Vector Retriever構築
    if HAS_PARENT:
        vector_retriever = ParentDocumentRetriever(vector_store, search_kwargs={"k": 4})
    else:
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # グラフ検索関数（Cypher直接実行）
    def get_graph_context(question: str) -> list:
        """Neo4jからグラフコンテキストを取得"""
        query = """
        MATCH (n)-[r]->(m)
        RETURN n.id AS start, type(r) AS type, m.id AS end
        LIMIT 10
        """
        try:
            result = graph.query(query)
            return result if result else []
        except Exception:
            return []

    # LCELチェイン構築
    def retriever_and_merge(question: str):
        """ベクトル検索とグラフ検索を実行してコンテキストをマージ"""
        # ベクトル検索
        docs = vector_retriever.invoke(question)

        # グラフ検索
        triples = get_graph_context(question)

        graph_lines = [
            f"{t.get('start')} -[{t.get('type')}]→ {t.get('end')}"
            for t in triples
        ] if triples else ["(グラフデータなし)"]

        context = (
            "<GRAPH_CONTEXT>\n" + "\n".join(graph_lines) + "\n</GRAPH_CONTEXT>\n\n" +
            "<DOCUMENT_CONTEXT>\n" + "\n---\n".join(d.page_content for d in docs) + "\n</DOCUMENT_CONTEXT>"
        )
        return {"context": context, "question": question}

    prompt = PromptTemplate.from_template(
        """あなたはドキュメントの専門家です。\n質問: {question}\n\n{context}\n\n---\n上記情報のみを根拠に、日本語で網羅的かつ正確に回答してください。"""
    )

    chain = (
        RunnablePassthrough()
        | RunnableLambda(retriever_and_merge)
        | prompt
        | AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            temperature=0
        )
        | StrOutputParser()
    )

    return chain, graph

# グラフ取得関数（改善版）
def get_enhanced_graph_data(graph, limit=200):
    """Neo4jから拡張グラフデータを取得（ノードタイプ、接続数含む）"""
    query = f"""
    MATCH (n)-[r]->(m)
    WITH n, r, m, labels(n) as source_labels, labels(m) as target_labels
    RETURN
      n.id AS source,
      CASE WHEN size(source_labels) > 0 THEN source_labels[0] ELSE 'Unknown' END AS source_type,
      type(r) AS relation,
      m.id AS target,
      CASE WHEN size(target_labels) > 0 THEN target_labels[0] ELSE 'Unknown' END AS target_type,
      COUNT {{ (n)--() }} AS source_degree,
      COUNT {{ (m)--() }} AS target_degree
    LIMIT {limit}
    """
    result = graph.query(query)
    return result

# 後方互換性のため
def get_graph_data(graph):
    """Neo4jからグラフデータを取得（シンプル版）"""
    return get_enhanced_graph_data(graph, limit=100)

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

        nodes = []
        edges = []
        node_dict = {}

        # ノード収集とタイプ判定
        for item in graph_data:
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

        # ノード作成（サイズを接続数に応じて調整）
        for node_id, node_info in node_dict.items():
            size = 10 + min(node_info['degree'] * 3, 50)  # 最小10、最大60
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
            edges.append(
                Edge(
                    source=item['source'],
                    target=item['target'],
                    label=item['relation'],
                    color="#888888"
                )
            )

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

        return agraph(nodes=nodes, edges=edges, config=config)

    except ImportError:
        return None

# Pyvis強化版可視化関数
def visualize_graph_pyvis_enhanced(graph_data):
    """Pyvisで強化されたグラフを可視化"""
    try:
        from pyvis.network import Network

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
            source_type = get_node_type(item['source'], item.get('source_type'))
            target_type = get_node_type(item['target'], item.get('target_type'))

            source_degree = item.get('source_degree', 1)
            target_degree = item.get('target_degree', 1)

            if item['source'] not in node_dict:
                node_dict[item['source']] = {
                    'type': source_type,
                    'degree': source_degree,
                    'color': get_color_for_type(source_type)
                }

            if item['target'] not in node_dict:
                node_dict[item['target']] = {
                    'type': target_type,
                    'degree': target_degree,
                    'color': get_color_for_type(target_type)
                }

        # ノード追加
        for node_id, node_info in node_dict.items():
            size = 15 + min(node_info['degree'] * 2, 40)
            net.add_node(
                node_id,
                label=node_id,
                color=node_info['color'],
                size=size,
                title=f"<b>{node_id}</b><br>タイプ: {node_info['type']}<br>接続数: {node_info['degree']}",
                borderWidth=2
            )

        # エッジ追加
        for item in graph_data:
            net.add_edge(
                item['source'],
                item['target'],
                label=item['relation'],
                title=item['relation'],
                arrows='to',
                color='#666666'
            )

        net.save_graph("graph_enhanced.html")
        with open("graph_enhanced.html", "r", encoding="utf-8") as f:
            html = f.read()
        return html

    except ImportError:
        return None

# 旧グラフ可視化関数（後方互換性）
def visualize_graph(graph_data):
    """pyvisでグラフを可視化（シンプル版）"""
    return visualize_graph_pyvis_enhanced(graph_data)

# メインUI
st.header("📁 ドキュメントアップロード")

# 既存グラフのチェック（初回のみ）
if not st.session_state.existing_graph_loaded and not st.session_state.initialized:
    try:
        temp_graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PW)
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
    except Exception as e:
        # Neo4j接続エラーは無視（後続の処理で対応）
        pass

uploaded_files = st.file_uploader(
    "PDF/テキストファイルをアップロード",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="複数ファイルをアップロード可能です"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} ファイルがアップロードされました")

    # ファイル一覧表示
    with st.expander("📄 アップロード済みファイル"):
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size} bytes)")

    # ナレッジグラフ構築ボタン
    if st.button("🚀 ナレッジグラフを構築", type="primary"):
        with st.spinner("ドキュメント読み込み中..."):
            try:
                text_content = load_documents(uploaded_files)
                st.info(f"テキスト長: {len(text_content)} 文字")
            except Exception as e:
                st.error(f"ファイル読み込みエラー: {e}")
                st.stop()

        with st.spinner("ナレッジグラフ構築中... (数分かかる場合があります)"):
            try:
                st.session_state.chain, st.session_state.graph = build_rag_system(text_content)
                st.session_state.initialized = True
                st.session_state.uploaded_files = [f.name for f in uploaded_files]
                st.success("✅ ナレッジグラフ構築完了!")
            except Exception as e:
                st.error(f"構築エラー: {e}")
                import traceback
                st.code(traceback.format_exc())

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("💬 質問入力")

    # 質問入力
    if st.session_state.initialized:
        question = st.text_area("質問を入力してください:", height=100)

        if st.button("🔍 質問する"):
            if question:
                with st.spinner("回答生成中..."):
                    try:
                        answer = st.session_state.chain.invoke(question)
                        st.markdown("### 📝 回答")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"エラー: {e}")
            else:
                st.warning("質問を入力してください")
    else:
        st.info("まずRAGシステムを初期化してください")

with col2:
    st.header("🕸️ ナレッジグラフ")

    if st.session_state.initialized and show_graph:
        if st.button("📊 グラフを表示"):
            with st.spinner("グラフ取得中..."):
                try:
                    # 拡張グラフデータ取得
                    graph_data = get_enhanced_graph_data(st.session_state.graph, limit=max_nodes)

                    if graph_data:
                        # フィルタリング
                        filtered_data = []
                        for item in graph_data:
                            source_type = get_node_type(item['source'], item.get('source_type'))
                            target_type = get_node_type(item['target'], item.get('target_type'))

                            # フィルター適用
                            type_filters = {
                                'Person': filter_person,
                                'Place': filter_place,
                                'Event': filter_event,
                                'Object': filter_object,
                                'Other': filter_other
                            }

                            if type_filters.get(source_type, True) and type_filters.get(target_type, True):
                                filtered_data.append(item)

                        if not filtered_data:
                            st.warning("フィルター条件に一致するデータがありません")
                        else:
                            # 統計情報表示
                            unique_nodes = set()
                            for item in filtered_data:
                                unique_nodes.add(item['source'])
                                unique_nodes.add(item['target'])

                            st.markdown(f"**統計情報:** ノード {len(unique_nodes)}個 / エッジ {len(filtered_data)}本")

                            # 可視化エンジン選択
                            if "Agraph" in viz_engine:
                                # Streamlit-Agraph可視化
                                result = visualize_graph_agraph(filtered_data)
                                if result is None:
                                    st.warning("Streamlit-Agraphが利用できません。Pyvisにフォールバックします。")
                                    html = visualize_graph_pyvis_enhanced(filtered_data)
                                    if html:
                                        st.components.v1.html(html, height=700)
                            else:
                                # Pyvis可視化
                                html = visualize_graph_pyvis_enhanced(filtered_data)
                                if html:
                                    st.components.v1.html(html, height=700)
                                else:
                                    st.warning("可視化ライブラリが利用できません。テーブル表示します。")
                                    st.dataframe(filtered_data)
                    else:
                        st.info("グラフデータが見つかりません")
                except Exception as e:
                    st.error(f"エラー: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        if not st.session_state.initialized:
            st.info("RAGシステムを初期化するとグラフが表示されます")

# フッター
st.markdown("---")
st.markdown("**Graph-RAG Demo** | Powered by LangChain, Neo4j & PGVector")
