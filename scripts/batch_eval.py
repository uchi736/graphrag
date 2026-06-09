#!/usr/bin/env python
"""
batch_eval.py - GraphRAG バッチ評価ツール
==========================================
CSVで質問一覧を入力し、各質問の回答・引用元・KGチャンク・
取得リレーションをCSVに出力する。

使用例:
    python batch_eval.py --input questions.csv --output results.csv
    python batch_eval.py --input questions.csv  # デフォルト出力名
"""

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tqdm import tqdm

from graphrag_core.config import get_settings

# プロジェクトモジュール
from graphrag_core.llm.factory import create_chat_llm
from graphrag_core.llm.langfuse_utils import get_langfuse_callback, observe, propagate_attributes

# Langfuseバッチセッション（main()で設定）
_batch_session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
from graphrag_core.prompts import QA_PROMPT
from graphrag_core.db.utils import add_connection_timeout, retry_on_timeout
from graphrag_core.retrieval.pipeline import retriever_and_merge as qa_retriever_and_merge

# ロガー（ファイルハンドラは main() で追加）
logger = logging.getLogger("graphrag_eval")
logger.setLevel(logging.INFO)


# ─── 設定 ───────────────────────────────────────────────────────────
def load_config() -> dict:
    """Settings から設定を読み込む"""
    s = get_settings()
    return {
        "pg_conn": s.pg_conn,
        "pg_collection": s.pg_collection,
        "enable_japanese_search": s.enable_japanese_search,
        "search_mode": s.search_mode,
        "retrieval_top_k": s.retrieval_top_k,
        "include_kg_source_chunks": s.include_kg_source_chunks,
        "enable_rerank": s.enable_rerank,
        "enable_entity_vector": s.enable_entity_vector_search,
        "entity_similarity_threshold": s.entity_similarity_threshold,
        "graph_hop_count": s.graph_hop_count,
        "path_max_candidates": s.path_max_candidates,
        "neo4j_uri": s.neo4j_uri,
        "neo4j_user": s.neo4j_user,
        "neo4j_pw": s.neo4j_pw,
    }


# ─── セットアップ ───────────────────────────────────────────────────
def setup_system(config: dict):
    """グラフ・ベクトルストア・エンティティベクトライザーを初期化"""
    from langchain_postgres import PGVector
    from graphrag_core.llm.factory import create_embeddings

    s = get_settings()

    # グラフ（Neo4j）
    from langchain_neo4j import Neo4jGraph
    graph = Neo4jGraph(
        url=config["neo4j_uri"],
        username=config["neo4j_user"],
        password=config["neo4j_pw"],
        enhanced_schema=False,
    )
    try:
        nr = graph.query("MATCH (n) RETURN count(n) AS c")
        er = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")
        node_count = nr[0]["c"] if nr else 0
        edge_count = er[0]["c"] if er else 0
    except Exception:
        node_count = edge_count = 0
    print(f"  グラフ (Neo4j): {node_count}ノード, {edge_count}エッジ")

    # Embeddings
    embeddings = create_embeddings()

    # PGVector
    pg_conn = add_connection_timeout(config["pg_conn"], timeout=30)

    def create_vs():
        return PGVector(
            connection=pg_conn,
            embeddings=embeddings,
            collection_name=config["pg_collection"],
        )

    vector_store = retry_on_timeout(create_vs, max_retries=3, delay=2.0)
    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": config["retrieval_top_k"]}
    )
    print(f"  PGVector接続: コレクション={config['pg_collection']}")

    # エンティティベクトライザー
    entity_vectorizer = None
    if config["enable_entity_vector"]:
        try:
            from graphrag_core.retrieval.entity_vector import EntityVectorizer
            entity_vectorizer = EntityVectorizer(config["pg_conn"], embeddings)
            print("  エンティティベクトル検索: 有効")
        except Exception as e:
            print(f"  エンティティベクトル検索: 無効 ({e})")

    return graph, embeddings, vector_store, vector_retriever, entity_vectorizer


# ─── 1質問の処理 ────────────────────────────────────────────────────
@observe(name="graphrag_qa")
def run_single_question(
    question: str, graph, vector_retriever, embeddings, entity_vectorizer, config: dict
) -> dict:
    """質問1件を処理して結果dictを返す"""
    from langchain_core.documents import Document

    _q_sid = f"{_batch_session_id}_q{hash(question) % 10000:04d}"
    with propagate_attributes(session_id=_q_sid):
        return _run_single_question_inner(
            question, graph, vector_retriever, embeddings, entity_vectorizer, config
        )


def _run_single_question_inner(
    question: str, graph, vector_retriever, embeddings, entity_vectorizer, config: dict
) -> dict:
    """run_single_questionの内部実装（qa_pipeline共通関数を使用）"""
    logger.info(f"\n{'─'*60}")
    logger.info(f"質問: {question}")
    logger.info(f"{'─'*60}")

    llm = create_chat_llm(temperature=0)

    # 1. 検索・マージ（qa_pipeline共通関数）
    merge_result = qa_retriever_and_merge(
        question, graph, llm, embeddings, vector_retriever,
        config["pg_conn"], config["pg_collection"], config
    )

    context = merge_result["context"]
    docs = merge_result.get("vector_sources", [])
    kg_chunks = merge_result.get("kg_source_chunks", [])
    triples = merge_result.get("graph_sources", [])
    extracted_entities = merge_result.get("extracted_entities", {})

    # 2. LLM回答生成
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate.from_template(QA_PROMPT)
        llm_chain = prompt | create_chat_llm(temperature=0) | StrOutputParser()
        answer = llm_chain.invoke(
            {"question": question, "context": context},
            config=get_langfuse_callback()
        )
    except Exception as e:
        answer = f"[回答生成エラー] {e}"

    # 3. 結果を構築
    doc_sources = list(set(
        d.metadata.get("source", "") for d in docs if d.metadata.get("source")
    ))
    triples_text = "\n".join(
        f"{t.get('start', '')} -[{t.get('type', '')}]-> {t.get('end', '')}"
        for t in triples
    )
    llm_ents = extracted_entities.get("llm_entities", [])
    vec_ents = extracted_entities.get("vector_entities", [])

    result = {
        "question": question,
        "answer": answer,
        "doc_sources": ", ".join(doc_sources),
        "graph_triples": triples_text,
        "llm_entities": ", ".join(llm_ents),
        "vector_entities": ", ".join(f"{eid}:{score:.3f}" for eid, score in vec_ents),
    }
    # 1チャンク1列
    for i, d in enumerate(docs):
        result[f"doc_chunk_{i+1}"] = d.page_content
    for i, d in enumerate(kg_chunks):
        result[f"kg_chunk_{i+1}"] = d.page_content
    return result


# ─── メイン ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG バッチ評価ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python batch_eval.py --input questions.csv --output results.csv
  python batch_eval.py --input questions.csv
        """,
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="質問CSVファイル（question列必須）")
    parser.add_argument("--output", "-o", type=Path, default=None, help="出力CSVファイル（デフォルト: results_YYYYMMDD_HHMMSS.csv）")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"入力ファイルが見つかりません: {args.input}")
        sys.exit(1)

    output_path = args.output or Path(f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # ログファイルセットアップ（CSVと同名の .log）
    log_path = output_path.with_suffix(".log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    print(f"ログ: {log_path}")

    # CSV読み込み
    questions = []
    extra_columns = []
    with open(args.input, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "question" not in reader.fieldnames:
            print("CSVに 'question' 列がありません")
            sys.exit(1)
        extra_columns = [c for c in reader.fieldnames if c != "question"]
        for row in reader:
            q = row.get("question", "").strip()
            if q:
                questions.append(row)

    if not questions:
        print("質問が0件です")
        sys.exit(1)

    print(f"入力: {args.input} ({len(questions)}問)")
    print(f"出力: {output_path}")

    # セットアップ
    config = load_config()
    if not config["pg_conn"]:
        print("PG_CONN が未設定です")
        sys.exit(1)
    if not all([config["neo4j_uri"], config["neo4j_user"], config["neo4j_pw"]]):
        print("NEO4J_URI, NEO4J_USER, NEO4J_PW が必須です")
        sys.exit(1)
    print(f"グラフバックエンド: Neo4j ({config['neo4j_uri']})")

    print("\nシステム初期化中...")
    graph, embeddings, vector_store, vector_retriever, entity_vectorizer = setup_system(config)

    # バッチ実行
    print(f"\n{'='*50}")
    print(f"バッチ評価開始: {len(questions)}問")
    print(f"{'='*50}")

    # 全質問を処理してから列数を確定（チャンク数が質問ごとに異なるため）
    results = []
    max_doc_chunks = 0
    max_kg_chunks = 0

    for row in tqdm(questions, desc="評価中"):
        question = row["question"]
        try:
            result = run_single_question(
                question, graph, vector_retriever, embeddings, entity_vectorizer, config
            )
            # 入力CSVの追加列をマージ
            for col in extra_columns:
                result[col] = row.get(col, "")
        except Exception as e:
            result = {col: row.get(col, "") for col in extra_columns}
            result.update({
                "question": question,
                "answer": f"[エラー] {e}",
                "doc_sources": "", "graph_triples": "",
                "llm_entities": "", "vector_entities": "",
            })
            tqdm.write(f"  エラー: {question[:30]}... -> {e}")

        # 最大チャンク数を追跡
        doc_n = max((int(k.split("_")[-1]) for k in result if k.startswith("doc_chunk_")), default=0)
        kg_n = max((int(k.split("_")[-1]) for k in result if k.startswith("kg_chunk_")), default=0)
        max_doc_chunks = max(max_doc_chunks, doc_n)
        max_kg_chunks = max(max_kg_chunks, kg_n)
        results.append(result)

    # 列名確定（チャンク数は全質問の最大値に合わせる）
    doc_chunk_cols = [f"doc_chunk_{i+1}" for i in range(max_doc_chunks)]
    kg_chunk_cols = [f"kg_chunk_{i+1}" for i in range(max_kg_chunks)]
    output_columns = extra_columns + [
        "question", "answer", "doc_sources",
    ] + doc_chunk_cols + kg_chunk_cols + [
        "graph_triples", "llm_entities", "vector_entities",
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"\n完了: {output_path}")
    print(f"  doc_chunk列数: {max_doc_chunks}, kg_chunk列数: {max_kg_chunks}")


if __name__ == "__main__":
    main()
