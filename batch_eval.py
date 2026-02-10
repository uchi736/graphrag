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
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# プロジェクトモジュール
from llm_factory import create_chat_llm
from prompt import ENTITY_EXTRACTION_PROMPT, RELATION_RANKING_PROMPT, QA_PROMPT
from networkx_graph import NetworkXGraph
from db_utils import add_connection_timeout, retry_on_timeout


# ─── 設定 ───────────────────────────────────────────────────────────
def load_config() -> dict:
    """環境変数から設定を読み込む"""
    return {
        "pg_conn": os.getenv("PG_CONN", ""),
        "pg_collection": os.getenv("PG_COLLECTION", "graphrag"),
        "enable_japanese_search": os.getenv("ENABLE_JAPANESE_SEARCH", "true").lower() == "true",
        "search_mode": os.getenv("SEARCH_MODE", "hybrid"),
        "retrieval_top_k": int(os.getenv("RETRIEVAL_TOP_K", "5")),
        "include_kg_source_chunks": os.getenv("INCLUDE_KG_SOURCE_CHUNKS", "true").lower() == "true",
        "enable_rerank": os.getenv("ENABLE_RERANK", "false").lower() == "true",
        "enable_entity_vector": os.getenv("ENABLE_ENTITY_VECTOR", "true").lower() == "true",
        "entity_similarity_threshold": float(os.getenv("ENTITY_SIMILARITY_THRESHOLD", "0.7")),
        "graph_hop_count": int(os.getenv("GRAPH_HOP_COUNT", "1")),
        "graph_path": os.getenv("GRAPH_PATH", "graph.pkl"),
    }


# ─── セットアップ ───────────────────────────────────────────────────
def setup_system(config: dict):
    """グラフ・ベクトルストア・エンティティベクトライザーを初期化"""
    from langchain_postgres import PGVector
    from langchain_openai import AzureOpenAIEmbeddings

    # グラフ
    graph = NetworkXGraph(storage_path=config["graph_path"], auto_save=True)
    node_count = graph.graph.number_of_nodes()
    edge_count = graph.graph.number_of_edges()
    print(f"  グラフ読み込み: {node_count}ノード, {edge_count}エッジ")

    # Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

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
            from entity_vectorizer import EntityVectorizer
            entity_vectorizer = EntityVectorizer(config["pg_conn"], embeddings)
            print("  エンティティベクトル検索: 有効")
        except Exception as e:
            print(f"  エンティティベクトル検索: 無効 ({e})")

    return graph, embeddings, vector_store, vector_retriever, entity_vectorizer


# ─── エンティティ抽出 (app.py L686-750 相当) ────────────────────────
def extract_entities(question: str, embeddings, entity_vectorizer, config: dict) -> dict:
    """質問からエンティティを抽出"""
    result = {"llm_entities": [], "vector_entities": [], "merged_entities": []}
    entities = []

    # 1. LLMによるエンティティ抽出
    extraction_prompt = ENTITY_EXTRACTION_PROMPT.format(question=question)
    try:
        llm = create_chat_llm(temperature=0)
        response = llm.invoke(extraction_prompt)
        llm_entities = [e.strip() for e in response.content.split(",") if e.strip()]
        if llm_entities and llm_entities[0] != "なし":
            result["llm_entities"] = llm_entities
            entities.extend(llm_entities)
    except Exception:
        fallback = [w for w in question.split() if len(w) > 1]
        result["llm_entities"] = fallback
        entities.extend(fallback)

    # 2. ベクトル検索によるエンティティ抽出
    if entity_vectorizer and config["enable_entity_vector"]:
        try:
            similar = entity_vectorizer.search_hybrid_entities(
                question,
                k=10,
                score_threshold=config["entity_similarity_threshold"],
                search_type=config["search_mode"],
            )
            result["vector_entities"] = similar
            for eid, _score in similar:
                if eid not in entities:
                    entities.append(eid)
        except Exception:
            pass

    result["merged_entities"] = entities
    return result


# ─── リランキング (app.py L752-795 相当) ────────────────────────────
def rank_relations(question: str, relations: list, top_k: int = 15, doc_context: str = "") -> list:
    """LLMを使って関係性をリランキング"""
    if not relations:
        return []

    relations_text = "\n".join(
        f"{i+1}. {r['start']} -[{r['type']}]-> {r['end']}"
        for i, r in enumerate(relations)
    )
    ranking_prompt = RELATION_RANKING_PROMPT.format(
        question=question, relations_text=relations_text, top_k=top_k,
        document_context=doc_context if doc_context else "(なし)"
    )

    try:
        llm = create_chat_llm(temperature=0)
        response = llm.invoke(ranking_prompt)
        output = response.content.strip()
        if not output:
            return relations[:top_k]

        selected_ids = []
        for x in output.split(","):
            x = x.strip()
            if x.isdigit():
                selected_ids.append(int(x))

        ranked = [relations[i - 1] for i in selected_ids if 1 <= i <= len(relations)]
        if not ranked:
            return relations[:top_k]
        return ranked
    except Exception:
        return relations[:top_k]


# ─── グラフ検索 (app.py L798-905 相当) ──────────────────────────────
def get_graph_context(question: str, graph, embeddings, entity_vectorizer, config: dict, doc_context: str = "") -> dict:
    """エンティティ抽出 → グラフ検索 → リランキング（ドキュメントコンテキスト付き）"""
    entity_result = extract_entities(question, embeddings, entity_vectorizer, config)
    entities = entity_result.get("merged_entities", [])
    if not entities:
        return {"triples": [], "extracted_entities": entity_result}

    hop_count = config["graph_hop_count"]
    top_k = {1: 15, 2: 20, 3: 25}.get(hop_count, 15)

    try:
        result = graph.query("", params={"entities": entities, "hop": hop_count})
        if result:
            result = rank_relations(question, result, top_k=top_k, doc_context=doc_context)
        return {"triples": result or [], "extracted_entities": entity_result}
    except Exception:
        return {"triples": [], "extracted_entities": entity_result}


# ─── 1質問の処理 ────────────────────────────────────────────────────
def run_single_question(
    question: str, graph, vector_retriever, embeddings, entity_vectorizer, config: dict
) -> dict:
    """質問1件を処理して結果dictを返す"""
    from langchain_core.documents import Document

    # 1. ドキュメント検索（先に実行し、リランキングのコンテキストに使用）
    docs = []
    if config["enable_japanese_search"]:
        try:
            from japanese_text_processor import SUDACHI_AVAILABLE
            if SUDACHI_AVAILABLE:
                from hybrid_retriever import HybridRetriever, rerank_with_llm
                hybrid_retriever = HybridRetriever(
                    config["pg_conn"], collection_name=config["pg_collection"]
                )
                query_embedding = embeddings.embed_query(question)
                hybrid_results = hybrid_retriever.search(
                    query_text=question,
                    query_vector=query_embedding,
                    k=config["retrieval_top_k"],
                    search_type=config["search_mode"],
                )
                # LLMリランキング
                if config["enable_rerank"] and hybrid_results:
                    rerank_llm = create_chat_llm(temperature=0)
                    hybrid_results = rerank_with_llm(
                        question, hybrid_results, rerank_llm, k=config["retrieval_top_k"]
                    )
                docs = [
                    Document(page_content=r["text"], metadata=r["metadata"])
                    for r in hybrid_results
                ]
            else:
                docs = vector_retriever.invoke(question)
        except Exception:
            docs = vector_retriever.invoke(question)
    else:
        docs = vector_retriever.invoke(question)

    # 2. グラフ検索（ドキュメントコンテキストでリランキング）
    doc_context = "\n---\n".join(d.page_content[:200] for d in docs[:5])
    graph_result = get_graph_context(question, graph, embeddings, entity_vectorizer, config, doc_context=doc_context)
    triples = graph_result.get("triples", [])
    extracted_entities = graph_result.get("extracted_entities", {})

    # 3. KGソースチャンク取得
    kg_chunks = []
    if triples and config["include_kg_source_chunks"]:
        entity_names = list(
            set([t.get("start") for t in triples] + [t.get("end") for t in triples])
        )
        if entity_names and hasattr(graph, "get_source_chunks_list"):
            try:
                existing_texts = {d.page_content for d in docs}
                chunk_results = graph.get_source_chunks_list(entity_names, limit=5)
                for r in chunk_results:
                    if r.get("text") and r["text"] not in existing_texts:
                        kg_chunks.append(
                            Document(
                                page_content=r["text"],
                                metadata={"id": r.get("chunk_id"), "source": r.get("source", "KG")},
                            )
                        )
                        existing_texts.add(r["text"])
            except Exception:
                pass

    # 4. コンテキスト構築
    graph_lines = []
    if triples:
        source_chunks = {}
        if hasattr(graph, "get_source_chunks_for_entities"):
            entity_ids = list(
                set([t.get("start") for t in triples] + [t.get("end") for t in triples])
            )
            source_chunks = graph.get_source_chunks_for_entities(entity_ids)

        for t in triples:
            start, rel, end = t.get("start", ""), t.get("type", ""), t.get("end", "")
            src = (
                source_chunks.get(start, {}).get("source")
                or source_chunks.get(end, {}).get("source")
                or ""
            )
            if src:
                graph_lines.append(f"{start} -[{rel}]-> {end} [出典: {src}]")
            else:
                graph_lines.append(f"{start} -[{rel}]-> {end}")
    else:
        graph_lines = ["(グラフデータなし)"]

    all_docs = docs.copy()
    if config["include_kg_source_chunks"]:
        all_docs.extend(kg_chunks)

    context = (
        "<GRAPH_CONTEXT>\n" + "\n".join(graph_lines) + "\n</GRAPH_CONTEXT>\n\n"
        + "<DOCUMENT_CONTEXT>\n"
        + "\n---\n".join(d.page_content for d in all_docs)
        + "\n</DOCUMENT_CONTEXT>"
    )

    # 5. LLM回答生成
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate.from_template(QA_PROMPT)
        llm_chain = prompt | create_chat_llm(temperature=0) | StrOutputParser()
        answer = llm_chain.invoke({"question": question, "context": context})
    except Exception as e:
        answer = f"[回答生成エラー] {e}"

    # 6. 結果を構築
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
