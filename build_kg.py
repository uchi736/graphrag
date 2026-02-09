#!/usr/bin/env python
"""
build_kg.py - CLI版ナレッジグラフ構築スクリプト
================================================
Streamlitのタイムアウト問題を回避するためのCLIツール。
フォルダ指定で複数ファイルを一括処理し、graph.pkl/graph.jsonに保存。

使用例:
    # フォルダ内の全ファイルを処理
    python build_kg.py --input ./docs

    # 新規構築（処理済みをクリア）
    python build_kg.py --input ./docs --fresh

    # 特定の拡張子のみ
    python build_kg.py --input ./docs --ext pdf,md
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chunk_utils import create_markdown_chunks
from tqdm import tqdm

from prompt import KG_SYSTEM_PROMPT, KG_USER_PROMPT

# 環境変数読み込み
load_dotenv()


def load_file(file_path: Path) -> Optional[Document]:
    """ファイルを読み込んでDocumentを返す"""
    try:
        suffix = file_path.suffix.lower()
        file_name = file_path.name

        if suffix == '.pdf':
            # Azure Document Intelligence または PyMuPDF
            azure_di_endpoint = os.getenv("AZURE_DI_ENDPOINT")
            azure_di_api_key = os.getenv("AZURE_DI_API_KEY")

            if azure_di_endpoint and azure_di_api_key:
                try:
                    from azure_di_processor import AzureDocumentIntelligenceProcessor, AzureDIConfig
                    config = AzureDIConfig()
                    processor = AzureDocumentIntelligenceProcessor(config)
                    docs = processor.process(str(file_path))
                    if docs:
                        text_content = docs[0].page_content

                        # Azure DI出力を保存
                        output_dir = Path("output")
                        output_dir.mkdir(exist_ok=True)
                        output_filename = file_path.stem + "_azure_di.md"
                        output_path = output_dir / output_filename
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(f"# {file_name}\n\n")
                            f.write(f"*Processed by Azure Document Intelligence*\n\n")
                            f.write("---\n\n")
                            f.write(text_content)
                        print(f"  📄 Azure DI出力を保存: {output_path}")
                    else:
                        text_content = ""
                except Exception as e:
                    print(f"  ⚠️ Azure DI処理エラー、PyMuPDFにフォールバック: {e}")
                    import fitz
                    pdf_doc = fitz.open(str(file_path))
                    text_parts = []
                    for page_num in range(len(pdf_doc)):
                        page = pdf_doc[page_num]
                        text = page.get_text("text", sort=True)
                        if text.strip():
                            text_parts.append(text)
                    pdf_doc.close()
                    text_content = "\n\n".join(text_parts)
            else:
                # PyMuPDF
                import fitz
                pdf_doc = fitz.open(str(file_path))
                text_parts = []
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    text = page.get_text("text", sort=True)
                    if text.strip():
                        text_parts.append(text)
                pdf_doc.close()
                text_content = "\n\n".join(text_parts)

        elif suffix in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            if suffix == '.md' and file_name.endswith('_azure_di.md'):
                print(f"  📄 Azure DI処理済みファイルを読み込み")
        else:
            print(f"  ⚠️ 未対応の拡張子: {suffix}")
            return None

        if not text_content.strip():
            print(f"  ⚠️ 空のファイル")
            return None

        return Document(
            page_content=text_content,
            metadata={"source": file_name}
        )

    except Exception as e:
        print(f"  ❌ 読み込みエラー: {e}")
        return None


def build_knowledge_graph(
    input_dir: Path,
    extensions: List[str],
    fresh: bool = False
):
    """ナレッジグラフを構築"""
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from llm_factory import create_chat_llm
    from networkx_graph import NetworkXGraph

    # ファイル一覧取得
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f"**/*.{ext}"))
    files = sorted(set(files))

    if not files:
        print(f"❌ 対象ファイルが見つかりません: {input_dir}")
        print(f"   拡張子: {extensions}")
        return

    print(f"\n📁 入力フォルダ: {input_dir}")
    print(f"📄 対象ファイル: {len(files)}件")
    print(f"   拡張子: {', '.join(extensions)}")

    # ファイル読み込み
    print(f"\n{'='*50}")
    print("📖 ファイル読み込み中...")
    print(f"{'='*50}")

    source_docs = []
    for file_path in tqdm(files, desc="読み込み"):
        print(f"\n  {file_path.name}")
        doc = load_file(file_path)
        if doc:
            source_docs.append(doc)

    if not source_docs:
        print("❌ 読み込めたファイルがありません")
        return

    total_chars = sum(len(doc.page_content) for doc in source_docs)
    print(f"\n✅ {len(source_docs)}ファイル読み込み完了（総文字数: {total_chars:,}文字）")

    # チャンク分割
    print(f"\n{'='*50}")
    print("✂️ チャンク分割中...")
    print(f"{'='*50}")

    # 2段階Markdownチャンキング（##, ### で分割 → 1024文字で再分割）
    all_chunks = create_markdown_chunks(source_docs, chunk_size=1024, chunk_overlap=100)

    # 重複除去
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
    print(f"✅ {len(chunks)}チャンク（重複除去後）")

    # グラフ初期化
    print(f"\n{'='*50}")
    print("🕸️ ナレッジグラフ構築中...")
    print(f"{'='*50}")

    graph = NetworkXGraph(storage_path="graph.pkl", auto_save=False)

    # 新規構築の場合は処理済みクリア
    if fresh:
        graph.clear_processed_hashes()
        print("🗑️ 処理済みデータをクリアしました")

    # 処理済みハッシュ取得
    processed_hashes = graph.get_processed_hashes()

    # 未処理チャンクフィルタ
    pending_chunks = [c for c in chunks if c.metadata.get("id") not in processed_hashes]
    skipped_count = len(chunks) - len(pending_chunks)

    if skipped_count > 0:
        print(f"📋 処理対象: {len(pending_chunks)}/{len(chunks)}チャンク（{skipped_count}件スキップ）")

    if not pending_chunks:
        print("✅ すべてのチャンクは処理済みです")
        return

    # LLMGraphTransformer設定
    llm_provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()
    print(f"🤖 LLMプロバイダー: {llm_provider}")

    if llm_provider == "vllm":
        llm = create_chat_llm(temperature=0)
        transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=["Term"],
            allowed_relationships=[
                "IS_A", "BELONGS_TO_CATEGORY", "PART_OF", "HAS_STEP",
                "HAS_ATTRIBUTE", "RELATED_TO", "AFFECTS", "CAUSES",
                "DEPENDS_ON", "APPLIES_TO", "OWNED_BY", "SAME_AS"
            ],
            strict_mode=False,
        )
    else:
        from langchain_core.prompts import ChatPromptTemplate
        llm = create_chat_llm(temperature=0)

        kg_prompt = ChatPromptTemplate.from_messages([
            ("system", KG_SYSTEM_PROMPT),
            ("user", KG_USER_PROMPT)
        ])

        transformer = LLMGraphTransformer(
            llm=llm,
            prompt=kg_prompt,
            allowed_nodes=["Term"],
            allowed_relationships=[
                "IS_A", "BELONGS_TO_CATEGORY", "PART_OF", "HAS_STEP",
                "HAS_ATTRIBUTE", "RELATED_TO", "AFFECTS", "CAUSES",
                "DEPENDS_ON", "APPLIES_TO", "OWNED_BY", "SAME_AS"
            ],
            strict_mode=False,
        )

    # チャンクごとに処理（100チャンクごとに定期保存）
    SAVE_INTERVAL = 100
    success_count = 0
    error_count = 0

    for chunk in tqdm(pending_chunks, desc="グラフ構築"):
        try:
            chunk_docs = transformer.convert_to_graph_documents([chunk])
            graph.add_graph_documents(chunk_docs, include_source=True)

            chunk_hash = chunk.metadata.get("id")
            if chunk_hash:
                graph.mark_chunk_processed(chunk_hash, save=False)

            success_count += 1

            # 定期保存（進捗を失わないため）
            if success_count % SAVE_INTERVAL == 0:
                graph.save()

        except Exception as e:
            error_count += 1
            tqdm.write(f"  ⚠️ エラー: {e}")
            graph.save()
            continue

    # 最終保存
    graph.save()

    # グラフ統計
    node_count = graph.graph.number_of_nodes()
    edge_count = graph.graph.number_of_edges()
    print(f"\n✅ グラフ構築完了: 成功{success_count}件, エラー{error_count}件")
    print(f"   ノード数: {node_count}, エッジ数: {edge_count}")

    # PGVector保存
    print(f"\n{'='*50}")
    print("📦 PGVector保存中...")
    print(f"{'='*50}")

    try:
        from langchain_openai import AzureOpenAIEmbeddings
        from db_utils import ensure_embedding_id_unique, ensure_schema_compatibility, ensure_hnsw_index, add_connection_timeout, batch_pgvector_from_documents

        PG_CONN = os.getenv("PG_CONN")
        PG_COLLECTION = os.getenv("PG_COLLECTION", "graphrag")

        if not PG_CONN:
            print("⚠️ PG_CONN未設定のためPGVector保存をスキップ")
        else:
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )

            # IDのNULLチェック
            ids = [c.metadata.get("id") for c in chunks if c.metadata.get("id")]
            if len(ids) != len(chunks):
                print("⚠️ 一部チャンクにIDがありません")

            ensure_embedding_id_unique(PG_CONN)
            ensure_schema_compatibility(PG_CONN)
            ensure_hnsw_index(PG_CONN)

            # タイムアウト設定
            pg_conn_with_timeout = add_connection_timeout(PG_CONN, timeout=30)

            # バッチ分割でPGVector保存（bindパラメータ上限対策）
            vector_store = batch_pgvector_from_documents(
                chunks,
                embeddings,
                connection=pg_conn_with_timeout,
                collection_name=PG_COLLECTION,
                pre_delete_collection=fresh,  # 新規構築時のみ削除
                progress_callback=lambda i, total, n: print(f"  PGVector: {i+n}/{total}チャンク"),
            )
            print(f"✅ PGVector保存完了: {len(chunks)}チャンク")

            # エンティティベクトル化
            print(f"\n{'='*50}")
            print("🔍 エンティティベクトル化中...")
            print(f"{'='*50}")

            try:
                from entity_vectorizer import EntityVectorizer

                entity_vectorizer = EntityVectorizer(PG_CONN, embeddings)

                # グラフからエンティティを抽出
                entities = entity_vectorizer.extract_entities_from_graph(
                    graph,
                    graph_backend="networkx"
                )
                print(f"  抽出エンティティ数: {len(entities)}")

                # エンティティをベクトル化して保存
                # graph_docsは空でもOK（エンティティIDだけ保存）
                num_saved = entity_vectorizer.add_entities(entities, [])

                if num_saved > 0:
                    print(f"✅ {num_saved}個のエンティティをベクトル化しました")
                elif len(entities) == 0:
                    print("⚠️ グラフにエンティティが見つかりません")
                else:
                    print(f"⚠️ {len(entities)}個のエンティティの保存に失敗しました（ログを確認してください）")

            except ImportError:
                print("⚠️ EntityVectorizerが見つかりません（スキップ）")
            except Exception as e:
                print(f"⚠️ エンティティベクトル化エラー: {e}")

    except Exception as e:
        print(f"⚠️ PGVector保存エラー: {e}")

    # 結果表示
    print(f"\n{'='*50}")
    print("📊 最終結果")
    print(f"{'='*50}")
    print(f"🕸️ グラフ: graph.pkl, graph.json")
    print(f"   ノード数: {node_count}")
    print(f"   エッジ数: {edge_count}")


def main():
    parser = argparse.ArgumentParser(
        description="CLI版ナレッジグラフ構築ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python build_kg.py --input ./docs
  python build_kg.py --input ./docs --fresh
  python build_kg.py --input ./docs --ext pdf,md
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="入力フォルダのパス"
    )
    parser.add_argument(
        "--ext", "-e",
        type=str,
        default="pdf,txt,md",
        help="処理する拡張子（カンマ区切り、デフォルト: pdf,txt,md）"
    )
    parser.add_argument(
        "--fresh", "-f",
        action="store_true",
        help="新規構築（処理済みデータをクリア）"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ フォルダが存在しません: {args.input}")
        sys.exit(1)

    extensions = [ext.strip().lower() for ext in args.ext.split(",")]

    print("🚀 ナレッジグラフ構築開始")
    print(f"   モード: {'新規構築' if args.fresh else '続きから再開'}")

    build_knowledge_graph(
        input_dir=args.input,
        extensions=extensions,
        fresh=args.fresh
    )

    print("\n✅ 完了！")
    print("   Streamlitで確認: streamlit run app.py")


if __name__ == "__main__":
    main()
