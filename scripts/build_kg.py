#!/usr/bin/env python
"""
build_kg.py - CLI版ナレッジグラフ構築スクリプト
================================================
Streamlitのタイムアウト問題を回避するためのCLIツール。
フォルダ指定で複数ファイルを一括処理し、Neo4jにKGを構築。

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
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from graphrag_core.text.chunking import create_markdown_chunks
from tqdm import tqdm

from graphrag_core.prompts import KG_SYSTEM_PROMPT, KG_USER_PROMPT
from graphrag_core.config import get_settings


def load_file(file_path: Path) -> Optional[Document]:
    """ファイルを読み込んでDocumentを返す"""
    from graphrag_core.document.pdf import load_pdf_text
    try:
        suffix = file_path.suffix.lower()
        file_name = file_path.name

        if suffix == '.pdf':
            text_content = load_pdf_text(str(file_path))
        elif suffix in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
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
    fresh: bool = False,
):
    """ナレッジグラフを構築"""
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from graphrag_core.llm.factory import create_chat_llm
    from graphrag_core.llm.langfuse_utils import is_langfuse_enabled

    s = get_settings()

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

    from langchain_neo4j import Neo4jGraph
    if not all([s.neo4j_uri, s.neo4j_user, s.neo4j_pw]):
        print("NEO4J_URI, NEO4J_USER, NEO4J_PW が必須です")
        return
    graph = Neo4jGraph(url=s.neo4j_uri, username=s.neo4j_user, password=s.neo4j_pw, enhanced_schema=True)
    print(f"  バックエンド: Neo4j ({s.neo4j_uri})")

    # 新規構築の場合は処理済みクリア
    if fresh:
        try:
            graph.query("MATCH (c:ProcessedChunk) DELETE c")
        except Exception:
            pass
        print("処理済みデータをクリアしました")

    # 処理済みハッシュ取得
    try:
        processed = graph.query("MATCH (c:ProcessedChunk) RETURN c.hash AS hash")
        processed_hashes = {r["hash"] for r in processed} if processed else set()
    except Exception:
        processed_hashes = set()

    # 未処理チャンクフィルタ
    pending_chunks = [c for c in chunks if c.metadata.get("id") not in processed_hashes]
    skipped_count = len(chunks) - len(pending_chunks)

    if skipped_count > 0:
        print(f"📋 処理対象: {len(pending_chunks)}/{len(chunks)}チャンク（{skipped_count}件スキップ）")

    skip_graph_build = not pending_chunks
    if skip_graph_build:
        print("✅ すべてのチャンクは処理済み（グラフ構築をスキップ → PGVector同期のみ実行）")

    # LLMGraphTransformer設定
    llm_provider = s.llm_provider.lower()
    print(f"🤖 LLMプロバイダー: {llm_provider}")

    # KG構築用のLangfuseセッションID
    _kg_session_id = f"kg_build_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    success_count = 0
    error_count = 0

    if not skip_graph_build:
        from langchain_core.prompts import ChatPromptTemplate

        llm = create_chat_llm(temperature=0)
        if is_langfuse_enabled():
            from langfuse.langchain import CallbackHandler as LangfuseHandler
            llm.callbacks = [LangfuseHandler(trace_context={"name": "kg_building", "session_id": _kg_session_id})]

        # VLLM (ignore_tool_usage=True) ではカスタムpromptを渡さない。
        # LLMGraphTransformerのデフォルトpromptがJSON出力形式を指示するため、
        # カスタムpromptを渡すとJSON指示が欠落しパース失敗する。
        _kg_additional = (
            "抽出する: 技術用語、概念、固有名詞、プロセス名、規格名。"
            "抽出しない: 一般的な名詞（「こと」「もの」「方法」）、代名詞、動詞。"
            "抽出しない: 数値・日付・年度・単位のみの値（「210円」「53」「令和6年度」）。"
            "値はノードにしない。"
            "RELATED_TOは他に適切な関係がない場合の最終手段として使用。"
        )
        from graphrag_core.graph.schema import get_allowed_node_types, get_allowed_relations
        # strict_mode=True: スキーマ外の野良関係タイプ（typo含め125種が混入した実績）を
        # 抽出時点でフィルタする。post-hocのrename（consolidate.py）は残存分の保険
        _kg_kwargs = dict(
            llm=llm,
            allowed_nodes=get_allowed_node_types(),
            allowed_relationships=get_allowed_relations(),
            strict_mode=True,
            ignore_tool_usage=(llm_provider == "vllm"),
        )
        if llm_provider == "vllm":
            _kg_kwargs["additional_instructions"] = _kg_additional
        else:
            kg_prompt = ChatPromptTemplate.from_messages([
                ("system", KG_SYSTEM_PROMPT),
                ("user", KG_USER_PROMPT)
            ])
            _kg_kwargs["prompt"] = kg_prompt

        transformer = LLMGraphTransformer(**_kg_kwargs)

        # チャンクごとに処理（並列実行でVLLM continuous batching活用）
        import concurrent.futures

        from graphrag_core.graph.enrichment import attach_source_chunks

        def _process_single_chunk(chunk):
            """1チャンクのグラフ変換 + Neo4j書き込み + edge source_chunks 付与"""
            chunk_docs = transformer.convert_to_graph_documents([chunk])
            graph.add_graph_documents(chunk_docs, include_source=True)
            chunk_hash = chunk.metadata.get("id")
            if chunk_hash:
                # edgeに「このtripleはこのチャンクから抽出された」記録を後付け
                attach_source_chunks(graph, chunk_docs, chunk_hash)
                graph.query(
                    "MERGE (c:ProcessedChunk {hash: $hash}) SET c.processed_at = datetime()",
                    {"hash": chunk_hash},
                )
            return chunk_hash

        max_workers = int(os.environ.get("KG_BUILD_WORKERS", "4"))
        print(f"🔄 並列ワーカー数: {max_workers} (KG_BUILD_WORKERS環境変数で変更可)")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(_process_single_chunk, chunk): chunk
                for chunk in pending_chunks
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_chunk),
                total=len(pending_chunks),
                desc="グラフ構築",
            ):
                try:
                    future.result()
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    tqdm.write(f"  ⚠️ エラー: {e}")
                    continue

    # グラフ統計
    try:
        nr = graph.query("MATCH (n) RETURN count(n) AS c")
        er = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")
        node_count = nr[0]["c"] if nr else 0
        edge_count = er[0]["c"] if er else 0
    except Exception:
        node_count = edge_count = 0
    if skip_graph_build:
        print(f"\n📊 既存グラフ: ノード数: {node_count}, エッジ数: {edge_count}")
    else:
        print(f"\n✅ グラフ構築完了: 成功{success_count}件, エラー{error_count}件")
        print(f"   ノード数: {node_count}, エッジ数: {edge_count}")

    # PGVector保存
    print(f"\n{'='*50}")
    print("📦 PGVector保存中...")
    print(f"{'='*50}")

    try:
        from graphrag_core.llm.factory import create_embeddings
        from graphrag_core.db.utils import ensure_embedding_id_unique, ensure_schema_compatibility, ensure_hnsw_index, add_connection_timeout, batch_pgvector_from_documents

        PG_CONN = s.pg_conn
        PG_COLLECTION = s.pg_collection

        if not PG_CONN:
            print("⚠️ PG_CONN未設定のためPGVector保存をスキップ")
        else:
            embeddings = create_embeddings()

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
            # skip_graph_build=True は「Neo4j既存・PGVector同期だけ実行」モード:
            # コレクションを一旦wipeして冪等に再投入する（古い orphan チャンクの掃除も兼ねる）。
            pre_delete = fresh or skip_graph_build
            vector_store = batch_pgvector_from_documents(
                chunks,
                embeddings,
                connection=pg_conn_with_timeout,
                collection_name=PG_COLLECTION,
                pre_delete_collection=pre_delete,
                progress_callback=lambda i, total, n: print(f"  PGVector: {i+n}/{total}チャンク"),
            )
            print(f"✅ PGVector保存完了: {len(chunks)}チャンク (pre_delete={pre_delete})")

            # 日本語トークン化（ハイブリッド検索用）
            from graphrag_core.text.japanese import get_japanese_processor
            from graphrag_core.db.utils import ensure_tokenized_schema, batch_update_tokenized

            japanese_processor = get_japanese_processor()

            if japanese_processor and s.enable_japanese_search:
                ensure_tokenized_schema(PG_CONN)
                print("日本語トークン化中...")
                for chunk in chunks:
                    try:
                        tokenized = japanese_processor.tokenize(chunk.page_content)
                        chunk.metadata['tokenized_content'] = tokenized
                    except Exception:
                        chunk.metadata['tokenized_content'] = None
                updated = batch_update_tokenized(PG_CONN, chunks)
                print(f"✅ 日本語トークン化完了: {updated}件")

            # エンティティベクトル化
            print(f"\n{'='*50}")
            print("🔍 エンティティベクトル化中...")
            print(f"{'='*50}")

            try:
                from graphrag_core.retrieval.entity_vector import EntityVectorizer

                entity_vectorizer = EntityVectorizer(PG_CONN, embeddings)

                # グラフからエンティティを抽出
                entities = entity_vectorizer.extract_entities_from_graph(
                    graph
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

    # KG統合処理: 値ノードflag・型分裂マージ・関係正規化
    # （辞書適用・enrichmentより先に実行: マージ後のノードに適用するため）
    print(f"\n{'='*50}")
    print("🔧 KG統合処理（値ノード/型分裂/関係正規化）...")
    print(f"{'='*50}")
    try:
        from graphrag_core.graph.consolidate import consolidate_post_build
        cstats = consolidate_post_build(graph)
        print(f"✅ 値ノードflag: {cstats['value_nodes_flagged']}件")
        print(f"✅ 型分裂マージ: {cstats['duplicate_merge']['merged_ids']}id "
              f"({cstats['duplicate_merge']['removed_nodes']}ノード削除)")
        print(f"✅ かな揺れマージ: {cstats['kana_variant_merge']['merged_groups']}組 "
              f"({cstats['kana_variant_merge']['removed_nodes']}ノード削除)")
        print(f"✅ 関係正規化: {sum(cstats['relation_normalize'].values())}エッジ")
    except Exception as e:
        print(f"⚠️ KG統合処理エラー: {e}")

    # 参照グラフ構築（節/ページ/文書名参照のREFERS_TOエッジ）+ 照応解決
    print(f"\n{'='*50}")
    print("🔗 参照グラフ構築 + 照応解決...")
    print(f"{'='*50}")
    try:
        from graphrag_core.graph.references import build_reference_graph
        from graphrag_core.graph.consolidate import resolve_anaphora_nodes
        ref_stats = build_reference_graph(graph)
        print(f"✅ 参照エッジ: {ref_stats['edges_written']}本, "
              f"文書名参照チャンク: {ref_stats['doc_ref_chunks']}件")
        ana_stats = resolve_anaphora_nodes(graph, ref_stats.get("alias_maps", {}))
        print(f"✅ 照応解決: {ana_stats['resolved']}件, 検索除外フラグ: {ana_stats['flagged']}件")
    except Exception as e:
        print(f"⚠️ 参照グラフ構築エラー: {e}")

    # 条件付き関係(qualifier/reify)の抽出・格納（規程・基準系コーパス向け、既定OFF）
    # consolidate の後・enrich(search_keys 再計算) の前に置く
    if s.enable_conditional_facts:
        try:
            from graphrag_core.graph.conditions import build_conditional_graph
            cf_stats = build_conditional_graph(graph, create_chat_llm(temperature=0), s.pg_collection)
            print(f"✅ 条件付き事実: 言明 {cf_stats['condfact_nodes']}件 / 条件 {cf_stats['cond_nodes']}件")
        except Exception as e:
            print(f"⚠️ 条件付き事実の構築エラー: {e}")

    # 専門用語辞書の適用（KG_DICTIONARY_PATH が指定されていれば）
    # 注: enrich_post_build の search_keys が aliases/canonical_form を取り込むため、
    # 辞書適用は enrichment より先に実行する
    if s.kg_dictionary_path:
        try:
            from graphrag_core.graph.dictionary import load_dictionary, apply_dictionary
            entries = load_dictionary(s.kg_dictionary_path)
            stats = apply_dictionary(graph, entries)
            print(f"✅ 用語辞書適用: {stats['applied_entries']}/{len(entries)}entries 適用 → "
                  f"{stats['term_updates']}ノード更新（未マッチentries: {stats['untouched_entries']}）")
        except FileNotFoundError as e:
            print(f"⚠️ 辞書ファイルが見つかりません: {e}")
        except Exception as e:
            print(f"⚠️ 辞書適用エラー: {e}")

    # プロパティ後付け: mention_count, pagerank, search_keys
    print(f"\n{'='*50}")
    print("🔧 KGプロパティ集計中...")
    print(f"{'='*50}")
    try:
        from graphrag_core.graph.enrichment import enrich_post_build
        stats = enrich_post_build(graph)
        print(f"✅ mention_count: {stats['mention_count']} ノード更新")
        print(f"✅ pagerank: {stats['pagerank']} ノード更新")
        print(f"✅ search_keys: {stats['search_keys']} ノード更新")
    except Exception as e:
        print(f"⚠️ プロパティ集計エラー: {e}")

    # スキーマメタ情報を Neo4j に刻印（EDC連携時の追跡用）
    try:
        from graphrag_core.graph.schema import stamp_schema_metadata, describe_schema
        stamp_schema_metadata(graph)
        print(f"✅ スキーマ刻印: {describe_schema()}")
    except Exception as e:
        print(f"⚠️ スキーマ刻印エラー: {e}")

    # graph.json エクスポート（ローカル + shared/ 両方）
    try:
        from graphrag_core.graph.neo4j_ops import export_graph_json
        export_graph_json(graph, output_path="graph.json")
        print("✅ graph.json エクスポート完了")
        # shared/ にも出力（cpt-data-pipeline 等の外部ツール連携用）
        shared_out = os.getenv("SHARED_GRAPH_JSON_PATH")
        if shared_out:
            from pathlib import Path
            Path(shared_out).parent.mkdir(parents=True, exist_ok=True)
            export_graph_json(graph, output_path=shared_out)
            print(f"✅ shared graph.json: {shared_out}")
    except Exception as e:
        print(f"⚠️ graph.json エクスポートエラー: {e}")

    # 結果表示
    print(f"\n{'='*50}")
    print("📊 最終結果")
    print(f"{'='*50}")
    print(f"グラフ: Neo4j ({s.neo4j_uri})")
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

    print("ナレッジグラフ構築開始")
    print(f"   モード: {'新規構築' if args.fresh else '続きから再開'}")
    print(f"   バックエンド: Neo4j")

    build_knowledge_graph(
        input_dir=args.input,
        extensions=extensions,
        fresh=args.fresh,
    )

    print("\n✅ 完了！")
    print("   Streamlitで確認: streamlit run app.py")


if __name__ == "__main__":
    main()
