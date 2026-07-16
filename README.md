# Graph-RAG (React UI + FastAPI / 完全オンプレ対応)

最新版のLangChain APIに対応した、Graph-RAGシステムの完全実装です。
**React SPA + FastAPI** のWeb UIを搭載し、PDF/テキストファイルから自動的にナレッジグラフを構築・可視化できます。
LLM・embedding・リランカー・PDF解析まで**完全オンプレミス構成**（vLLM）が既定です。

![Graph-RAG Demo](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 主な機能

### Web UI（React + FastAPI、5タブ）
- **質問応答**: SSEトークンストリーミング、根拠パネル（参照ドキュメント／**参照グラフ**＝回答に使われた関係の可視化／推論パス／KGソースチャンク）。参照カードから**文書内チャンクへジャンプ**（該当チャンクをハイライト）
- **グラフ探索**: force-graph可視化（タイプ別色分け・次数サイズ）、サブグラフ表示、ノード/エッジCRUD、自然言語→Cypher、**graph.jsonエクスポート**
- **登録ドキュメント**: ソース別チャンク集計、チャンク本文ブラウザ、**1文書の増分更新（改訂ファイル→差分のみ再抽出ジョブ）/完全削除**
- **構築/取り込み**: アップロード→KG構築ジョブ（SSE進捗・キャンセル・再開）、**KGスキーマカード**（現グラフ刻印 vs 次回ビルド設定の不一致警告、EDCスキーマ同期＝サーバ不要の内蔵実行、**人手キュレーション編集**）、**バッチ評価**（question列CSV→一括QA→結果CSVダウンロード）
- **設定**: ヘルスチェック、危険操作（グラフ/DBクリア）
- 旧 Streamlit UI（`scripts/app.py`）も移行期間中は併存

### ドキュメント取り込み・構築
- **PDF解析（構造保持Markdown）**: `PDF_PROCESSOR=doc_parser` で doc-parser サービス（:8770）経由の**表を潰さない**取り込み。エンジンは **docling（既定・IBM製＝非中国系スタックで完結）** / mineru（複雑帳票の結合セル精度優先）を切替。doclingの和文行折返しスペースは決定的正規化で自動補正
- **図の取り込み（図表RAG）**: doclingが切り出した図を **gemma4-vision がキャプション文化**し、①本文への差し込みに加えて②**独立した図チャンク**（`type=figure`）としてPGVector＋BM25に登録（`DOC_PARSER_FIGURE_CAPTIONS=true`）。質問がキャプションにハイブリッド検索で一致すると、**QAの根拠カードに図の画像そのものが表示**される（画像は `output/figures/` 保存・`/figures/` 配信）
- **フォールバック**: doc_parser → オンプレOCR（PaddleX）→ PyMuPDF。Azure DI も選択可
- **再開機能**: 処理が中断しても続きから再開可能（内容ハッシュのチャンクIDで管理）
- **増分更新**: 文書改訂時に該当文書スコープだけ差分更新（`scripts/update_doc.py` / API・UI）

### CLIツール
- **build_kg.py**: Streamlitを使わずにナレッジグラフを構築
  - フォルダ指定で複数ファイルを一括処理
  - 長時間処理でもタイムアウトなし
  - KG構築完了時に `graph.json` を自動エクスポート
- **batch_eval.py**: CSVバッチ評価ツール
  - 質問CSVを入力、回答・引用元・KGチャンクをCSV出力
  - Langfuseトレーシング対応
- **対話型質問応答**: Graph-First RAGによる高精度な回答生成
- **LLMリランキング**: 質問の意図に応じた関係性フィルタリング
- **CSVエッジインポート**: 外部関係データの取り込み対応
- **グラフ編集機能**: ノード・エッジの追加・編集・削除

### 高度な検索機能
- **高/低レベルキーワード分離（LightRAG流 dual-level）**: 質問から固有名（低）とテーマ語（高）を同時抽出。テーマ語は**エッジのキーワード索引**（`{collection}_relations`、`scripts/build_relation_keywords.py` で構築）に埋め込み照合し、固有名を含まない抽象質問でもKGが寄与する。エンティティ照合が全滅した場合は質問全文→エンティティ埋め込みのフォールバック（llm-graph-builder流）も発動。**グラフ再構築後は build_relation_keywords.py を再実行すること**
- **日本語ハイブリッド検索**: Sudachiによる形態素解析 + RRFスコア統合
- **cross-encoderリランキング**: bge-reranker-v2-m3等によるドキュメント/パス/KGチャンクの再評価（LLMリランクはフォールバック）。広めに取得（rerank_pool_size）→ top_k に絞る2段構成
- **正規化エンティティ照合**: NFKC正規化済み `search_keys`（id + 辞書aliases + canonical_form + かな揺れ骨格キー）に対するCONTAINS照合で表記揺れ（全角半角・大小文字・送り仮名・助詞・長音）を吸収
- **エンティティベクトル検索**: エンティティの類似度検索（類義語・関連語対応）
- **可変ホップ検索**: 1〜3ホップのグラフ探索。起点ノードは完全一致優先 + pagerank順で選択、パス候補は extraction_count / pagerank でスコアリングしてから取得
- **ノイズノード除外**: 数値・日付のみの値ノード（is_value）と「本製品」等の照応ノード（is_anaphor）をトラバーサルから除外（値ノードはパス終端のみ許可）
- **KGソースチャンク取得**: triple→source_chunks直引き、フォールバックはMENTIONS共起順（言及エンティティ数降順）+ cross-encoderリランク
- **参照追跡（オプション）**: 「P.98参照」「『◯◯』をご覧ください」を REFERS_TO エッジで1ホップ追跡（`enable_reference_follow`、デフォルトOFF）

### 高度なグラフ可視化
- **可視化エンジン**: `react-force-graph-2d`（Canvas・d3-force、React UI）/ `neo4j-viz`（Streamlit）
- **3つの表示モード**:
  - **グラフ可視化**: インタラクティブな探索（ノードクリック→詳細→中心にサブグラフ展開）
  - **データテーブル**: ノード・エッジ一覧とCSV出力
  - **Cypherクエリ検索**: 自然言語→Cypher変換機能（参照専用実行・書込拒否）
- **タイプ別色分け・接続数に応じたノードサイズ**: 重要なエンティティが一目でわかる
- **QA参照グラフ**: 回答に使われた関係だけを関係名ラベル付きで描画

### Graph-First RAGアーキテクチャ
```
質問 → エンティティ抽出 → エンティティベクトル検索
                              ↓
                        1-3 hop グラフ検索 (Neo4j Cypher)
                              ↓
                        LLMリランキング
                              ↓
                    関連チャンク取得 (MENTIONS関係)
                              ↓
                    フォールバック: ハイブリッド検索
                              ↓
                        Context Merge
                              ↓
                    Azure OpenAI / VLLM → 回答生成
```

### 技術スタック
- **フロントエンド**: React + TypeScript + Vite + Tailwind（zustand / TanStack Query / react-force-graph-2d、CDN不使用＝オフラインビルド可）
- **バックエンドAPI**: FastAPI（SSEストリーミング、プロセス内ジョブ管理 → **workers=1 必須**）
- **LLMGraphTransformer**: カスタムプロンプト対応（ChatPromptTemplate）
- **Neo4j**: グラフデータベース（必須）
- **PGVector**: ベクトルデータベース（PostgreSQL）
- **LLMプロバイダー**:
  - **VLLM**: セルフホステッドモデル（既定。gemma等）
  - **Azure OpenAI**: 明示的に有効化した場合のみ
- **PDF処理**:
  - **オンプレOCR**: PaddleX PP-OCRv5（リモートエンドポイント）
  - **Azure Document Intelligence**: 高精度PDF解析
  - **PyMuPDF**: 軽量PDF解析
- **日本語処理**: Sudachi形態素解析
- **スキーマ自動発見**: EDC（Extract-Define-Canonicalize、`vendor/EDC/` に同梱）
- **Langfuse**: LLMトレーシング（SDK v4、オプション）

## ファイル構成

```
graphrag/
├── graphrag_core/              # コアパッケージ
│   ├── config.py               # 設定管理（Settings dataclass）
│   ├── prompts.py              # プロンプトテンプレート集約
│   ├── llm/                    # LLM関連
│   │   ├── factory.py          # LLMプロバイダー切り替え（Azure/VLLM）
│   │   └── langfuse_utils.py   # Langfuseトレーシング設定（SDK v4）
│   ├── db/                     # データベース
│   │   └── utils.py            # PostgreSQL接続・インデックス管理
│   ├── text/                   # テキスト処理
│   │   ├── japanese.py         # Sudachi形態素解析（スレッドセーフ）+ エンティティ正規化
│   │   └── chunking.py         # Markdown対応チャンク処理（見出しパンくず付与）
│   ├── graph/                  # Neo4jグラフ操作
│   │   ├── base.py             # GraphBackend Protocol
│   │   ├── neo4j_ops.py        # Neo4j CRUD操作 + JSONエクスポート
│   │   ├── crud.py             # 統一CRUDディスパッチャ
│   │   ├── schema.py           # KGスキーマ管理（外部JSON差し替え対応）+ 共通Cypher述語
│   │   ├── enrichment.py       # ビルド後プロパティ付与（mention_count/pagerank/search_keys/source_chunks）
│   │   ├── consolidate.py      # KG統合（値ノードflag/型分裂・かな揺れマージ/関係正規化/照応解決）
│   │   ├── references.py       # 参照グラフ（節/ページ/文書名参照のルールベース抽出）
│   │   ├── conditions.py       # 条件付き関係(qualifier)のreify格納（:CondFact/:Cond/[:WHEN]）
│   │   ├── provenance.py       # グラフ出自(PG_COLLECTION)の刻印と整合性チェック
│   │   ├── incremental.py      # 文書スコープの部分グラフ更新（差分検出→剪定→再抽出→同期）
│   │   └── dictionary.py       # 専門用語辞書の適用（canonical_form/aliases付与）
│   ├── retrieval/              # 検索
│   │   ├── hybrid.py           # 日本語ハイブリッド検索（BM25 + Vector、RRF統合）
│   │   ├── reranker.py         # cross-encoderリランカークライアント（vLLM /score互換）
│   │   ├── entity_vector.py    # エンティティベクトル化・類似度検索
│   │   └── pipeline.py         # 共通QAパイプライン
│   ├── services/               # UI非依存サービス層（Streamlit / FastAPI 双方から利用）
│   │   ├── qa.py               # QA一元実装（検索+生成、SSEイベント生成）
│   │   ├── build.py            # KG構築/チャンク更新（進捗コールバック+協調キャンセル）
│   │   ├── graph_explore.py    # 可視化用エッジ取得・NL→Cypher・参照専用実行
│   │   ├── documents.py        # 登録文書集計・チャンク閲覧・増分更新用抽出関数
│   │   ├── schema_sync.py      # スキーマ突き合わせ + EDCスキーマ自動発見ジョブ
│   │   ├── ingest.py / admin.py / retrievers.py / progress.py
│   ├── document/               # ドキュメント処理
│   │   └── azure_di.py         # Azure Document Intelligence ほか
│   └── ui/                     # Streamlit UIコンポーネント（タブ別に分割）
│       ├── state.py            # session_state初期化 + 全タブ共有コンテキスト(ctx)
│       ├── system.py           # システム構築/既存グラフ復元ロジック
│       ├── css.py              # スタイル定義
│       ├── sidebar.py          # サイドバー設定UI
│       ├── qa_tab.py           # 質問応答タブ
│       ├── graph_tab.py        # グラフ探索タブ
│       ├── documents_tab.py    # 登録ドキュメントタブ
│       ├── build_tab.py        # 構築/取り込みタブ
│       ├── settings_tab.py     # 設定タブ
│       ├── feedback.py         # フィードバック収集
│       ├── visualization.py    # グラフ可視化
│       ├── data_tables.py      # データテーブル表示
│       └── dialogs.py          # 編集ダイアログ
├── api/                        # FastAPI 配送層（React UI のバックエンド）
│   ├── main.py                 # create_app + lifespan（AppState初期化）+ SPA静的配信
│   ├── state.py / deps.py / sse.py / jobs.py  # 共有状態・503ゲート・SSE・プロセス内ジョブ
│   └── routers/                # admin / qa / graph / documents / build
├── frontend/                   # React SPA（Vite + TypeScript + Tailwind）
│   └── src/{api,stores,hooks,pages,components,lib}/
├── scripts/                    # エントリーポイント
│   ├── app.py                  # Streamlit WebUI（移行期間中併存。ロジックはservices/*へ委譲）
│   ├── build_kg.py             # CLIナレッジグラフ構築（統合・参照グラフ・enrichment込み）
│   ├── update_doc.py           # 1文書の増分更新CLI（差分→剪定→再抽出→同期）
│   ├── edc_schema_sync.py      # EDCスキーマ自動発見CLI（チャンクディレクトリから）
│   ├── build_reference_graph.py # 既存グラフへの参照グラフ後付け（再構築不要）
│   ├── batch_eval.py           # CSVバッチ評価
│   ├── init_japanese_search.py # 日本語検索初期化
│   ├── check_pg_connection.py  # PostgreSQL接続テスト
│   ├── check_schema.py         # スキーマ確認ユーティリティ
│   └── reset_pgvector_tables.py # PGVectorテーブルリセット
├── vendor/EDC/                 # EDCフレームワーク同梱（スキーマ自動発見。VENDOR.md参照）
├── schemas/                    # EDC同期で生成したスキーマJSON（SHARED_SCHEMA_PATHに指定）
├── _bench/                     # Fujitsu RAG Hard Bench 実験一式（EXPERIMENTS.md に記録）
├── tests/                      # テスト
│   └── test_vllm.py            # VLLM接続テスト
├── docs/                       # ドキュメント（EDC_INTEGRATION.md ほか）
├── requirements.txt            # Python依存関係
├── .env.sample                 # 環境変数テンプレート
└── output/                     # Azure DI処理結果（自動生成）
```

## 必要な環境

- Python 3.10以上
- **Neo4j** (必須): Neo4j Aura またはローカルインスタンス（Docker推奨）
- **PostgreSQL** with PGVector拡張
- **Azure OpenAI** APIキー（またはVLLMサーバー）

## クイックスタート

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env`ファイルをプロジェクトのルートディレクトリに作成（詳細は `.env.sample` 参照）:

```env
# Neo4j Configuration (必須)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PW=your_neo4j_password

# PostgreSQL Configuration
PG_CONN=postgresql+psycopg://postgres:your_password@your-host:5432/postgres
PG_COLLECTION=graphrag

# Azure OpenAI Service Settings
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small

# Azure Document Intelligence (Optional - 高精度PDF解析)
AZURE_DI_ENDPOINT=https://your-di-resource.cognitiveservices.azure.com
AZURE_DI_API_KEY=your_document_intelligence_api_key
AZURE_DI_MODEL=prebuilt-layout

# Knowledge Graph Configuration
ENABLE_KNOWLEDGE_GRAPH=true

# Entity Vector Search Configuration
ENABLE_ENTITY_VECTOR_SEARCH=true
# 同義語/表記揺れ補完は高類似度が必須（0.85 が実効既定）
ENTITY_SIMILARITY_THRESHOLD=0.85

# Japanese Hybrid Search Configuration
ENABLE_JAPANESE_SEARCH=true

# Retrieval Configuration
RETRIEVAL_TOP_K=5
GRAPH_HOP_COUNT=2
# cross-encoder リランク（最強レバー +11.8pt）→ 既定ON
ENABLE_RERANK=true
# グラフ三つ組行を LLM コンテキストに含めるか（noLines がマルチホップで優位 → 既定OFF）
INCLUDE_GRAPH_LINES=false

# LLM Provider (azure_openai or vllm)
LLM_PROVIDER=azure_openai

# VLLM Configuration (LLM_PROVIDER=vllm の場合)
VLLM_ENDPOINT=https://your-vllm-server.example.com/v1
VLLM_TEMPERATURE=0.0
VLLM_MAX_TOKENS=4096

# Embedding Provider (azure_openai or vllm)
# 切替時はPGVectorコレクション再構築必須（次元が変わるため）
EMBEDDING_PROVIDER=azure_openai
VLLM_EMBEDDING_ENDPOINT=http://localhost:8001/v1
VLLM_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B

# Cross-encoder Reranker (vLLM /v1/score 互換、未設定時はLLMリランクにフォールバック)
RERANKER_ENABLED=true
VLLM_RERANKER_ENDPOINT=http://localhost:8006/v1
VLLM_RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# KG Schema (外部JSONでノードタイプ・関係セットを差し替え。未設定時はTerm+12関係)
SHARED_SCHEMA_PATH=_bench/fujitsu_kg_schema_v2.json

# 専門用語辞書 (canonical_form/aliases を Term に付与、search_keys 経由で検索に反映)
KG_DICTIONARY_PATH=

# PDF Preprocessing (onprem / azure_di / pymupdf)
PDF_PROCESSOR=onprem
PDF_BACKEND=vllm
VLLM_VISION_ENDPOINT=http://localhost:8004/v1
VLLM_VISION_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
PREPROCESSING_OPTIMIZER_PATH=../preprocessing_optimizer
```

### 3. Neo4j の起動（ローカルDockerの場合）

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/changeme123 \
  neo4j:5
```

### 4. アプリケーションの起動

**React UI + FastAPI（推奨）:**

```bash
# 初回のみ: フロントエンドをビルド
cd frontend && npm ci && npm run build && cd ..

# APIサーバ起動（React UI を同一ポートで配信。ジョブ管理がプロセス内のため workers=1 必須）
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

ブラウザで `http://localhost:8000` にアクセスしてください。
QAはSSEストリーミング、KG構築はジョブ進捗表示に対応しています。
（開発時は `cd frontend && npm run dev` で Vite dev server(:5173) + `/api` プロキシ）

**旧 Streamlit UI（移行期間中は併用可）:**

```bash
streamlit run scripts/app.py   # http://localhost:8501
```

## 使い方

### 初回利用時

1. **PDFまたはテキストファイルをアップロード**
   - 複数ファイルの同時アップロード可能
   - サポート形式: PDF, TXT, MD
   - オプション: edges.csv（関係データのインポート）

2. **「ナレッジグラフを構築」をクリック**
   - 自動的にエンティティとリレーションを抽出
   - Neo4jとPGVectorに保存

3. **質問を入力**
   - グラフとベクトル検索を組み合わせた高精度な回答
   - LLMリランキングで関連度の高い情報のみを利用

4. **グラフを探索**
   - **グラフ可視化**: インタラクティブにノードとエッジを探索
   - **データテーブル**: ノード・エッジ一覧をCSV出力、編集
   - **Cypherクエリ検索**: 自然言語→Cypher自動生成

### 2回目以降

既にグラフデータがある場合、再構築は不要です:
- **React UI**: ステートレスなので起動した時点でそのまま質問可能（グラフの有無・出自は構築タブとヘルスチェックに表示）
- **Streamlit UI**: 起動時にNeo4jのデータを自動検出 → 「既存グラフを読み込む」をクリック

### 再開機能

大量のドキュメントを処理する場合、途中で中断しても続きから再開できます（構築/取り込みタブ）:

- **「新規構築」**: 処理済みデータをクリアして最初から構築
- **「続きから再開」**: 未処理のチャンクのみ処理

**仕組み**:
- 各チャンクにSHA256ハッシュIDを付与
- Neo4jに処理済みチャンクノード（ProcessedChunk）を保存
- 再開時は処理済みハッシュと照合してスキップ

### CLI版 (build_kg.py)

Streamlitのセッションタイムアウトを回避したい場合や、大量のドキュメントを処理する場合はCLI版を使用:

```bash
# フォルダ内の全ファイルを処理
python scripts/build_kg.py --input ./docs

# 新規構築（処理済みをクリア）
python scripts/build_kg.py --input ./docs --fresh

# 特定の拡張子のみ
python scripts/build_kg.py --input ./docs --ext pdf,md
```

**オプション**:
| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 入力フォルダのパス | (必須) |
| `--ext`, `-e` | 処理する拡張子（カンマ区切り） | pdf,txt,md |
| `--fresh`, `-f` | 新規構築（処理済みデータをクリア） | false |

**CLI版の特徴**:
- セッションタイムアウトなし
- フォルダ内のファイルを再帰的に処理
- 再開機能対応（--freshなしで実行すると続きから）
- PGVectorへの自動保存
- エンティティベクトル化対応（全エンティティ対象、ページネーション取得）
- **ビルド後処理を自動実行**:
  1. KG統合（consolidate）: 値ノードflag・型分裂ノードのマージ・かな揺れマージ（送り仮名/助詞/長音のみの差分を文字種限定編集距離で統合）・関係タイプ正規化（逆関係/同義/typoの27ルール）
  2. 参照グラフ構築: 節/ページ/文書名参照の REFERS_TO エッジ + 略称定義による照応解決
  3. 用語辞書適用（KG_DICTIONARY_PATH 指定時）
  4. enrichment: mention_count / pagerank / search_keys（正規化照合キー）の付与
- 完了時に `graph.json` を自動エクスポート

既存グラフに参照グラフ・照応解決だけ後付けする場合（再構築不要、LLMコストゼロ）:

```bash
python scripts/build_reference_graph.py
```

### バッチ評価 (batch_eval.py)

質問CSVを入力し、各質問の回答・引用元をCSVに出力:

```bash
python scripts/batch_eval.py --input questions.csv --output results.csv
python scripts/batch_eval.py --input questions.csv  # デフォルト出力名
```

**入力CSV形式**: `question`列必須（追加列はそのまま出力に含まれる）

**オプション**:
| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 質問CSVファイル | (必須) |
| `--output`, `-o` | 出力CSVファイル | results_YYYYMMDD_HHMMSS.csv |

**出力CSV列**: 入力の追加列 + question, answer, doc_sources, doc_chunk_1..N, kg_chunk_1..N, graph_triples, llm_entities, vector_entities

## 検索設定（React UIは質問応答タブ上部、Streamlitはサイドバー）

### グラフ可視化設定
- **最大表示ノード数**: 表示エッジ上限を調整可能

### 検索設定
- **TopK**: 検索で取得するチャンク数（1〜20）
- **ホップ数**: グラフ探索の深さ（1〜3）
- **エンティティベクトル検索**: ON/OFF、類似度閾値
- **日本語ハイブリッド検索**: ハイブリッド/ベクトル/キーワード
- **LLMリランキング**: ON/OFF
- **KGソースチャンク**: トリプルの出典チャンクをコンテキストに含めるかどうか

## 高度な機能

### graph.json エクスポート

Neo4j上のグラフデータをJSON形式でエクスポートできます:

- `build_kg.py` でのKG構築完了時に自動生成
- 出力形式:
  ```json
  {
    "graph": {
      "directed": true,
      "multigraph": true,
      "nodes": [{"id": "エンティティA", "type": "Term"}, ...],
      "edges": [{"source": "A", "target": "B", "type": "RELATED"}, ...]
    }
  }
  ```
- 追加学習のインプットや外部ツールとの連携に利用可能
- 出力先は `graph.json` 固定（`.gitignore` で追跡対象外）

### Azure Document Intelligence

高精度なPDF解析が必要な場合:
- テーブル構造の保持
- 複雑なレイアウトのMarkdown変換
- OCR対応
- `.env`に`AZURE_DI_ENDPOINT`と`AZURE_DI_API_KEY`を設定

**出力キャッシュ**:
- Azure DIで処理したPDFは`output/`フォルダに`{ファイル名}_azure_di.md`として保存
- 次回は.mdファイルを直接アップロードすることでAzure DI呼び出しをスキップ可能
- コスト削減と処理時間短縮に有効

### CSVエッジインポート

外部の関係データを取り込む場合:
```csv
source,target,label
エンティティA,エンティティB,RELATED_TO
```
- `edges.csv`をアップロードするだけで自動取り込み

### リランキング（cross-encoder優先）

検索結果（ドキュメント・グラフパス・KGチャンク）を質問との関連度で再評価:

**仕組み**:
1. `RERANKER_ENABLED=true` なら cross-encoder（bge-reranker-v2-m3等、vLLM /v1/score）でスコアリング（LLMリランクの10〜100倍高速）
2. 未設定時は LLM によるID順位付けにフォールバック
3. グラフパスは「AはBの一種である」のような**自然文に変換してから**スコアリング（記号表記より安定）

**ホップ数ごとの設定**:
| ホップ数 | 取得上限 | リランク後 |
|---------|---------|-----------|
| 1-hop | 30件 | 15件 |
| 2-hop | 50件 | 20件 |
| 3-hop | 80件 | 25件 |

候補はDB側で extraction_count（エッジの抽出回数）/ pagerank によりスコアリングしてから取得するため、無作為なLIMIT切り捨ては発生しません。

### Langfuseトレーシング

LLM呼び出しの全フローをLangfuseでトレース可能（SDK v4対応）:
- `@observe()` デコレータによる階層トレース自動構築
- エンティティ抽出→グラフ検索→リランキング→回答生成の全ステップを可視化
- `.env`に `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` を設定
- 未設定時はゼロオーバーヘッドで従来通り動作
- セルフホスト（Docker Compose）・クラウド両対応

### KGスキーマの外部化

ノードタイプと関係タイプのセットは外部JSON（`SHARED_SCHEMA_PATH`）で差し替えられます。
未設定時はデフォルトの `Term` 1タイプ + 12関係
（`IS_A` / `BELONGS_TO_CATEGORY` / `PART_OF` / `HAS_STEP` / `HAS_ATTRIBUTE` / `RELATED_TO` /
`AFFECTS` / `CAUSES` / `DEPENDS_ON` / `APPLIES_TO` / `OWNED_BY` / `SAME_AS`）。

スキーマJSON例は [_bench/fujitsu_kg_schema_v2.json](_bench/fujitsu_kg_schema_v2.json)
（8ノードタイプ + 23関係、マルチドメインコーパス向け）を参照。
適用中のスキーマは Neo4j の `:SchemaMeta` ノードに刻印されます。

**スキーマ設計の指針**（実測に基づく、詳細は [_bench/EXPERIMENTS.md](_bench/EXPERIMENTS.md)）:
- 数値・日付・単位のみの値はノードにしない（プロンプトで禁止 + consolidateで除外）
- 逆方向ペア関係（HAS_PART等）は定義しない（検索は無向マッチのため重複だけ生む）
- `strict_mode=True` で運用し、スキーマ外の野良関係を抽出時にブロック

## アーキテクチャ

```
PDF/Text → Azure DI / PyMuPDF → Markdownチャンキング（見出しパンくず付与）
                                       ↓
                         LLMGraphTransformer (スキーマ外部化 + strict_mode)
                                       ↓
                            ┌──────────┴──────────┐
                            ↓                     ↓
                         Neo4j               PGVector
                      (グラフDB)            (ベクトルDB)
                            ↓                     ↓
              ビルド後処理:              エンティティベクトル化
              ・consolidate（値ノード/型分裂/関係正規化）
              ・参照グラフ（REFERS_TO）+ 照応解決
              ・辞書適用 → enrichment（pagerank/search_keys）
                            ↓                     ↓
                            └──────────┬──────────┘
                                       ↓
                              質問応答フロー
                                       ↓
        エンティティ抽出（正規化） ∥ 日本語ハイブリッド検索（並列実行）
                                       ↓
          1-3 hop グラフ探索 (search_keys照合・スコア順パス取得)
                                       ↓
              cross-encoderリランキング（パスは自然文化）
                                       ↓
        KGソースチャンク取得（source_chunks / MENTIONS共起順 + リランク）
                                       ↓
                        Context Merge + 回答生成
```

## 主要な依存ライブラリ

```
streamlit>=1.28.0
langchain-core>=0.1.0
langchain-openai>=0.1.0
langchain-experimental>=0.1.0
langchain-postgres>=0.0.14
langchain-graph-retriever>=0.1.0
neo4j>=5.20.0
pymupdf>=1.24.0
azure-ai-documentintelligence>=1.0.0
streamlit-agraph>=0.0.45
pyvis>=0.3.2
psycopg[binary]>=3.1.19
sudachipy>=0.6.8
sudachidict_core>=20240716
rank-bm25>=0.2.2
langfuse>=4.0.0
```

## トラブルシューティング

### グラフが表示されない
- `pyvis`または`streamlit-agraph`がインストールされているか確認
- ブラウザのJavaScriptが有効か確認

### Neo4j接続エラー
- `.env`の`NEO4J_URI`が正しいか確認
- Neo4jインスタンスが起動しているか確認
- ファイアウォールでポート7687が開いているか確認
- ローカルDockerの場合: `docker ps` でコンテナが稼働中か確認

### PGVector接続エラー
- PostgreSQLにPGVector拡張がインストールされているか確認
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```
- `PG_CONN`の接続文字列が正しいか確認

### Azure Document Intelligence エラー
- `AZURE_DI_ENDPOINT`と`AZURE_DI_API_KEY`が正しいか確認
- 未設定の場合はPyMuPDFにフォールバック

### コンテンツフィルターエラー
- Azure OpenAIのコンテンツフィルターに引っかかった場合
- ドキュメントの内容によっては回避不可
- 該当ドキュメントを除外して再処理

### 日本語検索が機能しない
- `sudachipy`と`sudachidict_core`がインストールされているか確認
  ```bash
  pip install sudachipy sudachidict_core
  ```

## カスタマイズ

### プロンプトのカスタマイズ

`graphrag_core/prompts.py`の`KG_SYSTEM_PROMPT`を編集することで、抽出ルールをカスタマイズできます。

### 関係タイプ・ノードタイプの追加

コードを変更せず、スキーマJSONを作って `.env` の `SHARED_SCHEMA_PATH` で指定します:

```json
{
  "domain": "your-domain",
  "version": "v1.0",
  "node_types": ["Term"],
  "relations": [
    {"name": "IS_A", "definition": "AはBの一種である", "examples": ["..."]},
    {"name": "YOUR_NEW_RELATION", "definition": "...", "examples": ["..."]}
  ]
}
```

`build_kg.py` が自動でこのスキーマを `allowed_nodes` / `allowed_relationships` に反映します。
逆方向関係や値ノードを作らない等の設計指針は「KGスキーマの外部化」の節を参照してください。

スキーマの手書きの代わりに **EDCフレームワークによる自動発見**も使えます
（UI構築タブの「EDCスキーマ同期」または `scripts/edc_schema_sync.py`）。
セットアップと使い方: [docs/EDC_INTEGRATION.md](docs/EDC_INTEGRATION.md)

## 評価・ベンチマーク

Fujitsu RAG Hard Benchmark（100問）による継続評価を `_bench/` で実施しています:

- 実験記録・スコア推移・設計判断の根拠: [_bench/EXPERIMENTS.md](_bench/EXPERIMENTS.md)
- 実行: `_bench/fujitsu_runner_kg.py`（KG込みパイプライン）/ `_bench/fujitsu_runner.py`（hybrid+rerankのみ）
- 採点: `_bench/evaluate_qa_local.py`（gemma judge）+ `_bench/_judge_azure.py`（第2judge）
- 解析: `_bench/analyze_by_axes.py`（軸別スライス）/ `_bench/_pair_compare.py`（ペア勝敗）/ `_bench/_schema_diag.py`（KG構造診断）

主な知見: KGの寄与は multi-hop推論・多文書entity比較で +7〜8pt、single-hop・数値表では中立〜負。
judge1系統のスコアは言い回しで±2〜5pt揺れるため、2judge + 参照カバレッジ（決定的指標）の併用を推奨。

## 関連ドキュメント

- [.env.sample](.env.sample): 環境変数の設定例（全オプション記載）
- [docs/langfuse_plan.md](docs/langfuse_plan.md): Langfuse統合の設計ドキュメント
- [_bench/EXPERIMENTS.md](_bench/EXPERIMENTS.md): ベンチマーク実験記録（FJH-01〜FJH-11）

---

**Built with LangChain, Neo4j, PGVector, Azure OpenAI & Sudachi**
