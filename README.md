# Graph-RAG with Streamlit UI

最新版のLangChain APIに対応した、Graph-RAGシステムの完全実装です。
**Streamlit UI**を搭載し、PDF/テキストファイルから自動的にナレッジグラフを構築・可視化できます。

![Graph-RAG Demo](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 主な機能

### Streamlit WebUI
- **PDF/テキスト/Markdownファイルアップロード**: ドラッグ&ドロップで簡単アップロード
  - **Azure Document Intelligence**: 高精度PDF解析（テーブル・レイアウト保持、OCR対応）
  - **PyMuPDF**: 軽量PDF解析（フォールバック）
  - **Azure DI出力キャッシュ**: 処理結果を.mdファイルとして保存、再利用可能
- **ワンクリックでナレッジグラフ構築**: LLMが自動的にエンティティとリレーションを抽出
- **既存グラフの自動検出**: Neo4jに既にデータがあれば即座に復元
- **再開機能**: 処理が中断しても続きから再開可能（チャンクハッシュで管理）

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
- **日本語ハイブリッド検索**: Sudachiによる形態素解析 + RRFスコア統合
- **エンティティベクトル検索**: エンティティの類似度検索（類義語・関連語対応）
- **可変ホップ検索**: 1〜3ホップのグラフ探索
- **パス探索**: BFS取得トリプルをパスに再構成し、LLMがチェーン単位で評価

### 高度なグラフ可視化
- **可視化エンジン**: Neo4j公式 `neo4j-viz` ライブラリ（ネイティブなグラフDBスタイル）
- **3つの表示モード**:
  - **グラフ可視化**: インタラクティブな探索
  - **データテーブル**: ノード・エッジ一覧とCSV出力
  - **Cypherクエリ検索**: 自然言語→Cypher変換機能
- **ノードタイプ自動判定**: 専門用語を自動分類
- **タイプ別色分け**: 視覚的にわかりやすいカラーコーディング
- **接続数に応じたノードサイズ**: 重要なエンティティが一目でわかる

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
- **LLMGraphTransformer**: カスタムプロンプト対応（ChatPromptTemplate）
- **Neo4j**: グラフデータベース（必須）
- **PGVector**: ベクトルデータベース（PostgreSQL）
- **LLMプロバイダー**:
  - **Azure OpenAI**: GPT-4o, text-embedding-3-small
  - **VLLM**: セルフホステッドモデル対応
- **PDF処理**:
  - **Azure Document Intelligence**: 高精度PDF解析
  - **PyMuPDF**: 軽量PDF解析
- **日本語処理**: Sudachi形態素解析
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
│   │   ├── japanese.py         # Sudachi形態素解析
│   │   └── chunking.py         # Markdown対応チャンク処理
│   ├── graph/                  # Neo4jグラフ操作
│   │   ├── base.py             # GraphBackend Protocol
│   │   ├── neo4j_ops.py        # Neo4j CRUD操作 + JSONエクスポート
│   │   └── crud.py             # 統一CRUDディスパッチャ
│   ├── retrieval/              # 検索
│   │   ├── hybrid.py           # 日本語ハイブリッド検索（BM25 + Vector、RRF統合）
│   │   ├── entity_vector.py    # エンティティベクトル化・類似度検索
│   │   └── pipeline.py         # 共通QAパイプライン
│   ├── document/               # ドキュメント処理
│   │   └── azure_di.py         # Azure Document Intelligence
│   └── ui/                     # Streamlit UIコンポーネント
│       ├── css.py              # スタイル定義
│       ├── sidebar.py          # サイドバー設定UI
│       ├── visualization.py    # グラフ可視化
│       ├── data_tables.py      # データテーブル表示
│       └── dialogs.py          # 編集ダイアログ
├── scripts/                    # エントリーポイント
│   ├── app.py                  # Streamlit WebUI
│   ├── build_kg.py             # CLIナレッジグラフ構築
│   ├── batch_eval.py           # CSVバッチ評価
│   ├── init_japanese_search.py # 日本語検索初期化
│   ├── check_pg_connection.py  # PostgreSQL接続テスト
│   ├── check_schema.py         # スキーマ確認ユーティリティ
│   └── reset_pgvector_tables.py # PGVectorテーブルリセット
├── tests/                      # テスト
│   └── test_vllm.py            # VLLM接続テスト
├── docs/                       # ドキュメント
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
ENTITY_SIMILARITY_THRESHOLD=0.7

# Japanese Hybrid Search Configuration
ENABLE_JAPANESE_SEARCH=true

# Retrieval Configuration
RETRIEVAL_TOP_K=5
GRAPH_HOP_COUNT=2

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

```bash
streamlit run scripts/app.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

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

既にグラフデータがある場合:
- アプリ起動時にNeo4jのデータを自動検出
- **「既存グラフを読み込む」をクリック**するだけで即座に利用可能
- 再構築不要！

### 再開機能

大量のドキュメントを処理する場合、途中で中断しても続きから再開できます:

**Streamlit UI**:
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
- エンティティベクトル化対応
- 完了時に `graph.json` を自動エクスポート

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

## サイドバー機能

### グラフ可視化設定
- **可視化エンジン**: Neo4j-viz
- **最大表示ノード数**: 50〜100,000個

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

### LLMリランキング

グラフ検索で取得した関係性をLLMで再評価し、質問に関連度の高いものだけを抽出:

**仕組み**:
1. グラフから候補となる関係性を取得（30〜80件）
2. LLMが各関係性を0-10でスコアリング
3. 高スコアの関係性のみをコンテキストとして使用

**ホップ数ごとの設定**:
| ホップ数 | 取得上限 | リランク後 |
|---------|---------|-----------|
| 1-hop | 30件 | 15件 |
| 2-hop | 50件 | 20件 |
| 3-hop | 80件 | 25件 |

### Langfuseトレーシング

LLM呼び出しの全フローをLangfuseでトレース可能（SDK v4対応）:
- `@observe()` デコレータによる階層トレース自動構築
- エンティティ抽出→グラフ検索→リランキング→回答生成の全ステップを可視化
- `.env`に `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` を設定
- 未設定時はゼロオーバーヘッドで従来通り動作
- セルフホスト（Docker Compose）・クラウド両対応

### カスタムKG抽出プロンプト

専門用語に特化したナレッジグラフを構築できます。

**対応する関係タイプ (16種類)**:

**階層・分類関係**
- `IS_A`: 上位下位関係（MySQL → データベース）
- `BELONGS_TO_CATEGORY`: カテゴリ所属
- `PART_OF`: 部分構成関係（エンジン → 自動車）
- `HAS_STEP`: プロセスのステップ

**属性・特性関係**
- `HAS_ATTRIBUTE`: 属性保持
- `RELATED_TO`: 一般的な関連性

**因果・依存関係**
- `AFFECTS`: 影響関係
- `CAUSES`: 原因結果
- `DEPENDS_ON`: 依存関係

**適用・制約関係**
- `APPLIES_TO`: 適用対象
- `APPLIES_WHEN`: 適用条件
- `REQUIRES_QUALITY_GATE`: 品質ゲート要求
- `REQUIRES_APPROVAL_FROM`: 承認要求

**所有・責任関係**
- `OWNED_BY`: 所有者

**同義語関係**
- `SAME_AS`: 完全同義
- `ALIAS_OF`: エイリアス・略称

## アーキテクチャ

```
PDF/Text → Azure DI / PyMuPDF → Markdown対応チャンキング
                                       ↓
                         LLMGraphTransformer (カスタムプロンプト)
                                       ↓
                            ┌──────────┴──────────┐
                            ↓                     ↓
                         Neo4j               PGVector
                      (グラフDB)            (ベクトルDB)
                            ↓                     ↓
                  KGリレーション作成       エンティティベクトル化
                            ↓                     ↓
                            └──────────┬──────────┘
                                       ↓
                              質問応答フロー
                                       ↓
                   エンティティ抽出 + ベクトル検索
                                       ↓
                     1-3 hop グラフ探索 (Cypher)
                                       ↓
                             LLMリランキング
                                       ↓
                    関連チャンク取得 (MENTIONS)
                                       ↓
               フォールバック: 日本語ハイブリッド検索
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

### 関係タイプの追加

`allowed_relationships`リストに新しい関係タイプを追加:

```python
transformer = LLMGraphTransformer(
    llm=llm,
    prompt=kg_prompt,
    allowed_nodes=["Term"],
    allowed_relationships=[
        "IS_A", "PART_OF", "YOUR_NEW_RELATION",
        ...
    ]
)
```

## 関連ドキュメント

- [.env.sample](.env.sample): 環境変数の設定例（全オプション記載）
- [docs/langfuse_plan.md](docs/langfuse_plan.md): Langfuse統合の設計ドキュメント

---

**Built with LangChain, Neo4j, PGVector, Azure OpenAI & Sudachi**
