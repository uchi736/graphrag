# Graph-RAG with Streamlit UI

最新版のLangChain APIに対応した、Graph-RAGシステムの完全実装です。
**Streamlit UI**を搭載し、PDF/テキストファイルから自動的にナレッジグラフを構築・可視化できます。

![Graph-RAG Demo](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 主な機能

### ✨ Streamlit WebUI
- 📁 **PDF/テキスト/Markdownファイルアップロード**: ドラッグ&ドロップで簡単アップロード
  - **Azure Document Intelligence**: 高精度PDF解析（テーブル・レイアウト保持、OCR対応）
  - **PyMuPDF**: 軽量PDF解析（フォールバック）
  - **Azure DI出力キャッシュ**: 処理結果を.mdファイルとして保存、再利用可能
- 🚀 **ワンクリックでナレッジグラフ構築**: LLMが自動的にエンティティとリレーションを抽出
- 🔄 **既存グラフの自動検出**: Neo4j/NetworkXに既にデータがあれば即座に復元
- ▶️ **再開機能**: 処理が中断しても続きから再開可能（チャンクハッシュで管理）

### 🖥️ CLIツール
- **build_kg.py**: Streamlitを使わずにナレッジグラフを構築
  - フォルダ指定で複数ファイルを一括処理
  - 長時間処理でもタイムアウトなし
  - 再開機能対応
- 💬 **対話型質問応答**: Graph-First RAGによる高精度な回答生成
- 🎯 **LLMリランキング**: 質問の意図に応じた関係性フィルタリング
- 📥 **CSVエッジインポート**: 外部関係データの取り込み対応
- ✏️ **グラフ編集機能**: ノード・エッジの追加・編集・削除

### 🔍 高度な検索機能
- **日本語ハイブリッド検索**: Sudachiによる形態素解析 + RRFスコア統合
- **エンティティベクトル検索**: エンティティの類似度検索（類義語・関連語対応）
- **可変ホップ検索**: 1〜3ホップのグラフ探索
- **クロスドキュメント推論**: 共通エンティティを持つドキュメント間の自動リンク

### 🎨 高度なグラフ可視化
- **2つの可視化エンジン**:
  - **Pyvis** (推奨): 高度な物理演算とリッチなビジュアル
  - **Streamlit-Agraph**: 軽量でシンプル
- **3つの表示モード**:
  - 🕸️ **グラフ可視化**: インタラクティブな探索
  - 📊 **データテーブル**: ノード・エッジ一覧とCSV出力
  - 🔍 **Cypherクエリ検索**: 自然言語→Cypher変換機能
- **ノードタイプ自動判定**: 専門用語を自動分類
- **タイプ別色分け**: 視覚的にわかりやすいカラーコーディング
- **接続数に応じたノードサイズ**: 重要なエンティティが一目でわかる

### 🧠 Graph-First RAGアーキテクチャ
```
質問 → エンティティ抽出 → エンティティベクトル検索
                              ↓
                        1-3 hop グラフ検索
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

### 🔧 技術スタック
- **LLMGraphTransformer**: カスタムプロンプト対応（ChatPromptTemplate）
- **グラフバックエンド**:
  - **NetworkX**: 軽量インメモリグラフ（pickle永続化）
  - **Neo4j**: 本格グラフデータベース（オプション）
- **PGVector**: ベクトルデータベース（PostgreSQL）
- **LLMプロバイダー**:
  - **Azure OpenAI**: GPT-4o, text-embedding-3-small
  - **VLLM**: セルフホステッドモデル対応
- **PDF処理**:
  - **Azure Document Intelligence**: 高精度PDF解析
  - **PyMuPDF**: 軽量PDF解析
- **日本語処理**: Sudachi形態素解析
- **RecursiveCharacterTextSplitter**: 重複を防ぐ確実なテキスト分割
- **LCEL**: ハイブリッド検索チェイン

## 📁 ファイル構成

```
graphrag/
├── app.py                      # メインアプリケーション（Streamlit UI）
├── build_kg.py                 # CLIナレッジグラフ構築ツール
├── llm_factory.py              # LLMプロバイダー切り替え（Azure/VLLM）
├── graphrag.py                 # Graph-RAG コア検索ロジック
├── networkx_graph.py           # NetworkXバックエンド実装
├── hybrid_retriever.py         # 日本語ハイブリッド検索（RRF統合）
├── japanese_text_processor.py  # Sudachi形態素解析
├── entity_vectorizer.py        # エンティティベクトル化
├── azure_di_processor.py       # Azure Document Intelligence処理
├── db_utils.py                 # PostgreSQL接続・インデックス管理
├── test_vllm.py                # VLLM接続テスト
├── init_japanese_search.py     # 日本語検索初期化スクリプト
├── reset_pgvector_tables.py    # PGVectorテーブルリセット
├── migrate_to_jsonb.sql        # JSONB移行SQL
├── requirements.txt            # Python依存関係
├── .env.sample                 # 環境変数テンプレート
├── README.md                   # メインドキュメント
├── VLLM_Integration_Guide.md   # VLLM統合ガイド
├── graph.pkl                   # NetworkXグラフデータ（自動生成）
├── graph.json                  # グラフデータJSON形式（自動生成）
└── output/                     # Azure DI処理結果（.md形式、自動生成）
```

### 📄 主要ファイル詳細

| ファイル | 説明 |
|---------|------|
| `app.py` | Streamlit UI、グラフ構築、質問応答、可視化すべてを統合 |
| `build_kg.py` | CLIナレッジグラフ構築（フォルダ一括処理、再開機能） |
| `llm_factory.py` | Azure OpenAI / VLLM を環境変数で切り替え |
| `graphrag.py` | エンティティ抽出→グラフ検索→コンテキスト構築 |
| `networkx_graph.py` | Neo4j互換APIをNetworkXで実装、処理済みハッシュ管理 |
| `hybrid_retriever.py` | ベクトル検索 + キーワード検索をRRFで統合 |
| `japanese_text_processor.py` | Sudachi + トークン正規化 |
| `entity_vectorizer.py` | エンティティ名をベクトル化してPGVectorに保存 |
| `azure_di_processor.py` | PDF→Markdown変換（テーブル・OCR対応） |
| `db_utils.py` | 接続文字列正規化、インデックス作成 |

## 📋 必要な環境

- Python 3.8以上
- **グラフバックエンド** (いずれか選択):
  - **NetworkX** (デフォルト): Neo4j不要、軽量、小〜中規模データ向け
  - **Neo4j**: 大規模データ、高度なクエリ向け (Neo4j Aura またはローカルインスタンス)
- PostgreSQL with PGVector拡張
- Azure OpenAI API キー（またはVLLMサーバー）

## 🚀 クイックスタート

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env`ファイルをプロジェクトのルートディレクトリに作成（詳細は `.env.sample` 参照）:

```env
# Graph Backend Configuration
GRAPH_BACKEND=networkx

# Neo4j Configuration (GRAPH_BACKEND=neo4j の場合のみ必要)
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PW=your_neo4j_password

# PostgreSQL Configuration
PG_CONN=postgresql+psycopg://postgres:your_password@your-host:5432/postgres

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
GRAPH_HOP_COUNT=1

# LLM Provider (azure_openai or vllm)
LLM_PROVIDER=azure_openai

# VLLM Configuration (LLM_PROVIDER=vllm の場合)
VLLM_ENDPOINT=https://your-vllm-server.example.com/v1
VLLM_TEMPERATURE=0.0
VLLM_MAX_TOKENS=4096
```

### 3. アプリケーションの起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

## 📖 使い方

### 初回利用時

1. **PDFまたはテキストファイルをアップロード**
   - 複数ファイルの同時アップロード可能
   - サポート形式: PDF, TXT, MD
   - オプション: edges.csv（関係データのインポート）

2. **「🚀 ナレッジグラフを構築」をクリック**
   - 自動的にエンティティとリレーションを抽出
   - グラフバックエンドとPGVectorに保存

3. **質問を入力**
   - グラフとベクトル検索を組み合わせた高精度な回答
   - LLMリランキングで関連度の高い情報のみを利用

4. **グラフを探索**
   - **🕸️ グラフ可視化**: インタラクティブにノードとエッジを探索
   - **📊 データテーブル**: ノード・エッジ一覧をCSV出力、編集
   - **🔍 Cypherクエリ検索**: 自然言語で質問→Cypher自動生成

### 2回目以降

既にグラフデータがある場合:
- アプリ起動時に自動検出 (Neo4jまたはNetworkX)
- **「🔄 既存グラフを読み込む」をクリック**するだけで即座に利用可能
- 再構築不要！

### 再開機能

大量のドキュメントを処理する場合、途中で中断しても続きから再開できます:

**Streamlit UI**:
- **「🚀 新規構築」**: 処理済みデータをクリアして最初から構築
- **「▶️ 続きから再開」**: 未処理のチャンクのみ処理

**仕組み**:
- 各チャンクにSHA256ハッシュIDを付与
- 処理済みハッシュをgraph.pklに保存
- 再開時は処理済みハッシュと照合してスキップ

### CLI版 (build_kg.py)

Streamlitのセッションタイムアウトを回避したい場合や、大量のドキュメントを処理する場合はCLI版を使用:

```bash
# フォルダ内の全ファイルを処理
python build_kg.py --input ./docs

# 新規構築（処理済みをクリア）
python build_kg.py --input ./docs --fresh

# 特定の拡張子のみ
python build_kg.py --input ./docs --ext pdf,md
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

## 🎯 サイドバー機能

### 📊 グラフ可視化設定
- **可視化エンジン選択**: Pyvis (推奨) / Agraph
- **最大表示ノード数**: 50〜500個

### 🔍 検索設定
- **TopK**: 検索で取得するチャンク数（1〜20）
- **ホップ数**: グラフ探索の深さ（1〜3）
- **エンティティベクトル検索**: ON/OFF、類似度閾値
- **日本語ハイブリッド検索**: ハイブリッド/ベクトル/キーワード

## 🔧 高度な機能

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

### クロスドキュメント推論

複数ドキュメント間の関連性を自動検出:
- 共通エンティティを2つ以上持つドキュメント間に`SHARES_TOPICS_WITH`リレーションを作成
- Neo4j / NetworkX両対応

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

**メリット**:
- ノイズの多いグラフでも高精度な回答
- コンテキストサイズの削減でコスト・レイテンシ改善
- 使用モデル: `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`（回答生成と同じ）

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

## 🏗️ アーキテクチャ

```
PDF/Text → Azure DI / PyMuPDF → RecursiveCharacterTextSplitter
                                       ↓
                         LLMGraphTransformer (カスタムプロンプト)
                                       ↓
                            ┌──────────┴──────────┐
                            ↓                     ↓
                    Graph Backend            PGVector
                  (Neo4j/NetworkX)         (ベクトルDB)
                            ↓                     ↓
              クロスドキュメント推論      エンティティベクトル化
                            ↓                     ↓
                            └──────────┬──────────┘
                                       ↓
                              質問応答フロー
                                       ↓
                   エンティティ抽出 + ベクトル検索
                                       ↓
                          1-3 hop グラフ探索
                                       ↓
                             LLMリランキング
                                       ↓
                    関連チャンク取得 (MENTIONS)
                                       ↓
               フォールバック: 日本語ハイブリッド検索
                                       ↓
                        Context Merge + 回答生成
```

**グラフバックエンド比較**:

| 機能 | NetworkX | Neo4j |
|------|----------|-------|
| セットアップ | 不要（すぐ使える） | Neo4j Aura/Server必要 |
| データ規模 | 小〜中規模 | 大規模対応 |
| 永続化 | pickle/JSON | サーバーDB |
| クエリ | 1-3 hop探索 | 高度なCypher |
| クロスドキュメント推論 | ✅ 対応 | ✅ 対応 |
| グラフ編集 | ✅ 対応 | ✅ 対応 |
| パフォーマンス | メモリ依存 | スケーラブル |
| コスト | 無料 | Aura有料/自前サーバー |

## 📦 主要な依存ライブラリ

```
streamlit>=1.28.0
langchain-core>=0.1.0
langchain-openai>=0.1.0
langchain-experimental>=0.1.0
langchain-postgres>=0.0.14
neo4j>=5.20.0
pymupdf>=1.24.0
azure-ai-documentintelligence>=1.0.0
streamlit-agraph>=0.0.45
pyvis>=0.3.2
networkx>=3.0
psycopg[binary]>=3.1.19
sudachipy>=0.6.8
sudachidict_core>=20240716
```

## 🐛 トラブルシューティング

### グラフが表示されない
- `pyvis`または`streamlit-agraph`がインストールされているか確認
- ブラウザのJavaScriptが有効か確認

### Neo4j接続エラー (GRAPH_BACKEND=neo4j の場合)
- `.env`の`NEO4J_URI`が正しいか確認
- Neo4j Auraのインスタンスが起動しているか確認
- ファイアウォールでポートが開いているか確認
- **解決策**: NetworkXに切り替えて試す (`GRAPH_BACKEND=networkx`)

### NetworkXでのデータ読み込みエラー
- `graph.pkl`ファイルが破損していないか確認
- サイドバーから「データベースをクリア」を実行
- 再度ドキュメントをアップロードして構築

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

## 💡 カスタマイズ

### プロンプトのカスタマイズ

`app.py`の`kg_system_prompt`を編集することで、抽出ルールをカスタマイズできます。

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

## 🤝 コントリビューション

プルリクエストを歓迎します！

1. このリポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 📚 関連ドキュメント

- [.env.sample](.env.sample): 環境変数の設定例（全オプション記載）

---

**⚡ Built with LangChain, Neo4j/NetworkX, PGVector, Azure OpenAI & Sudachi**
