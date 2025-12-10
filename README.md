# Graph-RAG with Streamlit UI

最新版のLangChain APIに対応した、Graph-RAGシステムの完全実装です。
**Streamlit UI**を搭載し、PDF/テキストファイルから自動的にナレッジグラフを構築・可視化できます。

![Graph-RAG Demo](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 主な機能

### ✨ Streamlit WebUI
- 📁 **PDF/テキストファイルアップロード**: ドラッグ&ドロップで簡単アップロード（PyMuPDF対応）
- 🚀 **ワンクリックでナレッジグラフ構築**: LLMが自動的にエンティティとリレーションを抽出
- 🔄 **既存グラフの自動検出**: Neo4jに既にデータがあれば即座に復元
- 💬 **対話型質問応答**: Graph-First RAGによる高精度な回答生成
- 🎯 **LLMリランキング**: 質問の意図に応じた関係性フィルタリング（~¥0.6/クエリ）

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
質問 → エンティティ抽出 → 1-hop双方向グラフ検索 (30件)
                              ↓
                        LLMリランキング (15件)
                              ↓
                    関連チャンク取得 (Neo4j)
                              ↓
                    フォールバック: Vector検索
                              ↓
                        Context Merge
                              ↓
                        Azure OpenAI → 回答生成
```

**精度向上の取り組み**:
- ✅ **Phase 1**: 無向→有向トラバーサル（逆向きパス除外）
- ✅ **Phase 1.5**: 2-hop可変長→1-hop直接関係（中間ノード除外）
- ✅ **Phase 4**: LLMリランキング（質問の意図を考慮）

### 🔧 技術スタック
- **LLMGraphTransformer**: カスタムプロンプト対応（ChatPromptTemplate）
- **グラフバックエンド**:
  - **NetworkX**: 軽量インメモリグラフ（pickle永続化）
  - **Neo4j**: 本格グラフデータベース（オプション）
- **PGVector**: ベクトルデータベース（PostgreSQL）
- **Azure OpenAI**: LLMとEmbedding
- **PyMuPDF**: 高精度PDF読み取り（pypdfより2-3倍高速）
- **RecursiveCharacterTextSplitter**: 重複を防ぐ確実なテキスト分割
- **LCEL**: ハイブリッド検索チェイン

## 📋 必要な環境

- Python 3.8以上
- **グラフバックエンド** (いずれか選択):
  - **NetworkX** (デフォルト): Neo4j不要、軽量、小〜中規模データ向け
  - **Neo4j**: 大規模データ、高度なクエリ向け (Neo4j Aura またはローカルインスタンス)
- PostgreSQL with PGVector拡張
- Azure OpenAI API キー

## 🚀 クイックスタート

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env`ファイルをプロジェクトのルートディレクトリに作成:

```env
# Graph Backend Configuration
# "neo4j" または "networkx" を選択
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
```

**グラフバックエンドの選択**:
- **`GRAPH_BACKEND=networkx`** (デフォルト):
  - Neo4jのインストール・設定不要
  - グラフデータは`graph.pkl`に保存
  - 小〜中規模データに最適
  - 1-2 hop近傍探索をサポート

- **`GRAPH_BACKEND=neo4j`**:
  - Neo4j Auraまたはローカルインスタンスが必要
  - 高度なCypherクエリに対応
  - 大規模データ・複雑なクエリに最適

### 3. アプリケーションの起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

## 📖 使い方

### 初回利用時

1. **PDFまたはテキストファイルをアップロード**
   - 複数ファイルの同時アップロード可能
   - サポート形式: PDF (PyMuPDF), TXT

2. **「🚀 ナレッジグラフを構築」をクリック**
   - 自動的にエンティティとリレーションを抽出
   - Neo4jとPGVectorに保存

3. **質問を入力**
   - グラフとベクトル検索を組み合わせた高精度な回答
   - LLMリランキングで関連度の高い情報のみを利用

4. **グラフを探索**
   - **🕸️ グラフ可視化**: インタラクティブにノードとエッジを探索
   - **📊 データテーブル**: ノード・エッジ一覧をCSV出力
   - **🔍 Cypherクエリ検索**: 自然言語で質問→Cypher自動生成

### 2回目以降

既にグラフデータがある場合:
- アプリ起動時に自動検出 (Neo4jまたはNetworkX)
- **「🔄 既存グラフを読み込む」をクリック**するだけで即座に利用可能
- 再構築不要！

### バックエンドの切り替え

`.env`ファイルの`GRAPH_BACKEND`を変更するだけ:

```bash
# NetworkXに切り替え (Neo4j不要)
GRAPH_BACKEND=networkx

# Neo4jに切り替え (高度なクエリ用)
GRAPH_BACKEND=neo4j
```

**注意**: バックエンドを切り替えた場合、既存のグラフデータは引き継がれません。必要に応じてデータベースをクリアして再構築してください。

## 🎯 サイドバー機能

### 📊 グラフ可視化設定
- **可視化エンジン選択**: Pyvis (推奨) / Agraph
- **最大表示ノード数**: 50〜500個

## 🔧 高度な機能

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

### 自然言語→Cypherクエリ変換

**例**:
```
入力: 「桃太郎に関するすべての関係を表示して」
↓
自動生成されたCypherクエリ:
MATCH (n {id: "桃太郎"})-[r]-(m)
WHERE type(r) <> 'MENTIONS'
AND NOT n.id =~ '[0-9a-f]{32}'
RETURN n.id AS source, type(r) AS relation, m.id AS target
LIMIT 50
```

## 🏗️ アーキテクチャ

```
PDF/Text → PyMuPDF/TextLoader → SemanticChunker
                                       ↓
                         LLMGraphTransformer (カスタムプロンプト)
                                       ↓
                            ┌──────────┴──────────┐
                            ↓                     ↓
                    Graph Backend            PGVector
                  (Neo4j/NetworkX)         (ベクトルDB)
                            ↓                     ↓
                   1. エンティティ抽出       VectorRetriever
                   2. 1-hop双方向検索              ↓
                   3. LLMリランキング      (フォールバック)
                            └──────────┬──────────┘
                                       ↓
                          関連チャンクをグラフから取得
                                       ↓
                               Context Merge
                                       ↓
                                Azure OpenAI
                                       ↓
                                   回答生成
```

**グラフバックエンド比較**:

| 機能 | NetworkX | Neo4j |
|------|----------|-------|
| セットアップ | 不要（すぐ使える） | Neo4j Aura/Server必要 |
| データ規模 | 小〜中規模 | 大規模対応 |
| 永続化 | pickle/JSON | サーバーDB |
| クエリ | 1-2 hop探索 | 高度なCypher |
| パフォーマンス | メモリ依存 | スケーラブル |
| コスト | 無料 | Aura有料/自前サーバー |

## 🔧 技術的な詳細

### Neo4j Cypherクエリ

Neo4j 5.x以降の新しい構文に対応:
```cypher
MATCH (n)-[r]-(m)
WHERE type(r) <> 'MENTIONS'
AND NOT n.id =~ '[0-9a-f]{32}'
AND NOT m.id =~ '[0-9a-f]{32}'
WITH n, r, m, labels(n) as source_labels, labels(m) as target_labels
RETURN
  n.id AS source,
  type(r) AS relation,
  m.id AS target,
  COUNT { (n)--() } AS source_degree,
  COUNT { (m)--() } AS target_degree
LIMIT 200
```

### PyMuPDF高精度抽出

```python
pdf_doc = fitz.open(tmp_path)
for page in pdf_doc:
    text = page.get_text("text", sort=True)  # レイアウト保持・ソート
```

**メリット**:
- pypdfより2-3倍高速
- 複雑なレイアウト対応
- 日本語PDF高精度読み取り

### 可視化の特徴
- **ノードサイズ**: 接続数に応じて自動調整（Pyvis: 最小12、最大30）
- **カラーコード**: ノードタイプ別に色分け
- **物理エンジン**: Pyvisの高度な物理演算

## 📦 主要な依存ライブラリ

```
streamlit>=1.28.0
langchain-core>=0.1.0
langchain-openai>=0.1.0
langchain-experimental>=0.1.0
neo4j>=5.20.0
pymupdf>=1.24.0
streamlit-agraph>=0.0.45
pyvis>=0.3.2
networkx>=3.0
psycopg[binary]>=3.1.19
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

### PyMuPDF関連エラー
- `pymupdf>=1.24.0`がインストールされているか確認
  ```bash
  pip install pymupdf>=1.24.0
  ```

### LLMリランキングのコスト
- デフォルトで30件→15件に絞り込み
- 約¥0.6/クエリ（Azure OpenAI GPT-4使用時）
- `top_k`パラメータで調整可能

## 💡 カスタマイズ

### プロンプトのカスタマイズ

`app.py`の`kg_system_prompt`を編集することで、抽出ルールをカスタマイズできます。

```python
kg_system_prompt = """
あなたはテキストから専門用語とその関係性を抽出する専門家です。
...
"""
```

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

- [課題メモ.md](課題メモ.md): 実装の詳細とトラブルシューティング
- [.env.sample](.env.sample): 環境変数の設定例

---

**⚡ Built with LangChain, Neo4j, PGVector, PyMuPDF & Azure OpenAI**
