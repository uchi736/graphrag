# EDC 連携（スキーマ自動発見）

graphrag のKG構築は、許可ノードタイプ・関係タイプを外部スキーマJSON
（`.env` の `SHARED_SCHEMA_PATH`）で差し替えられる。
このスキーマの手キュレーションを **EDC (Extract-Define-Canonicalize,
[arXiv:2404.03868](https://arxiv.org/abs/2404.03868)) によるスキーマ自動発見**で置き換えるのが本連携。

```
コーパスのサンプル文書
  → EDC API /extract (doctype=auto, enrich_schema=True)   ... 関係・型を発見/定義/正規化
  → graphrag形式スキーマJSON（node_types + relations）
  → SHARED_SCHEMA_PATH に指定して build_kg
  → 構築時に Neo4j へ :SchemaMeta として刻印（どのスキーマで作ったかの証跡）
```

## EDC 本体のセットアップ（別リポジトリ）

EDC は独立したプログラムとして https://github.com/uchi736/EDC で管理している。

```bash
git clone https://github.com/uchi736/EDC.git
cd EDC/edc
# 環境構築は EDC リポジトリの README 参照（environment.yml / Dockerfile あり）
python -m uvicorn api:app --port 8080
```

- ヘルスチェック: `GET http://127.0.0.1:8080/health`
- graphrag 側からのエンドポイント指定: 環境変数 `EDC_ENDPOINT`（既定 `http://127.0.0.1:8080`）

### API仕様の罠

`/extract` レスポンスの関係スキーマのキーは **`schema_`**（末尾アンダースコア。
pydantic の `BaseModel.schema` シャドウ回避のため）。`schema` では取れない。

## スキーマ同期の実行方法（2通り）

### A. UI から（構築/取り込みタブ → KGスキーマカード → 「EDCスキーマ同期」）

現在の PGVector コレクションから文書サンプル（既定: 4文書×6チャンク）を取り、
`schemas/edc_<collection>.json` に書き出すジョブが走る。
実装: `graphrag_core/services/schema_sync.py` / `POST /api/build/edc-sync`

### B. CLI から（チャンクディレクトリを直接サンプリング）

```bash
python scripts/edc_schema_sync.py \
    --chunks-dir C:/work/makedataset/data/chunks_synth \
    --domain synth_v1 --out _bench/edc_schema_synth.json \
    --docs 4 --chunks-per-doc 6
```

## 生成したスキーマでKGを構築

```bash
SHARED_SCHEMA_PATH=schemas/edc_<collection>.json python scripts/build_kg.py ...
# または .env の SHARED_SCHEMA_PATH を書き換える
```

構築時に `graphrag_core/graph/schema.py` が JSON を読み、
`LLMGraphTransformer` の `allowed_nodes` / `allowed_relationships` に注入し、
Neo4j に `:SchemaMeta {kind:'active'}` を刻印する。

## スキーマの確認・不一致警告

- UI: 構築タブの「KGスキーマ」カードが **アクティブ（現グラフ刻印）** と
  **次回ビルド設定（SHARED_SCHEMA_PATH）** を並べて表示し、不一致なら警告する。
  （`.env` が別スキーマを指したまま再構築して黙ってスキーマが変わる事故の防止）
- API: `GET /api/graph/schema`

## 既知の運用ノート

- EDC発見スキーマは Cause / Defect / Norm など**命題的タイプ**を含むため、
  LLMが文まるごとをノードIDにすることがある（是正事例系で顕著）。
  結合キーとして機能しないので、気になる場合は `node_type_definitions` に
  「短い名詞句で命名」等の指示を追記してから構築する。
- 生成済みスキーマの実例: [_bench/edc_schema_synth.json](../_bench/edc_schema_synth.json)
  （synth_v1 用、22タイプ・31関係）
