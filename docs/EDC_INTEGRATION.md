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

## セットアップ = 不要（graphragの一機能として内蔵実行）

**EDC 一式はこのリポジトリに同梱しており（[vendor/EDC/](../vendor/EDC/)）、既定では
サーバも venv も .env も不要。** スキーマ同期ジョブが `vendor/EDC/extract_cli.py` を
子プロセスとしてワンショット実行する:

- 依存は graphrag の `requirements.txt` の部分集合（実測済み）→ 追加インストールなし
- LLM/embedding の接続先（`VLLM_*`）は graphrag の `.env` から自動注入
- cwd=vendor/EDC で起動するため、EDC側のテンプレート/スキーマ解決も無設定で動く

### 外部EDCサーバを使う場合（オプション）

Dify等と共用のEDCを別プロセス/別マシンで運用する場合のみ、`EDC_ENDPOINT` を設定する
（設定時はHTTP経路に切り替わる）:

```bash
cd vendor/EDC
python -m uvicorn api:app --host 0.0.0.0 --port 8080   # graphragと同じvenvで動く
# graphrag側: EDC_ENDPOINT=http://<host>:8080
```

- EDC自体の開発は上流リポジトリ https://github.com/uchi736/EDC で行い、
  [vendor/EDC/VENDOR.md](../vendor/EDC/VENDOR.md) の手順で再取り込みする（同梱分は読み取り専用スナップショット）

### API仕様の罠

`/extract` レスポンスの関係スキーマのキーは **`schema_`**（末尾アンダースコア。
pydantic の `BaseModel.schema` シャドウ回避のため）。`schema` では取れない。

## スキーマ同期の実行方法（2通り）

### A. UI から（構築/取り込みタブ → KGスキーマカード → 「EDCスキーマ同期」）

現在の PGVector コレクションから文書サンプル（既定: 4文書×6チャンク）を取り、
`schemas/edc_<collection>.json` に書き出すジョブが走る。
実装: `graphrag_core/services/schema_sync.py` / `POST /api/build/edc-sync`

同期後は同カードの **「編集」で人手キュレーション**できる（発見された
タイプ/関係のチェックを外して削除・追加・保存。`.bak` を残して上書きし、
次回ビルドから有効）。API: `GET/PUT /api/graph/schema/file`

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
