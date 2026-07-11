# 共有グラフ検証ゲート 実行手順（ステージ1-5）

llm-graph-builder + EDC が構築したグラフの上で graphrag の Graph-First 検索が
品質を出せるかを、GB本体に手を入れずに判定する手順。
**この検証を通過してからステージ2（一本化本体）に着手する。**

## 前提（GB側・1回だけ）

1. GB backend を再起動して今回の変更を反映:
   ```
   cd "C:\work\RAG\Graph Builder\llm-graph-builder"   # ※composeを動かしているホストで
   docker compose up -d backend
   ```
   反映される内容: ruri-v3-310m 埋め込み（vllmプロバイダ）/ ENTITY_EMBEDDING=true /
   共存ハードニング / GRAPH_SCHEMA_CONSOLIDATION_ENABLED=false
2. UIで再接続 → ベクトル索引の次元不一致アラートが出たら索引を再作成（384→768）
3. **評価用文書セットを取り込み**（モデルは edc 推奨。gemma4 でも可）
4. UIから **後処理を実行**（フルテキスト索引・SIMILAR・エンティティ埋め込み）

## 手順（graphrag側・このリポジトリで）

### 1. enrichバッチ（派生プロパティ付与 + provenance刻印）

```powershell
cd C:\work\RAG\graphrag
# まず対象件数の確認（書き込みなし）
.\myenv\Scripts\python.exe scripts\enrich_external_graph.py --dry-run `
    --neo4j-uri neo4j://192.168.0.250:7688 --neo4j-user neo4j --neo4j-pw <GB側のパスワード> `
    --pg-collection gb_mirror
# 本実行
.\myenv\Scripts\python.exe scripts\enrich_external_graph.py `
    --neo4j-uri neo4j://192.168.0.250:7688 --neo4j-user neo4j --neo4j-pw <GB側のパスワード> `
    --pg-collection gb_mirror
```

### 2. PGVectorミラー同期（チャンク本文+埋め込みコピー+Sudachiトークン化）

```powershell
.\myenv\Scripts\python.exe scripts\mirror_gb_chunks.py `
    --neo4j-uri neo4j://192.168.0.250:7688 --neo4j-user neo4j --neo4j-pw <GB側のパスワード> `
    --pg-collection gb_mirror
```

### 3. batch_eval を共有グラフ向け設定で実行

`questions.csv`（question 列必須）を用意して:

```powershell
$env:KG_CHUNK_LABEL  = "Chunk"
$env:KG_CHUNK_EDGE   = "HAS_ENTITY"
$env:NEO4J_URI       = "neo4j://192.168.0.250:7688"
$env:NEO4J_USER      = "neo4j"
$env:NEO4J_PW        = "<GB側のパスワード>"
$env:PG_COLLECTION   = "gb_mirror"
.\myenv\Scripts\python.exe scripts\batch_eval.py --input questions.csv --output results_shared_graph.csv
```

### 4. 対照群: GB既存モードで同じ質問

GB UI（graph+vector モード）で同じ質問を実行、または `/chat_bot` API を直接叩いて回答を収集。

## 判定ポイント

| 観点 | 見るところ |
|---|---|
| 回答品質 | results_shared_graph.csv の回答 vs GB graph_vector の回答（正確性・根拠の妥当性） |
| KGゲート | ログに `provenance_mismatch` / `graph_not_connected` が出ていないか（出たら手順1のprovenance刻印を確認） |
| パス取得 | results の graph_paths / KGチャンク列が空でないか（空ならenrichの search_keys / pagerank を確認） |
| 混在語彙 | gemma4英語関係名とEDC日本語関係名が混ざった文書セットでリランクが破綻しないか |
| GB側の無傷確認 | GB UIで可視化・名寄せ・スキーマパネルに SchemaMeta / GraphProvenance 等が露出していないか |

## 運用順序（以後の定常運用）

GBで文書を追加/削除/再処理するたびに:
`GB後処理 → enrich_external_graph.py → mirror_gb_chunks.py` の順で実行
（ステージ2でGBの後処理から自動実行に配線予定）
