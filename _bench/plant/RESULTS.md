# plant_v15 評価結果

IHI技報＋ボイラ/高圧ガス安全法規ドメインの KG-QA データセット（makedataset 由来）で
graphrag パイプラインを評価。

## データ
- 評価セット: `C:/work/makedataset/data/reviewed/plant_v15.jsonl`（25問、review_status=accepted）
- 元データ: `C:/work/makedataset/data/chunks_plant/*.jsonl`（158文書 / 3,175チャンク、チャンク済み）
- 参照粒度: doc_id（page は null）
- 投入先: PGVector コレクション `plant_v15`（ruri-v3-310m 768d、BM25トークン化済み、他コレクションと分離）

## PLANT-01: KG-off baseline（hybrid + cross-encoder rerank、KGなし）

| 指標 | 値 |
|---|---|
| 回答 accuracy（Azure gpt-4.1-mini judge） | **72.0%** (18/25) |
| 参照 full-coverage（doc_id） | **96.0%** (24/25) |
| 参照 match-rate | 98.0% |

条件: top_k=10, fetch_k=20, search=hybrid, rerank=cross-encoder(bge-reranker-v2-m3), LLM=Azure gpt-4.1-mini

### 軸別（回答 accuracy）
| 軸 | 値 |
|---|---|
| retrieval_level=Easy | 80.0% (4/5) |
| retrieval_level=Medium | 75.0% (6/8) |
| retrieval_level=Hard | 66.7% (8/12) |
| answer_level=Easy | 88.9% (8/9) |
| answer_level=Medium | 62.5% (10/16) |
| **kg_query_type=traceability** | **40.0% (2/5)** ← KGの主戦場 |

### 所見
- **検索（参照カバレッジ）は非常に高い（96%）** — コーパス投入とハイブリッド検索は当ドメインで良好に機能
- 回答72%の取りこぼしは Hard / answer_Medium / **traceability**（追跡型）に集中。traceability 40% は
  「文書間の参照を辿る」型で、まさに KG（参照グラフ）が補完しうる領域
- 回答が実質拒否だったのは 3/25 のみ（過剰拒否は限定的）

### 次の比較候補
traceability/Hard が低いことから、KG-on（plant コーパスでKG構築 → グラフ探索 + 参照追跡）で
これらが伸びるかが論点。ただし Neo4j に現在 Fujitsu graph があるため破壊的（要判断）。

## PLANT-02: 生成プロンプト改修（2026-06-13）

失敗診断（[condition_diagnosis.md](condition_diagnosis.md) + 6失敗の並列深掘りワークフロー）で、
plantの実損失は **KG/reification/参照グラフではなく生成プロンプト側** と確定（全6失敗で検索成功・
根拠は文脈内）。失敗類型: 否定/例外の取りこぼし・捏造（コーパスに無い別表/JIS番号）・条件分岐の
取り違え・準用拒否（文脈に条文があるのに「規定が無い」と回答放棄）・準拠先/様式の取り違え。

プロンプトトーナメント（3候補×実ベンチ25問×Azure judge）:

| プロンプト | accuracy | 回復(6失敗中) | 退行 | net |
|---|---|---|---|---|
| baseline（簡潔4行ルール） | 72% (18/25) | - | - | - |
| v1 チェックリスト型 | 88% | 5 | 1 | +4 |
| v2 接地・反捏造優先 | 88% | 4 | 0 | +4 |
| v3 構造化3ステップ | 88% | 4 | 0 | +4 |
| **graft（v3構造+v2反捏造）★採用** | **92% (23/25)** | **5** | **0** | **+5** |

graft = `variant_graft.txt` → run_plant.py の GEN_PROMPT に確定反映。アプリ経路の
`graphrag_core/prompts.py` QA_PROMPT にも汎用テーマ（反捏造・否定/例外網羅・条件分岐ガード・
準用採用・主題固定）を追記。**KG/Neo4j非依存、+20pt をプロンプトのみで達成。**

残り2失敗はプロンプト対象外:
- auto_907515d4: 02_boiler_anzen を要するクロスドキュメントmulti-hop（参照グラフ領分）
- auto_5e0e245c: 45_tokutei_setsubi を検索が取れない検索ミス（retriever領分、条件問ですらない）

教訓: 高出現率（条件56.6%）はKG表現課題の存在証明でしかなく、実損失の主因は生成だった。
診断（失敗モード分類＋コーパス再帰性）→トーナメント→graft、の順で投資先を実測特定した。
（→ メモリ feedback_representation_bottleneck.md）

## 運用メモ
- gemma judge（DGX @ 192.168.0.250:8000）は実行時点で停止中 → Azure judge を使用。
- factory の AzureChatOpenAI は timeout 未設定（langchainデフォルト=無限待ち）。
  並列runnerでハングしうるため、runner/judge側で request_timeout を設定している。
- 再現:
  ```
  python _bench/plant/ingest_plant.py --collection plant_v15 --fresh
  python _bench/plant/run_plant.py --top-k 10 --fetch-k 20 --concurrency 2
  python _bench/plant/eval_plant.py --pred _bench/plant/pred_plant_retrieval.json --judge-backend azure
  ```
