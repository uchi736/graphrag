# Experiments — Fujitsu RAG Hard Benchmark

評価対象: `neoai-inc/Japanese-RAG-Generator-Benchmark` (J-RAGBench, 114問) と
`FujitsuResearch/Fujitsu-RAG-Hard-Benchmark` (FJ-Hard, 100問)。

オンプレ環境: gemma-4-26B-A4B-it (8000) / ruri-v3-310m (8003) / bge-reranker-v2-m3 (8006)
+ Neo4j + PGVector (DGX Spark @ 192.168.0.250)

---

## 環境

| 項目 | 値 |
|---|---|
| LLM (Generator + Judge) | google/gemma-4-26B-A4B-it (FP8) |
| Embedding | cl-nagoya/ruri-v3-310m (768d, 1024tok max) |
| Reranker (cross-encoder) | BAAI/bge-reranker-v2-m3 |
| Vector store | PGVector (langchain_postgres) |
| Graph store | Neo4j (langchain_neo4j) |
| Judge bias | self-judge: ⚠️ Generator と Judge が同一モデル |

---

## J-RAGBench (Generator-only, gold context 与える)

クリーン条件で Gemma 4 の Generator としての素性を測る。Retriever は介在しない。

### 全体

| 指標 | スコア |
|---|---|
| Key fact coverage 平均 | 89.94% |
| LLM-judge 平均 (0/1/2) | 1.92 / 2.0 |
| 完全正解率 (score=2) | 95.8% (23/24) |

※ p2-5 限定の縮小評価。コーパスのノイズ・retrieval は介在しない上限値。

### 一方、フル114問 (24 questions/category) の判定

| Category | N | 完全正解率 | Avg | Note |
|---|---|---|---|---|
| Refusal | 54 | 94% | 1.89 | 情報無し時の拒否能力は強い |
| Integration | 25 | 72% | 1.52 | 2-3文書統合は概ねOK |
| Table | 8 | 62% | 1.38 | マークダウン表読解は普通 |
| Reasoning | 17 | 59% | 1.24 | **多段推論・計算は弱い** |
| Single | 10 | 60% | 1.20 | 単一文書でも文脈解釈ミス |
| **全体** | **114** | **78.9%** | **1.61** | クリーン条件の上限 |

### 失敗パターン (定性)

1. **Over-refusal** — 情報が文書にあるのに「回答不可」と諦める (Q18, Q23, Q28, Q62, Q92)
2. **Hallucination on refusal cases** — 拒否すべき場面で他文書の知識を流用して捏造 (Q6, Q72, Q101)
3. **矛盾検出失敗** — 文書内の矛盾を見逃して片方を採用 (Q31, Q83)
4. **計算ミス** — 単純な乗算でも間違う (Q12: 1320円 → 495円)

---

## Fujitsu RAG Hard Benchmark (E2E, 自前 retrieval)

実コーパス + retrieval 込みでウチのRAG全体を測る。

### コーパス

| 項目 | 値 |
|---|---|
| PDF 数 | 34本 (付属11 + DL_URL 23) |
| 総ページ | 1,794 |
| 総文字 | 2,449,359 |
| チャンク数 (page-aware, max=1000char) | 8,452 |
| 質問数 | 100 |

### 実験条件と結果

| # | 条件 | success | 回答評価 | 参照 full-cov | 参照 match-rate | Note |
|---|---|---|---|---|---|---|
| ① | hybrid only (BM25+vector, top-5) | 98/100 | 29.59% | 51.02% | 65.43% | 旧chunks ベースライン |
| ② | + cross-encoder rerank (fetch=30 → top=5) | 100/100 | 39.00% | 52.00% | 63.26% | +9.4pt |
| ③ | + cross-encoder rerank (fetch=10 → top=5) | 99/100 | **41.41%** | 50.51% | 63.06% | 旧chunks best, +11.8pt |
| ④ | + KG 12関係 (hybrid + KG path + KG source chunks) | 100/100 | 35.00% | 52.00% | **66.13%** | overall -6.4pt vs ③ |
| ⑤ | hybrid (markdown chunks: pp_optimizer + PyMuPDF fallback) | 99/100 | **36.36%** | 48.48% | 61.95% | **+6.8pt** vs ①: markdownが効く |
| ⑥ | + rerank10 (markdown chunks) | 97/100 | 40.21% | 48.45% | 62.80% | -1.2pt vs ③: 微減 |
| ⑦ | + KG 12関係 (markdown chunks) | 100/100 | 31.00% | 49.00% | 62.33% | **-4.0pt** vs ④: KGが弱まる |

ベース ③ (rerank10→5, 旧chunks) との差分:

| 設定 | Δ vs rerank10 (overall) |
|---|---|
| hybrid only (旧) | -11.8 pt |
| rerank 30→5 (旧) | -2.4 pt |
| KG 12関係 (旧) | -6.4 pt |
| hybrid (md) | -5.1 pt |
| rerank10 (md) | -1.2 pt |
| KG 12関係 (md) | **-10.4 pt** |

### chunking 効果の分離 (FJH-04 vs FJH-04-md)

| 軸 | 旧chunks (PyMuPDF) | markdown chunks | Δ |
|---|---|---|---|
| hybrid 単体 | 29.6% | **36.4%** | **+6.8** ⭐ |
| rerank10 | 41.4% | 40.2% | -1.2 |
| KG 12関係 | 35.0% | 31.0% | **-4.0** |

**観察**:
1. **hybrid 単体では markdown が大きく効く (+6.8pt)** — LLMが構造化されたcontextを読みやすい。これは "KG無しでも GraphRAG級の改善" を意味する
2. **rerank ではほぼ差なし (-1.2pt)** — cross-encoder が順序最適化するので chunking粒度の効果が薄まる
3. **KGでは逆に悪化 (-4pt)** — 予想外。原因推測:
   - 8,452 → 3,918 chunks (54%削減) で Term抽出が不足
   - markdown chunk は意味的にまとまっているが、KG の細粒度概念抽出には粗い
   - Term数 34,199 → 14,074 (59%減) で KG 探索の recall が落ちる

### カテゴリ別 (KG_md vs KG_old)

KG_md が勝った場面:
| カテゴリ | N | KG_old | KG_md | Δ |
|---|---|---|---|---|
| tables_charts=False | 30 | 50.0% | **56.7%** | +6.7 |
| low_locality=False | 45 | 24.4% | 28.9% | +4.4 |

KG_md が大きく負けた場面:
| カテゴリ | N | KG_old | KG_md | Δ |
|---|---|---|---|---|
| negation=True | 11 | 36.4% | 9.1% | **-27.3** |
| remote_reference=True | 26 | 26.9% | 15.4% | **-11.5** |
| low_locality=True | 55 | 43.6% | 32.7% | **-10.9** |
| temporal=True | 40 | 32.5% | 22.5% | -10.0 |
| cause_effect=True | 22 | 40.9% | 31.8% | -9.1 |
| tables_charts=True | 70 | 28.6% | 20.0% | **-8.6** |

**重要な解釈**: markdown chunking + KG の組み合わせは「**細粒度な根拠が必要な複雑質問** (negation, remote_reference, low_locality, temporal)」で弱い。理由は粗いchunkがKG抽出時の「単語単位の関係」を失わせるから。

逆に「**シンプルなページ参照型** (low_locality=False, tables_charts=False)」では KG_md が活躍。

### カテゴリ別の差分 (KG vs rerank10、ΔのみのN≥10 抜粋)

**✅ KG が勝ったカテゴリ:**
| カテゴリ | N | rerank10 | KG | Δ |
|---|---|---|---|---|
| answer_level = Hard (難問) | 17 | 17.6% | 23.5% | +5.9 |
| Procedure/Method | 8 | 62.5% | 75.0% | +12.5 |
| Factoid | 5 | 20.0% | 40.0% | +20.0 |
| coord-ref (座標参照) | 29 | 31.0% | 34.5% | +3.4 |
| tables_charts = True | 70 | 25.7% | 28.6% | +2.9 |
| complex_layout = True | 28 | 25.0% | 28.6% | +3.6 |
| remote_reference = True | 26 | 23.1% | 26.9% | +3.8 |
| **Complex reasoning/Multi-hop** | **22** | **22.7%** | **22.7%** | **±0** |

**❌ KG が負けたカテゴリ:**
| カテゴリ | N | rerank10 | KG | Δ |
|---|---|---|---|---|
| hier-ref (階層参照) | 7 | 85.7% | 42.9% | -42.9 |
| tables_charts = False | 30 | 79.3% | 50.0% | -29.3 |
| Enumeration | 21 | 75.0% | 52.4% | -22.6 |
| answer_level = Easy | 19 | 77.8% | 57.9% | -19.9 |
| large_enumeration = True | 11 | 90.0% | 72.7% | -17.3 |
| output_type = list | 25 | 66.7% | 52.0% | -14.7 |
| reasoning_depth = single | 29 | 78.6% | 65.5% | -13.1 |
| abstraction_discrepancy = True | 15 | 33.3% | 20.0% | -13.3 |
| reasoning_depth = multi | 71 | 26.8% | 22.5% | -4.2 |
| multi_chunk = True | 58 | 47.4% | 39.7% | -7.7 |

### KG 構築コスト

| 項目 | 値 |
|---|---|
| 構築時間 | 5.6時間 (8 workers, gemma4 並列) |
| 成功 / 失敗 | 8,451 / 1 |
| Neo4j 規模 | 51,125 nodes / 151,030 edges |
| Term 数 | 34,199 (mention_count + pagerank付与) |
| Entity vector | 1,000 (LIMIT 1000上限がボトルネック) |

---

## 仮説と次の実験

### 仮説 1: **固定スキーマ (12関係) がドメイン不一致** ⭐

現在の KG スキーマは `IS_A / BELONGS_TO_CATEGORY / PART_OF / HAS_STEP / HAS_ATTRIBUTE / RELATED_TO / AFFECTS / CAUSES / DEPENDS_ON / APPLIES_TO / OWNED_BY / SAME_AS` の12種固定 (
[`graphrag_core/graph/schema.py`](../graphrag_core/graph/schema.py)
)。

Fujitsu コーパスは **多ドメイン混在**:
- HACCP (食品衛生): 「重要管理点」「管理基準」「衛生工程」「危害要因」
- 富士通決算: 「売上収益」「セグメント」「前年比」「事業区分」
- 統計局白書: 「対象期間」「統計値」「増減率」「分類区分」
- 熱中症対策: 「症状」「対処法」「危険度」「対象施設」
- 学校統計: 「学校種別」「在籍数」「設置者」
- LIFEBOOKカタログ: 「搭載モデル」「互換」「オプション」「対応OS」
- 要件定義ガイド: 「機能要件」「成果物」「ステップ」
- クラウドセキュリティ: 「設定不備」「責任共有」「対策」

これらを **無理やり12関係に押し込んだ結果**、抽出が `HAS_ATTRIBUTE` と `RELATED_TO` に過剰集中 (モデル就業規則eval時の観察と整合) → triples の情報量が低下 → グラフ探索で得られる文脈が表面的になる。

**観察的根拠**:
- KGが効くのは Procedure/Method (8問, HAS_STEP系で明示的) や Factoid (Term直結)
- 効かないのは Enumeration や Hard answer の細かい区別 (固定12関係では区別できない属性多数)

#### 提案する拡張スキーマ ([fujitsu_kg_schema.json](fujitsu_kg_schema.json))

34文書を全部見て手curationした拡張スキーマ:
- **Node types: 8種** — Person / Organization / Product / Process / Standard / Indicator / Concept / Document
- **Relations: 36種** — 既存12関係 + 24新規

新規24関係の主な追加:
| カテゴリ | 既存 | 新規追加 | 狙い |
|---|---|---|---|
| 分類 | `IS_A` `BELONGS_TO_CATEGORY` `PART_OF` | `HAS_PART` (PART_OFの逆) | 双方向探索強化 |
| 手順・時系列 | `HAS_STEP` | `FOLLOWS` `PRECEDES` `REQUIRES_BEFORE` | HACCP/要件定義工程の順序情報 |
| 属性 | `HAS_ATTRIBUTE` | `HAS_VALUE` `MEASURED_IN` | Indicator (財務/統計指標) の値・単位分離 |
| 因果 | `AFFECTS` `CAUSES` | `PREVENTS` `MITIGATES` | 熱中症対策・HACCP・セキュリティの予防系 |
| 依存 | `DEPENDS_ON` | `REQUIRES` `ENABLES` | より細かい依存関係 |
| 適用範囲 | `APPLIES_TO` | `COVERS` `EXCLUDES` `TARGETS` | 統計の調査対象、施設のターゲット表現 |
| 参照 | `SAME_AS` | `DEFINED_BY` `REFERENCES` | 条文参照・定義参照 |
| 組織・所有 | `OWNED_BY` | `MANAGED_BY` `ISSUED_BY` `ENACTED_BY` `OPERATED_BY` | 文書発行者、法律制定者、施設運営者 |
| 時間軸 | (なし) | `OCCURRED_IN` `VALID_FROM` `COMPARED_TO` | 決算/統計/規則改正の時間性 |
| 文書 | (なし) | `DESCRIBED_IN` `USES` | 文書記載関係・使用関係 |

**特に注力した3点**:
1. **`HAS_ATTRIBUTE` 過剰収束を防ぐ** — `Indicator → HAS_VALUE → 数値, MEASURED_IN → 単位` で財務/統計データを構造化
2. **`RELATED_TO` 削減** — 既存KGで445本中21本 (4.7%) を占めた汎用関係を、明示的に「最終手段のみ」と定義 + 他関係で受け皿を増やす
3. **8 Node types** — `Person` `Organization` `Product` `Process` `Standard` `Indicator` `Concept` `Document` で意味の絞り込み

**次の実験 FJH-05**:
- `.env` に `SHARED_SCHEMA_PATH=_bench/fujitsu_kg_schema.json` 追加 → build_kg.py が自動でこのスキーマを使う
- 再構築コスト: 同じ 5-6時間 (LLM呼び出し数は同じ)
- 評価: 同じ 100問パイプラインで `fujitsu_predictions_kg_v2.json` を出力 → FJH-04 (12関係版) と比較
- **見たい指標**:
  - 全体スコア: 35.0% → ?%
  - Procedure/Method: 75% → ?% (HAS_STEP/FOLLOWS/REQUIRES_BEFORE で更に効くか)
  - Enumeration: 52.4% → ?% (HAS_VALUE/MEASURED_IN で表データの抽出改善するか)
  - Easy questions: 57.9% → ?% (Node type絞込でノイズが減るか)

### 仮説 2: **graph_lines が LLM の注意を分散させてる**

`<GRAPH_CONTEXT>` に `推論パスN: A -[R1]-> B -[R2]-> C` の連鎖を含めると、簡単な質問では LLM が抽象的なエッジを読み込む副作用が出る。

**次の実験**:
- KG path/triples を context に含めず、検索結果chunkのソートだけに使う mode
- KG context 上限を 3-5本に絞る ablation
- 実装: `pipeline.py` の `graph_lines` 構築をconfig化

### 仮説 3: **Entity vector LIMIT 1000 が穴**

34,199 Term のうち 1,000 しかベクトル化されない (`entity_vector.py` 内ハードコード) →
質問→エンティティ抽出のリコールが落ちる。

**次の実験**:
- LIMIT を撤廃 or 上限を 10,000-30,000 へ
- pagerank 高いTermから優先選別
- Entity vector 再構築: 5-10分 (KG構築不要)
- 評価: 同条件再実行

### 仮説 4: **マルチホップではページ単位の検索粒度が勝る**

Fujitsu の Multi-hop 問題 (N=22) で KG vs rerank10 が ±0 (どちらも 22.7%) だった事実から、
「文書AのページX → 文書BのページY」という横断は **graph 経路よりも cross-encoder の意味マッチで十分**。

**次の実験**:
- top_k=10 まで広げて rerank した時の Multi-hop カテゴリ精度を測る
- ハイブリッド: query expansion (gemma4で関連語生成 → 多段検索)

---

## 実験管理表

| ID | 日付 | 条件 | 入力 | 出力 | 回答評価 | 参照full-cov | 参照match-rate | 備考 |
|---|---|---|---|---|---|---|---|---|
| JRAG-01 | 2026-06-04 | Generator only, full benchmark | gold context (positive + negative) | judge 0/1/2 | 78.9% (score=2率) | - | - | クリーン条件の上限 |
| FJH-01 | 2026-06-06 | hybrid (BM25+vector, top-5) | 8,452 page chunks | fujitsu_predictions.json | 29.59% | 51.02% | 65.43% | ベースライン |
| FJH-02 | 2026-06-06 | + rerank fetch30→5 | 同上 | fujitsu_predictions_rerank.json | 39.00% | 52.00% | 63.26% | rerank 効果確認 |
| FJH-03 | 2026-06-06 | + rerank fetch10→5 | 同上 | fujitsu_predictions_rerank10.json | **41.41%** | 50.51% | 63.06% | **best**, fetch ratio最適 |
| FJH-04 | 2026-06-07 | + KG (12関係固定) | 同上 + Neo4j | fujitsu_predictions_kg.json | 35.00% | 52.00% | **66.13%** | KG construction 5.6h |
| FJH-01-md | 2026-06-07 | hybrid + markdown chunks (pp_optimizer + PyMuPDF fallback) | 8,219 chunks | fujitsu_predictions_md.json | **36.36%** | 48.48% | 61.95% | **chunking効果 +6.8pt** |
| FJH-03-md | 2026-06-07 | rerank10 + markdown chunks | 同上 | fujitsu_predictions_rerank10_md.json | 40.21% | 48.45% | 62.80% | chunking効果 -1.2pt |
| FJH-04-md | 2026-06-07 | KG 12関係 + markdown chunks | 同上 + Neo4j 21,911nodes | fujitsu_predictions_kg_md.json | 31.00% | 49.00% | 62.33% | **chunking効果 -4.0pt (KGには逆効果)** |
| FJH-05 | (killed) | + KG (extended schema, 36関係, 旧chunks) | 中断 (build時正規化未実装) | - | - | - | - | FJH-06 に置換 |
| FJH-06 | 2026-06-07 | + KG (extended schema, 36関係, 旧chunks, **build-time normalize**) | 8,452 chunks + 51K nodes | fujitsu_predictions_kg_v3.json | 33.0% | 52.0% | 66.1% | FJH-04に対し overall -2pt: 拡張schemaの効果限定的 |
| FJH-06+EL | 2026-06-07 | + Entity Linking (top5000, ruri+LLM-judge, 757クラスタ統合) | 1,293 nodes 削除 / 15,118 edges 移動 | fujitsu_predictions_kg_v3_el.json | **30.0%** | 52.0% | 66.1% | **EL過剰マージで -5pt悪化**: 数値・型違いの誤統合が主因 |
| FJH-06+EL_v2 | 2026-06-09 | + Refined EL (numeric/substring/type-strict filter + LLM judge厳格化 + size=2 safe subset + 1char-diff/model-variant post-filter) | 95 nodes 削除 / 936 edges 移動 (FJH-06+EL の 1/14 規模) | fujitsu_predictions_kg_v3_el_v2.json | **30.0%** | 52.0% | 66.1% | **overall は変わらず -3pt だが multi_document=True で +9.1pt (9.1→18.2%)** ⭐ <br/>cross-doc multi-hopは改善、ただし single-doc / verification / quantitative で誤マージ起因の悪化が overall を相殺 |
| FJH-06+EL_v2+noLines | 2026-06-09 | + graph_lines off (仮説2): KG をエンティティ抽出/グラフトラバースに使うが、triples を LLM context に渡さない | 同上 | fujitsu_predictions_kg_v3_el_v2_no_lines.json | **35.0%** | 52.0% | 66.1% | **🎯 大幅改善: multi_doc=True +13.6pt / remote_ref=True +15.4pt / retrieval_Hard +13.0pt** <br/>graph triples が LLM の attention を逸らしてた仮説検証成功。35% は FJH-04 baseline と同じだが multi-hop 質指標は劇的に向上 |
| FJH-06+EL_v2+noLines+k10 | 2026-06-09 | + top_k 5→10, fetch_k 10→20 (仮説4: multi-hop は検索粒度が支配) | 同上 | fujitsu_predictions_kg_v3_el_v2_no_lines_k10.json | **42.0%** | **62.0%** | **72.3%** | **🎯 全軸プラス、regression ゼロ**: multi_doc +18.2 / remote_ref +19.2 / retrieval_Hard +17.4 / answer_Easy +15.8 / multi_chunk +13.8 / single-hop +13.8 / low_locality +10.9 / overall +9 <br/>多くが multi-page を必要としてて top_k=5 では足りなかった。検索粒度こそ最大レバー |
| FJH-07+gpt-4.1-mini | 2026-06-09 | LLM を gemma-4-26B → Azure gpt-4.1-mini に切替 (Phase A) | 同上 | fujitsu_predictions_azure_k10.json | **47.0%** ⭐ | 64.0% | - | **+5pt**: multi_doc=True 27.3→45.5 (+18.2) / remote_ref=True 30.8→46.2 (+15.4) / retrieval_Hard 30.4→47.8 (+17.4) <br/>gemma の refuse 多用パターン解消。賢いLLM が cross-doc/hard で大きく効く |
| FJH-07+gpt-4.1 | 2026-06-09 | gpt-4.1 (full) で 100Q | 同上 | fujitsu_predictions_azure_gpt41_k10.json | **52.0%** ⭐⭐ | 64.0% | - | **mini から +5pt、gemma から +10pt**。concurrency=2 で rate limit 回避完走。<br/>残失敗は (1) PDF表/図フラット化での読解失敗、(2) 質問前提誤りの認識、(3) 多chunk列挙、(4) 時系列ソート判定など LLM のpick力限界 |
| FJH-07+gpt-4.1+prompt v2 | 2026-06-09 | + 詳細プロンプト (前提検証+年代+列挙+比較+可否) | 同上 | fujitsu_predictions_azure_gpt41_k10_pv2.json | 50.0% | 63.0% | - | -2pt: 救った 4問 (年代/列挙) vs 損なった 6問 (簡単事実で over-thinking) |
| FJH-07+gpt-4.1+prompt v3 | 2026-06-09 | + 簡略プロンプト (最古判定+列挙のみ) | 同上 | fujitsu_predictions_azure_gpt41_k10_pv3.json | **52.0%** | 64.0% | - | v1 と同じ、v1 と内訳 flip 4-4。**プロンプト工夫は ±2pt ノイズ範囲、52% が gpt-4.1 + 現状 retrieval の天井** |
| FJH-07+gpt-4.1+md | 2026-06-09 | + markdown chunks (fjrag_hard_md) | 同上 | fujitsu_predictions_azure_gpt41_md_k10.json | 49.0% | 63.0% | 72.9% | -3pt: single-hop で -10pt 悪化。md chunks は大きく embedding 距離が変わり simple match が外れる。**preprocessing は現状 chunker の交換だけでは改善しない、新 chunker (表/図 抽出) が必要** |
| FJH-07 | TODO | + KG + entity vec LIMIT off | 34K Term全部ベクトル化 | - | - | - | - | 仮説3検証 |
| FJH-08 | TODO | rerank fetch20→10 | top_k=10 で多段問題に対応 | - | - | - | - | 仮説4検証 |
| FJH-09-off | 2026-06-11 | KG-off baseline: rerank fetch20→10 (gpt-4.1-mini) | fjrag_hard | fujitsu_predictions_rerank_k10_v2.json | 48.0% | 62.0% | - | fable5 リファクタ後の再ベースライン |
| FJH-09-on | 2026-06-11 | **KG-on (fable5修正版)**: search_keys正規化照合 / pagerank・extraction_count順パス列挙 / entity vec LIMIT撤廃 / KGチャンクrerank / noLines・k10 | 同上 + Neo4j 51K nodes (search_keys/pagerank backfill済) | fujitsu_predictions_kg_fix_k10_nolines.json | **51.0%** | 62.0% | - | **KG効果 +3pt (vs FJH-09-off)、fable5修正で +4pt (vs FJH-07 mini 47%)**。ペア分析: KG救済14問 / KG破壊11問 (net +3)。救済=entity関連・多文書比較・要件列挙、破壊=数値表・時系列 |

### FJH-09 詳細 (2026-06-11, fable5 リファクタ後の KG 効果再検証)

LLM=Azure gpt-4.1-mini, Judge=gemma-4-26B (self-judge), reranker=bge-reranker-v2-m3, 全 100Q, k10/noLines/hop2。

| 指標 | KG-off (rerank) | KG-on (fable5修正) | Δ |
|---|---|---|---|
| 回答 accuracy | 48.0% | **51.0%** | **+3.0** |
| 参照 full-coverage | 62.0% | 62.0% | 0 |
| 両方正解 | 37 | — | — |
| 両方不正解 | 38 | — | — |
| KG-onのみ正解 (救済) | — | 14 | — |
| KG-offのみ正解 (破壊) | — | 11 | — |

**fable5 修正の効果**: 同条件 (noLines/k10) の過去最良 FJH-07 mini=47.0% から **51.0% へ +4pt**。
主な修正: ① エンティティ照合を `search_keys`(NFKC正規化) 化で表記揺れ吸収 (lifebook等が0→51ノードヒット)、
② パス候補を pagerank/extraction_count 順で列挙 (旧: 無作為LIMIT + 長さ優先ソート)、
③ entity vector の LIMIT 1000 撤廃 (1000→34,215全件)、④ KGソースチャンクを取得拡大+cross-encoder rerank、
⑤ ドキュメントリランクのデッドコード修正。加えて **Sudachi tokenizer のスレッド安全化** (並列評価時の "Already borrowed" panic を解消し entity/BM25 検索の degrade を防止)。

**KG が救済した質問 (14)**: HACCP 手順推論・CCP 判定、多文書比較 (東京都vs大阪府の構成比/児童数)、
entity関連 (子会社名+収益合計、指紋認証+静脈センサー併用可否)、要件定義の列挙系。
→ EXPERIMENTS 既存知見 (KG=cross-doc multi-hop/entity-relation で効く) と整合。

**KG が破壊した質問 (11)**: 熱中症救急搬送の数値表比較、売上収益・前年比、1株当たり利益の時系列最大、
表/図セル参照系。→ 既存知見 (KG は数値表で N-hop が無関係パスに走る) と整合。

**結論**: fable5 のロジック修正で KG-on は 47→51% に改善し、KG効果 (+3pt) は依然 cross-doc/entity-relation 由来。
数値表系の破壊は KG の射程外 (preprocessing 課題) で不変。net で KG はプラス寄与を維持。

---

## FJH-10: スキーマ統合 (consolidation) 実験 (2026-06-11)

スキーマ実態診断 (`_bench/_schema_diag.py`) で判明した構造問題への対処を実装し、同条件 (k10/noLines/hop2, gpt-4.1-mini) で測定。
judge は gemma (self-judge相当) + **Azure gpt-4.1-mini (第2judge, `_bench/_judge_azure.py`)** の2系統。

### 診断で判明していた構造問題 (consolidation前)
- 関係タイプ: 定義36種に対し実際 **125種** (strict_mode=False で野良~90種、typo含む)
- 属性系3関係 (HAS_VALUE 19.6% + HAS_PART 16.6% + HAS_ATTRIBUTE 13.0%) で全エッジの49%
- **数値・日付のみのノード 5,715件** (`53 -[HAS_VALUE]-> 2` 等のゴミ、「令和6年度」が deg=141 のハブ)
- **同一idの複数ラベル分裂 2,600 id / 5,557ノード** (型揺れ分裂 — 名寄せ問題の最大成分)
- 逆方向ペア二重登録 287組 (HAS_PART/PART_OF — 検索は無向なので情報量ゼロ)

### 実施した統合 (graphrag_core/graph/consolidate.py, ビルド後処理として build_kg.py にも組込み)
1. 値ノード5,801件に `is_value` flag → 検索/enrichment/entity vector から除外
2. 型分裂 2,494 id をマージ (2,851ノード削除、エッジ・MENTIONS をkeeperへ移設)
3. 関係正規化 ~22,600エッジ (HAS_VALUE→HAS_ATTRIBUTE 10,357 / HAS_PART→PART_OF反転 8,589 など27ルール)
4. スキーマ v2 (`fujitsu_kg_schema_v2.json`, 36→23関係) + strict_mode=True + プロンプトに値ノード抽出禁止

### 結果 (100Q)

| 条件 | gemma judge | Azure judge | 参照 full-cov |
|---|---|---|---|
| KG-off (rerank k10) | 48.0% | 50.0% | 62.0% |
| KG-on (consolidation前 = FJH-09-on) | 51.0% | **56.0%** | 62.0% |
| KG-on + consolidation (値ノード完全排除) | 46.0% | 49.0% | 65.0% |
| KG-on + consolidation (値ノード終端のみ許可) | 47.0% | 50.0% | **66.0%** |
| **KG-on + consolidation + KGチャンク共起順位付け (v2c)** | **52.0%** | 55.0% | 65.0% |

### 学び
1. **構造统合は参照カバレッジを +3〜4pt 改善** (62→65-66%、全実験史上最高) — 統合自体は検索に有効
2. **しかし最初の2バリアントで回答が退行** (-5pt)。原因はマージで MENTIONS が集約された結果、
   無順位 `LIMIT` の KGチャンク候補プールが不安定化したこと (スキーマではなく chunk pull の問題)
3. **修正: MENTIONS 経由のチャンク取得を「言及エンティティ数 (共起) 降順」で順位付け + fetch 30** → 両judgeで回復
4. 値ノードは「起点・中継禁止 / 終端許可」が正解 — 完全排除すると数値質問の出典チャンクpullまで失われる
5. judge感度に注意: 同一内容の回答でも言い回しで gemma 判定が flip する。**2judge体制 + 参照カバレッジ (決定的指標) の併用が必須**

### 最終状態
- 本番グラフ: 48,266ノード / 134,238エッジ (統合済み、フルバックアップ: `_bench/backup_full_graph_20260611.jsonl.gz`)
- 逆方向ペア 0、型分裂は値ノードの106 idのみ (検索対象外)、正規23関係が上位を占有
- KG効果 (v2c vs KG-off): **gemma +4pt / Azure +5pt / 参照 +3pt**

### KGターゲット質問のスライス検証 (`_bench/_kg_signature_analysis.py`, 2026-06-11)

「KGでしか解けないQAが解けているか」の直接検証。

**KGターゲット軸 (KG-off → KG-v2c, gemma / Azure):**
| 軸 | N | KG-off | KG-v2c | Δ(gemma) | Δ(Azure) |
|---|---|---|---|---|---|
| reasoning_depth=multi | 71 | 32.4/36.6% | 40.8/43.7% | **+8.5** | **+7.1** |
| low_locality=True | 55 | 54.5/54.5% | 60.0/61.8% | +5.5 | +7.3 |
| remote_reference=True | 26 | 38.5/38.5% | 42.3/42.3% | +3.8 | +3.8 |
| multi_document=True | 22 | 45.5/36.4% | 45.5/40.9% | ±0 | +4.5 |
| retrieval_level=Hard | 23 | — /39.1% | — /43.5% | — | +4.4 |
| negation=True | 11 | 36.4% | 72.7% | **+36.4** | — |
| cause_effect=True | 22 | 50.0% | 68.2% | +18.2 | — |
| 逆に reasoning_depth=single | 29 | 86.2% | 79.3% | **-6.9** | — |

**両judge一致で「KGのみ正解」: 7問**（全て reasoning_depth=multi）
- 手順・因果推論: CCP非該当理由 / 乳酸菌飲料の工程順序の正誤 / 無線LANの波長理由
- 多文書統計比較: 専修学校構成比1-3位(東京vs大阪) [multi_doc+remote_ref+Hard] / 大阪府の女性多数教員の校種
- entity-relation: ユビキタス子会社名+3Q累計収益 [multi_doc+remote_ref+Hard]
- スペック制約: 指紋認証電源ボタン+静脈センサー併用可否

**両judge一致で「KGが壊した」: 4問**（豪州HACCP導入分野 / 熱中症搬送の期間比較 / アイデア発想列挙 / 3Q売上前年比）→ net **+3問**

**FJH-07 シグネチャ5問の追跡:** Q45専修学校=KGのみ正解を維持✓ / Q52女性教員=KGのみ正解✓ /
Q67副社長=KG-offでも正解化(gpt-4.1-mini+k10で吸収) / Q65 3Q売上=KGが逆に阻害✗ / Q17混信=全条件不正解。

**結論**: KGの貢献はターゲット領域に明確に存在する — multi-hop +7〜8.5pt、KG専用正解7問はすべて
「チャンク単独では繋がらない推論・比較・entity関連」。一方 single-hop (-6.9pt) と数値表時系列では
依然ノイズ源であり、**質問タイプによる KG 有効/無効の動的切替**（例: エンティティ抽出数・質問分類で
graph検索をスキップ）が次の改善レバー。

---

## 生成物の場所

```
_bench/
├── fujitsu_ingest.py             # PDF → page-aware chunks → PGVector
├── fujitsu_runner.py             # 100Q を HybridRetriever + 任意rerank で実行
├── fujitsu_runner_kg.py          # 100Q を retriever_and_merge (KG込み) で実行
├── fujitsu_build_kg.py           # 全PDFから page-aware KG構築 (5-8h)
├── evaluate_qa_local.py          # Fujitsu evaluate_qa.py の vLLM 向けフォーク
├── analyze_by_axes.py            # 4軸メタデータでスライス集計
├── fujitsu_predictions.json      # FJH-01 予測
├── fujitsu_predictions_rerank.json   # FJH-02
├── fujitsu_predictions_rerank10.json # FJH-03
├── fujitsu_predictions_kg.json   # FJH-04
├── build_kg.log                  # KG構築ログ
├── runner_kg.log                 # KGランナーログ
└── results/
    └── evaluation_result_*.json  # 採点結果 (各条件×参照モード)
```

---

## キーインサイト (実用判断材料)

1. **cross-encoder rerank が最強の改善レバー** (+11.8pt)。投資対効果が圧倒的に高い
2. **fetch_k の最適化が大事** (30→5 より 10→5 が良い、初期検索ノイズを増やしすぎない)
3. **KG は『何にでも効く』ではなく『カテゴリ依存』** — Procedure/Method や難問では効くが、Enumeration や単発事実問では悪影響
4. **固定12関係スキーマは多ドメインコーパスでボトルネック** (仮説1) → 次の優先実験
5. **Judge同一モデルバイアスに注意** — 別モデル (gpt-oss-120b等) で再評価して差分確認すべき
6. **markdown chunking は hybrid 単体で +6.8pt の効果**だが、rerank と組み合わせると効果なし、KGとは相性が悪い
7. **preprocessing_optimizer の page categorization は過剰**: 表+テキスト混在ページが「画像」と判定されて text 抽出スキップ → PyMuPDF fallback 必須

### chunking 戦略の使い分け (FJH-04 vs FJH-04-md より)

| ターゲット | 推奨chunks | 理由 |
|---|---|---|
| 純 vector検索 (KG無し) | **markdown chunks** | LLMへの context質向上で +6.8pt |
| rerank ありの retrieval | どちらでも (rerank が支配的) | 差は -1.2pt 程度 |
| KG 構築 | **旧 PyMuPDF chunks** | 細粒度Term抽出が確保される、-4pt の悪化を回避 |

つまり **「KG用のチャンキング」と「retrieval用のチャンキング」を分離する**のが最適解。これは設計の大幅刷新を意味する: 2セットのコレクション (`fjrag_hard_md` for retrieval + `fjrag_hard` for KG build) を併用する構成。

### 次の検証 (FJH-05) の修正版

仮説1「拡張スキーマ」検証は **旧PyMuPDF chunks (FJH-04 のチャンクセット) でやり直すべき**。
markdown chunks では KG の効果が下がるので、フェアな比較にならない。

---

## KG の真の貢献領域 (FJH-07 検証, 2026-06-09)

### 観測
- 全 100問の recall@5: KGあり 60.5% vs KGなし 59.9% (差 +0.6pt → 一見「KG は無意味」)
- **しかし非図表系 33問だけ見ると recall@5: KGあり 23/33 (70%) vs KGなし 18/33 → +5問改善**
- **gemma同士で「KGなし不正解→KGあり正解」の質問: 5問** (Q17, Q45, Q52, Q65, Q67)

### KG でしか解けない 5問の特徴
| Q# | パターン | KG の貢献メカニズム |
|---|---|---|
| Q17 | 時系列最少 (混信件数2012-2024) | 追加 chunk pull (富士通レポ参照) で時系列補完 |
| Q45 | 多文書比較 (専修学校 東京vs大阪) | `1_report.pdf p.11` を rank に出す (KGなしは取れず refuse) |
| Q52 | 列挙 (大阪府女性教員多い校種) | KG context で LLM が refuse→「幼稚園と小学校」 |
| Q65 | 数値+期間 (3Q売上と前年比) | 前年比 chunk (p.5-7) を追加 |
| Q67 | entity-relation (副社長は誰) | 副社長 entity 起点に正しい p.76 と株主総会PDF を pull |

### 結論
KG の真の役割: **vector retrieval が拾えない entity 関連 chunk を補完**
- 同一entity の別ページ・別 PDF を引き寄せる (Q17, Q65)
- entity-relation 直接照会 (Q67)
- 多文書間の同一 entity マッピング (Q45)
- 全体精度への寄与: **5pt** (52% のうち 5pt は KG 直接貢献)
- KG なし設計なら 47% 程度。KG ありで 52% = +5pt 効いてる

### 図表系では KG 不発の理由
- 図表系質問の答えは表セル/グラフ内の数値で、entity 関係ではない
- N-hop traversal は「日本→大阪府→8.1」のような無関係 path に走る
- 図表系の改善は preprocessing (表/図抽出) 側の課題、KG は射程外

---

## FJH-06+EL_v2 詳細 (2026-06-09)

### 経緯
- bi-encoder ベース失敗分類 v2 で multi-hop 失敗 66問中 60問 (91%) に「true_split」候補が存在
- EL_v1 の失敗教訓 (numeric/substring/type/cluster) を反映した精緻化版を実装
- 段階 v1→v2b→v2c で dry-run iteration, さらに size=2 限定 + 1char-diff/model-variant/substring_len_diff=1 を post-filter
- 154 size=2 候補 → **95 cluster (61%)** を実マージ (前回 EL_v1 は 757 cluster の 1/14)

### 結果
| 軸 | FJH-06 | FJH-06+EL_v2 | Δ |
|---|---|---|---|
| Overall | 33% | 30% | -3 |
| Ref accuracy (full-cov) | 52% | 52% | 0 |
| **multi_document=True (22Q)** | 9.1% | **18.2%** | **+9.1** ⭐ |
| multi_chunk=True (58Q) | 32.8% | 34.5% | +1.7 |
| low_locality=True (55Q) | 36.4% | 38.2% | +1.8 |
| remote_reference=True (26Q) | 11.5% | 15.4% | +3.8 |
| reasoning_depth=multi (71Q) | 22.5% | 16.9% | -5.6 |
| multi_chunk=False (42Q) | 33.3% | 23.8% | -9.5 |
| hier-ref (7Q) | 85.7% | 42.9% | -42.9 |

### Split削減 (post-merge classify_v2 再実行)
| | pre | post | Δ |
|---|---|---|---|
| 質問単位 true_split | 60 | 59 | -1 |
| Entity単位 true_split | 223 | 212 | -11 |

### 解釈
- **cross-doc multi-hop は明確に改善** (+9.1pt) — 仮説 (split解消が multi-doc に効く) 検証成功
- **overall -3pt の原因** は single-doc 質問への collateral damage:
  - 社外取締役↔社外監査役 (距離=2、1char-diff filter で取れない誤マージ)
  - 一般社団法人...協議会 ↔ ...販売団体協議会 (異なる団体名の混同)
  - CRUD図のID ↔ CRUD図の名称 (役割違い)
- 95 merge では split 削減はわずか (-11 entity)。inferred: 残り 200+ entityは LLM judge を通せなかった/低 pagerank で対象外
- **本命の hier-ref axis で -42.9pt** は深刻 — 階層参照系の質問が破壊された

### 次のアクション候補
1. **distance=2 で role/組織名末尾差を catch する追加ルール** (取締役/監査役、協議会/販売団体協議会 等)
2. **LLM judge をより強力なモデル (gpt-4.1) に切替** — 4.1-miniの判断ミスを除外
3. **失敗質問特定→該当 merge ban → 部分 revert** で hier-ref/quant 系を救済
4. **EL不採用に戻す**: backup `/var/lib/neo4j/import/fjh06_pre_el.cypher` で FJH-06 baseline 復元
5. **cross-doc 改善のみ採用**: 95 merge keep + multi_doc=True 質問にだけ EL 効果を限定的に適用

---

## 仮説2検証: graph_lines off (2026-06-09)

### 経緯
EL_v2 の overall -3pt 損失を見て、「graph triples を LLM context に出すこと自体が attention dilution を引き起こしてる」仮説 (EXPERIMENTS.md 仮説2) を検証。
`include_graph_lines: False` config を pipeline に追加 → KG はエンティティ抽出/Cypherトラバースには使うが、`<GRAPH_CONTEXT>` ブロックを LLM プロンプトから除外。

### 結果 (FJH-06+EL_v2 + graph_lines off)
| 軸 | FJH-06 | EL_v2 (lines on) | EL_v2 noLines | Δ noLines vs FJH-06 |
|---|---|---|---|---|
| **Overall** | 33.0% | 30.0% | **35.0%** | **+2.0** |
| Ref full-coverage | 52% | 52% | 52% | 0 |
| Ref match-rate | 66.1% | 66.1% | 66.1% | 0 |
| **multi_document=True (22Q)** | 9.1% | 18.2% | **22.7%** | **+13.6** ⭐⭐ |
| **remote_reference=True (26Q)** | 11.5% | 15.4% | **26.9%** | **+15.4** ⭐⭐ |
| **retrieval_level=Hard (23Q)** | 13.0% | 21.7% | **26.1%** | **+13.0** ⭐ |
| reasoning_depth=multi (71Q) | 22.5% | 16.9% | 26.8% | +4.3 |
| multi_chunk=True (58Q) | 32.8% | 34.5% | 37.9% | +5.1 |
| low_locality=True (55Q) | 36.4% | 38.2% | 40.0% | +3.6 |
| answer_level=Hard (17Q) | 11.8% | 17.6% | 17.6% | +5.9 |
| Comparison (7Q) | 14.3% | 14.3% | 28.6% | +14.3 |

### 解釈
- **multi-hop 系全軸で大幅改善** (multi_doc +13.6, remote_ref +15.4, retrieval_Hard +13.0)
- 「KG triples を見せると LLM が情報を統合しきれない」が実証
- KG の真の価値は **エンティティ抽出/グラフ traversal で関連chunk を取ってくる** こと。triples は LLM context には不要
- Overall 35% は FJH-04 (35%, 旧12関係スキーマ) と同じだが、**質的に multi-hop が遥かに良い**
- ref accuracy 不変 = 取れてるページは同じ。LLMの回答品質だけが上がった

### 含意 — 設計指針の更新
- KG context output (`graph_lines`) はデフォルト OFF にすべき
- KG の役割を **「retrieval enrichment 専用」** に再定義: entity抽出 + N-hop traversal + source chunk pull
- これで rerank と KG が両立しやすくなる (両方とも entity経由)

---

## FJH-11: 参照追跡グラフ + 照応解決 (2026-06-11)

法令参照抽出の手法（北野・天笠 NLP2026: パターンベース F0.935）をマニュアルコーパスに翻案。
`graphrag_core/graph/references.py` + `scripts/build_reference_graph.py`（既存グラフに後付け可、LLMコストゼロ）。

### 構築結果（実グラフ適用済み）
- REFERS_TOエッジ **3,122本**（page 2,798 / section 324）+ 文書名参照チャンク4件
- 略称定義 17件 → 照応解決10件（「本公開買付者」→パロマ・リームHD等、ALIAS_OF + search_keys注入）
- **is_anaphor フラグ 136ノード**（「本製品」deg=114 等の偽統合ハブを検索除外）← これは恒久採用
- 検索時: ヒットチャンクから1ホップ（節/ページ参照→直接、文書名参照→参照先文書スコープの再検索）+ cross-encoderゲート

### 個別実証（SIM質問）
B5FL1891 p.97「『内蔵無線WANをお使いになる方へ』をご覧ください」→ B5FL0331 p.13「2.1.1 SIM接続する」の取得に成功
（正解refヒット 1/7→2/7、接続方法の本文がコンテキストに入る）。機構としては設計どおり動作。

### ベンチ結果（100Q, ref-follow ON vs v2c）
| 指標 | v2c (OFF) | ref-follow ON | Δ |
|---|---|---|---|
| gemma | 52.0% | 47.0% | -5.0 |
| Azure mini | 55.0% | 53.0% | -2.0 |
| 参照 full-cov | 65.0% | 64.0% | -1.0 |
| remote_reference=True (26Q, azure) | 11/26 | 9/26 | -2 |

### 解釈と判断
- **ターゲット軸 (remote_reference) ですら改善せず**。flip分析では参照追跡と無関係な統計比較系の
  不安定問題が行き来しているだけで、ベンチのremote_referenceラベルの実体は「明示的な参照ポインタを辿る」
  型ではなく「離れたページ・表に答えが分散」型が大半。明示ポインタ型はSIM/eSIMの2-3問のみ
- 追加コンテキストが境界線上の生成を揺らすコスト > 数問の参照解決ゲイン
- **判断: `enable_reference_follow` デフォルトOFF**（フラグとして温存。規程・マニュアル個別QAでは有効な場面あり）。
  is_anaphor除外と略称解決（ALIAS_OF）は副作用がなく恒久採用
- 教訓: 「機構が正しく動く」と「ベンチが上がる」は別物。個別実証→全体A/B→軸別検証の3段で判断すべき

---

## FJH-12: かな揺れ名寄せ (2026-06-12)

ベンチ最適化ではなく一般品質改善として実装（「ガス軸受/ガス軸受け」「データの連携/データ連携」「サーバ/サーバー」型）。

### 実装 (`consolidate.merge_kana_variant_nodes` + `japanese.kana_variant_key`)
- **文字種限定レーベンシュタイン**: 差分文字がひらがなのみ（長音ーは末尾のみ）∧ 語頭一致 ∧ 差分≤3字で同一判定。
  数字・英字・漢字の差分は距離1でも禁止 → U7314/U7414、第3/第4四半期、取締役/監査役、サーバー/サバ を構造的に排除
  （単体検証20ケース全パス。埋め込み類似EL(FJH-06+EL, -5pt)との本質的差分は誤統合クラスの決定的遮断）
- 候補生成は `kana_variant_key`（送り仮名・助詞・末尾長音を除いた骨格、内容語脱落ガード付き）
- マージされた表記はkeeperの aliases に保存 → search_keys 再計算で照合キーとして残る
- 検索側: ノードsearch_keys + クエリ側エンティティの両方にかな揺れ骨格キーを追加（双方向照合）

### 適用結果（実グラフ）
- 候補1,002ノード → 厳格ルールで **319組 / 335ノードをマージ**（ガードが2/3を棄却）
- リグレッションベンチ: gemma 50%(v2c 52) / Azure 52%(55) / **参照 full-cov 65%（不変）**
  → 決定的指標は不変、judge差は既知のノイズ帯域内。リグレッションなしと判断

最終更新: 2026-06-12
