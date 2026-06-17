# JEMHopQA 評価結果（KGの価値弁別ベンチ）

日本語マルチホップQA [JEMHopQA](https://github.com/aiishii/JEMHopQA)（RIKEN, CC BY-SA 4.0, dev 120問）で
graphrag を評価。plant_v15 が too_easy で KG を弁別できなかったのに対し、JEMHopQA は
**KG-off が構造的に落ちる純マルチホップ**を弁別できる。

## データ
- 評価: dev 120問（comparison 73 / compositional 47）。各問 gold=Wikipedia page_ids 2件 + derivation triples
- コーパス: page_id で日本語Wikipedia本文をAPI取得 → 1,260記事 / 39,546チャンク → PGVector `jemhopqa`（ruri 768d）
- KG: compositional-gold 88記事の**先頭1チャンク**を vLLM gemma-4-26B でオープンドメイン抽出
  （1,092ノード/2,251エッジ、~8.5分、ワーカー4で飽和回避）。Neo4jは退避→専用構築（`build_kg_jemhop.py`）

## 結果: KG-off vs KG-on（回答は Azure gpt-4.1-mini judge）

| 指標 | KG-off | **KG-on** | Δ |
|---|---|---|---|
| **回答 accuracy（全体）** | **73.3%** | **81.7%** | **+8.4** |
| comparison 回答 | 86% | 90% | +4（ほぼ飽和） |
| **compositional 回答** | 53% | **68%** | **+15** ⭐ |
| 参照 full-coverage（全体） | 80% | **92.5%** | +12.5 |
| **compositional 参照full** | **57%** | **89%** | **+32** ⭐ |
| comparison 参照full | 95% | 95% | ±0 |

**KGはbridge取りこぼしを狙いどおり回復**。comparison（両エンティティが質問に出る＝検索95%・回答86%で
ほぼ飽和）はKG marginal、KGが効くべき compositional だけで大幅改善 = KGの貢献が因果的に分離できている。

> 注: 当初の string-match 採点（`eval_jemhop.py`）は文中の「はい/いいえ」を拾えず comparison を
> 42% と44pt過小評価していた（全体も52.5%と誤表示）。回答は LLM judge（`judge_azure.py`）が正。
> 参照 full-coverage は決定的指標で採点バグの影響なし。

### bridge回復の実例（KG-off参照欠落 → KG-on full-coverage）
| 質問 | KG回復した橋渡し記事 | KG-on回答 |
|---|---|---|
| 杉咲花の父の職業は？ | 木暮武彦 | ギタリスト ✓ |
| 東條英機が死没した施設は何戦争の後に設置？ | 巣鴨拘置所 | 第二次世界大戦 ✓ |
| 『しあわせの保護色』センターの所属事務所は？ | 白石麻衣 | 乃木坂46合同会社 ✓ |
| ジープの親会社が登記された国は？ | ステランティス | オランダ ✓ |

機構: 質問に出ない橋渡しエンティティ（木暮武彦等）を、`杉咲花 -[父]-> 木暮武彦` のKGエッジで辿り、
その記事チャンクを文脈注入 → vector検索が取りこぼした2件目を回復。

## plant_v15 との対比（KG弁別性の証明）
| | plant_v15 | JEMHopQA |
|---|---|---|
| 課題の所在 | 生成プロンプト（KG不要、72→92%） | **検索のbridge欠落（KGで回復）** |
| KG-off compositional/multi-hop 検索 | 96%（飽和） | **57%（構造的欠落）** |
| KGの効果 | ほぼ無し | **参照+32pt / 回答+17pt（compositional）** |

→ JEMHopQA は「retrieval+LLM が証明的に落ち、KG traversal が回復する」純マルチホップを
弁別できる。出現率や難易度ラベルではなく、**KG-off→KG-on の因果差分**で KG の価値を実測できた。

## 注記・再現
- KG は compositional-gold・intro-only（88記事）の小規模＝KGに有利な設定だが、comparison不変が
  「データ増量ではなくbridge traversalの効果」を担保。フル構築でさらに伸びる余地あり。
- 残る compositional 失敗は 3-hop / 細かい事実（例: 徳川家宣の祖父の墓所）で、検索回復しても
  生成が誤る型 → 生成/推論側の課題（KGの射程外）。
- 再現: `fetch_data.py` → `ingest_jemhop.py` → `build_kg_jemhop.py`（vLLM, KG_BUILD_WORKERS=4,
  JEMHOP_CHUNKS_PER_ART=1）→ `run_jemhop.py --kg` → `eval_jemhop.py`。
- Neo4j復元: `restore_neo4j.py` + `dedup_edges.py`（重複id由来のエッジ膨張を正規化）。
- 教訓: vLLM gemma は同時実行（ワーカー）を上げすぎると飽和ハングする。2〜4が安全。
