# Langfuseトレーシング統合 (v4) — 実装完了

## Context

Graph-RAGアプリケーションの全LLM呼び出しをLangfuseでトレースできるようにした。
Langfuse SDK v4 + `@observe()` デコレータによる階層トレース自動構築を使用。

## 設計方針

- **`llm_factory.py` は変更しない** — LLMコンストラクタには触らない
- **`@observe()` デコレータで関数単位のトレース階層を自動構築** — v4推奨パターン
- **`langfuse_config.py`** — `observe`, `propagate_attributes`, `get_langfuse_callback`, `update_current_span` 等を集約
- **env var未設定時はゼロオーバーヘッド** — no-opフォールバックで通常動作
- **`capture_input=False` + `update_current_span()`** — ベクトル等の大きな引数を除外し有用な情報のみ記録

## コア統合パターン (v4)

```python
from langfuse_config import observe, get_langfuse_callback, update_current_span

# @observe() でトレース階層を自動構築
@observe(name="entity_extraction")
def extract_entities(question, llm, ...):
    response = llm.invoke(prompt, config=get_langfuse_callback())
    return result

# ベクトル引数を除外して有用な情報のみ記録
@observe(name="hybrid_search", capture_input=False)
def search(query_text, query_vector, k):
    update_current_span(input={"query_text": query_text, "k": k})
    ...

# LLMGraphTransformer（invokeを制御できないのでインスタンスレベル注入）
if is_langfuse_enabled():
    from langfuse.langchain import CallbackHandler
    llm.callbacks = [CallbackHandler()]
```

## トレースラベル体系

| observe name | 目的 | ファイル |
|---|---|---|
| `entity_extraction` | 質問からエンティティ抽出 | qa_pipeline.py |
| `relation_ranking` | グラフ関係性ランキング | qa_pipeline.py |
| `path_ranking` | パスランキング | qa_pipeline.py |
| `get_graph_context` | グラフコンテキスト構築 | qa_pipeline.py |
| `retriever_and_merge` | 検索・マージ統合 | qa_pipeline.py |
| `hybrid_search` | ハイブリッド検索 | hybrid_retriever.py |
| `doc_reranking` | ドキュメントリランキング | hybrid_retriever.py |
| `graphrag_qa` | バッチ質問処理 | batch_eval.py |
| `kg_building` | KGトリプル抽出 | app.py, build_kg.py |

## 主要ファイル

| ファイル | 役割 |
|---|---|
| `langfuse_config.py` | `observe`, `propagate_attributes`, `get_langfuse_callback`, `update_current_span` 等のヘルパー |
| `qa_pipeline.py` | QAパイプライン全関数に `@observe` デコレータ付与 |
| `hybrid_retriever.py` | 検索関数に `@observe` + `capture_input=False` + `update_current_span` |
| `batch_eval.py` | `@observe` + `propagate_attributes` でバッチセッション管理 |
| `build_kg.py` | LLMインスタンスにCallbackHandler注入 |

## セットアップ

1. `pip install langfuse>=4.0.0`
2. `.env` に以下を追加:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # またはセルフホストURL
```
3. アプリケーションを通常通り起動（設定なしなら従来通り動作）
