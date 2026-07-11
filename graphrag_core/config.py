"""
統一設定モジュール
プロジェクト唯一の load_dotenv() 呼出地点。
全環境変数をここに集約する。
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_bool(key: str, default: str = "true") -> bool:
    return _env(key, default).lower() == "true"


def _env_int(key: str, default: str = "0") -> int:
    try:
        return int(_env(key, default))
    except ValueError:
        return int(default)


def _env_float(key: str, default: str = "0.0") -> float:
    try:
        return float(_env(key, default))
    except ValueError:
        return float(default)


@dataclass
class Settings:
    """アプリケーション全体の設定"""

    # --- Neo4j ---
    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI"))
    neo4j_user: str = field(default_factory=lambda: _env("NEO4J_USER"))
    neo4j_pw: str = field(default_factory=lambda: _env("NEO4J_PW"))

    # --- PostgreSQL ---
    pg_conn: str = field(default_factory=lambda: _env("PG_CONN"))
    pg_collection: str = field(default_factory=lambda: _env("PG_COLLECTION", "graphrag"))

    # --- Azure OpenAI ---
    azure_openai_api_key: str = field(default_factory=lambda: _env("AZURE_OPENAI_API_KEY"))
    azure_openai_endpoint: str = field(default_factory=lambda: _env("AZURE_OPENAI_ENDPOINT"))
    azure_openai_api_version: str = field(default_factory=lambda: _env("AZURE_OPENAI_API_VERSION"))
    azure_openai_chat_deployment: str = field(default_factory=lambda: _env("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"))
    azure_openai_embedding_deployment: str = field(default_factory=lambda: _env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"))

    # --- Azure Document Intelligence ---
    azure_di_endpoint: str = field(default_factory=lambda: _env("AZURE_DI_ENDPOINT"))
    azure_di_api_key: str = field(default_factory=lambda: _env("AZURE_DI_API_KEY"))
    azure_di_model: str = field(default_factory=lambda: _env("AZURE_DI_MODEL", "prebuilt-layout"))

    # --- LLM Provider ---
    # 完全オンプレ既定: vllm（DGXのgemma）。Azureを使う場合のみ LLM_PROVIDER=azure_openai を明示。
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "vllm"))

    # --- VLLM ---
    vllm_endpoint: str = field(default_factory=lambda: _env("VLLM_ENDPOINT"))
    vllm_model: str = field(default_factory=lambda: _env("VLLM_MODEL", "openai/gpt-oss-120b"))
    vllm_api_key: str = field(default_factory=lambda: _env("VLLM_API_KEY", "EMPTY"))
    vllm_temperature: float = field(default_factory=lambda: _env_float("VLLM_TEMPERATURE", "0.0"))
    vllm_max_tokens: int = field(default_factory=lambda: _env_int("VLLM_MAX_TOKENS", "4096"))
    vllm_timeout: int = field(default_factory=lambda: _env_int("VLLM_TIMEOUT", "60"))
    vllm_reasoning_effort: str = field(default_factory=lambda: _env("VLLM_REASONING_EFFORT", "low"))

    # --- OpenAI direct (gpt-4.1 等) ---
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4.1"))
    openai_timeout: int = field(default_factory=lambda: _env_int("OPENAI_TIMEOUT", "120"))

    # --- Anthropic (claude) ---
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    anthropic_timeout: int = field(default_factory=lambda: _env_int("ANTHROPIC_TIMEOUT", "120"))

    # --- KG Schema (shared JSON for EDC等の外部スキーマ発見ツール連携) ---
    # 未設定 or ファイル無しならデフォルト12関係にフォールバック
    shared_schema_path: str = field(default_factory=lambda: _env("SHARED_SCHEMA_PATH", ""))

    # --- KG Chunk Schema (外部構築グラフへの接続プロファイル) ---
    # チャンクノードのラベルとチャンク→エンティティのエッジ型。
    # 既定は graphrag ネイティブ（LangChain include_source 由来の :Document / MENTIONS）。
    # llm-graph-builder 製グラフに接続する場合: KG_CHUNK_LABEL=Chunk, KG_CHUNK_EDGE=HAS_ENTITY
    kg_chunk_label: str = field(default_factory=lambda: _env("KG_CHUNK_LABEL", "Document"))
    kg_chunk_edge: str = field(default_factory=lambda: _env("KG_CHUNK_EDGE", "MENTIONS"))

    # --- Term Dictionary (外部用語辞書) ---
    # JSON/CSV のパス。指定されていれば build_kg 末尾で Term ノードに
    # canonical_form / aliases / category / definition を後付けする
    kg_dictionary_path: str = field(default_factory=lambda: _env("KG_DICTIONARY_PATH", ""))

    # --- Reranker (cross-encoder via vLLM /v1/score) ---
    reranker_enabled: bool = field(default_factory=lambda: _env_bool("RERANKER_ENABLED", "true"))
    vllm_reranker_endpoint: str = field(default_factory=lambda: _env("VLLM_RERANKER_ENDPOINT", "http://localhost:8006/v1"))
    vllm_reranker_model: str = field(default_factory=lambda: _env("VLLM_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
    vllm_reranker_api_key: str = field(default_factory=lambda: _env("VLLM_RERANKER_API_KEY", "EMPTY"))

    # --- Embedding Provider ---
    # 完全オンプレ既定: vllm（DGXのruri-v3）。
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "vllm"))
    vllm_embedding_endpoint: str = field(default_factory=lambda: _env("VLLM_EMBEDDING_ENDPOINT", "http://localhost:8001/v1"))
    vllm_embedding_model: str = field(default_factory=lambda: _env("VLLM_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"))
    vllm_embedding_api_key: str = field(default_factory=lambda: _env("VLLM_EMBEDDING_API_KEY", "EMPTY"))

    # --- PDF Preprocessing (on-prem via preprocessing_optimizer) ---
    # "onprem" = preprocessing_optimizer + vLLM Vision (DGX Spark)
    # "azure_di" = Azure Document Intelligence (cloud, 旧仕様)
    # "pymupdf" = PyMuPDF plain text (最軽量)
    pdf_processor: str = field(default_factory=lambda: _env("PDF_PROCESSOR", "onprem"))
    pdf_backend: str = field(default_factory=lambda: _env("PDF_BACKEND", "vllm"))
    vllm_vision_endpoint: str = field(default_factory=lambda: _env("VLLM_VISION_ENDPOINT", "http://localhost:8004/v1"))
    vllm_vision_model: str = field(default_factory=lambda: _env("VLLM_VISION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    vllm_vision_api_key: str = field(default_factory=lambda: _env("VLLM_VISION_API_KEY", "EMPTY"))
    preprocessing_optimizer_path: str = field(default_factory=lambda: _env("PREPROCESSING_OPTIMIZER_PATH", "../preprocessing_optimizer"))
    # PaddleX remote (DGX Spark CPU) - PP-OCRv5 サーバ
    paddlex_endpoint: str = field(default_factory=lambda: _env("PADDLEX_ENDPOINT", "http://localhost:8005"))
    paddlex_timeout: int = field(default_factory=lambda: _env_int("PADDLEX_TIMEOUT", "180"))
    # PaddleX Layout サーバ（レイアウト検出専用、現状はenvのみ登録・呼び出し未実装）
    paddlex_layout_endpoint: str = field(default_factory=lambda: _env("PADDLEX_LAYOUT_ENDPOINT", ""))

    # --- Search / Retrieval ---
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", "5"))
    search_mode: str = field(default_factory=lambda: _env("SEARCH_MODE", "hybrid"))
    # 検索結果の cross-encoder リランキング（最強レバー +11.8pt, EXPERIMENTS.md）→ 既定ON
    enable_rerank: bool = field(default_factory=lambda: _env_bool("ENABLE_RERANK", "true"))
    enable_japanese_search: bool = field(default_factory=lambda: _env_bool("ENABLE_JAPANESE_SEARCH", "true"))
    # グラフ三つ組の "A --rel--> B" 行をコンテキストに含めるか。
    # EXPERIMENTS.md: noLines がマルチホップで優位のため既定OFF。
    include_graph_lines: bool = field(default_factory=lambda: _env_bool("INCLUDE_GRAPH_LINES", "false"))

    # --- Knowledge Graph ---
    enable_knowledge_graph: bool = field(default_factory=lambda: _env_bool("ENABLE_KNOWLEDGE_GRAPH", "true"))
    graph_hop_count: int = field(default_factory=lambda: _env_int("GRAPH_HOP_COUNT", "2"))
    path_max_candidates: int = field(default_factory=lambda: _env_int("PATH_MAX_CANDIDATES", "30"))

    # --- Entity Vector Search ---
    enable_entity_vector_search: bool = field(default_factory=lambda: _env_bool("ENABLE_ENTITY_VECTOR_SEARCH", "true"))
    # 同義語/表記揺れ補完は高い類似度が必須（誤マッチ防止）。
    # 従来 pipeline 側で max(.,0.85) に切り上げられ 0.7 は死値だったため、
    # 正直な既定 0.85 を採用。0.85 未満も設定可能（下限切り上げは撤廃済み）。
    entity_similarity_threshold: float = field(default_factory=lambda: _env_float("ENTITY_SIMILARITY_THRESHOLD", "0.85"))

    # --- KG Source Chunks ---
    include_kg_source_chunks: bool = field(default_factory=lambda: _env_bool("INCLUDE_KG_SOURCE_CHUNKS", "true"))

    # --- 条件付き関係(qualifier/reify) ---
    # 全て既定OFF。規程・基準系コーパス + 条件起点(列挙/横断/閾値)の用途に限定。
    # 単一チャンクの条件照会では効果中立(A/B検証済み)なので blanket-on しない。
    # build側（build_kg.py 後処理で :CondFact/:Cond/[:WHEN] を抽出・格納）
    enable_conditional_facts: bool = field(default_factory=lambda: _env_bool("ENABLE_CONDITIONAL_FACTS", "false"))
    # entity_node_predicate の :CondFact/:Cond 除外 + consolidate の条件正規化を有効化
    enable_conditional_relations: bool = field(default_factory=lambda: _env_bool("CONDITIONAL_RELATIONS_ENABLED", "false"))
    # 条件機能を有効化するコレクションのホワイトリスト（カンマ区切り、空=全許可だが既定OFF）
    conditional_facts_corpus_tag: str = field(default_factory=lambda: _env("CONDITIONAL_FACTS_CORPUS_TAG", ""))
    # retrieval側（条件起点ルーティング + <CONDITION_FACTS> 表示）
    enable_condition_routing: bool = field(default_factory=lambda: _env_bool("ENABLE_CONDITION_ROUTING", "false"))
    include_condition_lines: bool = field(default_factory=lambda: _env_bool("INCLUDE_CONDITION_LINES", "false"))
    condition_routing_top_k: int = field(default_factory=lambda: _env_int("CONDITION_ROUTING_TOP_K", "20"))

    # --- Langfuse ---
    langfuse_public_key: str = field(default_factory=lambda: _env("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: str = field(default_factory=lambda: _env("LANGFUSE_SECRET_KEY"))
    langfuse_host: str = field(default_factory=lambda: _env("LANGFUSE_HOST"))


_settings: Settings | None = None


def get_settings() -> Settings:
    """シングルトンで Settings を返す"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """テスト用: Settings をリセット"""
    global _settings
    _settings = None


def build_pipeline_config(settings: "Settings | None" = None, **overrides) -> dict:
    """retrieval/KG パイプライン用 config dict を Settings から1か所で構築する。

    pipeline._cfg が読む全キーを網羅し、Settings.enable_entity_vector_search →
    ランタイムキー 'enable_entity_vector' のキー名変換もここだけで行う。
    batch_eval / app / sidebar はこの結果に対話的な上書きを overlay するだけにする。
    bench runner は config dict を明示的に組むため本ヘルパを経由せず影響を受けない。

    Args:
        settings: 省略時は get_settings()。
        **overrides: 返り値 dict を上書きするキー（例: app の session_state 値）。
    """
    s = settings or get_settings()
    cfg = {
        # --- DB / 接続 ---
        "pg_conn": s.pg_conn,
        "pg_collection": s.pg_collection,
        "neo4j_uri": s.neo4j_uri,
        "neo4j_user": s.neo4j_user,
        "neo4j_pw": s.neo4j_pw,
        # --- 検索 / リランク ---
        "retrieval_top_k": s.retrieval_top_k,
        "search_mode": s.search_mode,
        "enable_rerank": s.enable_rerank,
        "rerank_pool_size": 20,
        "enable_japanese_search": s.enable_japanese_search,
        # --- グラフ ---
        "graph_hop_count": s.graph_hop_count,
        "path_max_candidates": s.path_max_candidates,
        "include_graph_lines": s.include_graph_lines,
        # --- エンティティベクトル（キー名変換は唯一ここ） ---
        "enable_entity_vector": s.enable_entity_vector_search,
        "entity_similarity_threshold": s.entity_similarity_threshold,
        # --- KGソースチャンク ---
        "include_kg_source_chunks": s.include_kg_source_chunks,
        "kg_chunk_top_k": 5,
        # --- 参照追跡（既定OFF, FJH-11） ---
        "enable_reference_follow": False,
        "reference_follow_top_k": 5,
        # --- 条件付き関係の検索ルーティング（既定OFF・規程系限定） ---
        "enable_condition_routing": s.enable_condition_routing,
        "include_condition_lines": s.include_condition_lines,
        "condition_routing_top_k": s.condition_routing_top_k,
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
