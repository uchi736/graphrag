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
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "azure_openai"))

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
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "azure_openai"))
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
    enable_rerank: bool = field(default_factory=lambda: _env_bool("ENABLE_RERANK", "false"))
    enable_japanese_search: bool = field(default_factory=lambda: _env_bool("ENABLE_JAPANESE_SEARCH", "true"))

    # --- Knowledge Graph ---
    enable_knowledge_graph: bool = field(default_factory=lambda: _env_bool("ENABLE_KNOWLEDGE_GRAPH", "true"))
    graph_hop_count: int = field(default_factory=lambda: _env_int("GRAPH_HOP_COUNT", "2"))
    path_max_candidates: int = field(default_factory=lambda: _env_int("PATH_MAX_CANDIDATES", "30"))

    # --- Entity Vector Search ---
    enable_entity_vector_search: bool = field(default_factory=lambda: _env_bool("ENABLE_ENTITY_VECTOR_SEARCH", "true"))
    entity_similarity_threshold: float = field(default_factory=lambda: _env_float("ENTITY_SIMILARITY_THRESHOLD", "0.7"))

    # --- KG Source Chunks ---
    include_kg_source_chunks: bool = field(default_factory=lambda: _env_bool("INCLUDE_KG_SOURCE_CHUNKS", "true"))

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
