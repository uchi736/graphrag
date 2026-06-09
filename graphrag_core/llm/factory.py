"""
llm_factory.py
==============
LLM Factory Pattern for switching between Azure OpenAI and VLLM
"""

from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

# config.py で load_dotenv() 済み
from graphrag_core.config import get_settings


def create_chat_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model_override: Optional[str] = None
) -> Any:
    """
    Create a chat LLM instance based on configuration.

    Args:
        temperature: Override temperature setting
        max_tokens: Override max_tokens setting
        model_override: Override the model name (for specific use cases)

    Returns:
        Either AzureChatOpenAI or VLLMChatClient instance
    """
    s = get_settings()

    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        if not s.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        openai_temperature = temperature if temperature is not None else 0.0
        openai_max_tokens = max_tokens if max_tokens is not None else 4096
        logger.info(f"Using OpenAI model: {s.openai_model}")

        return ChatOpenAI(
            api_key=s.openai_api_key,
            model=s.openai_model,
            temperature=openai_temperature,
            max_tokens=openai_max_tokens,
            timeout=s.openai_timeout,
        )

    if s.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not s.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        anthropic_temperature = temperature if temperature is not None else 0.0
        anthropic_max_tokens = max_tokens if max_tokens is not None else 4096
        logger.info(f"Using Anthropic model: {s.anthropic_model}")

        return ChatAnthropic(
            api_key=s.anthropic_api_key,
            model=s.anthropic_model,
            temperature=anthropic_temperature,
            max_tokens=anthropic_max_tokens,
            timeout=s.anthropic_timeout,
        )

    if s.llm_provider == "vllm":
        # Use VLLM via ChatOpenAI (OpenAI互換API経由)
        from langchain_openai import ChatOpenAI

        if not s.vllm_endpoint:
            logger.error("VLLM_ENDPOINT not configured")
            raise ValueError("VLLM_ENDPOINT environment variable is required when LLM_PROVIDER=vllm")

        vllm_temperature = temperature if temperature is not None else s.vllm_temperature
        vllm_max_tokens = max_tokens if max_tokens is not None else s.vllm_max_tokens

        logger.info(f"Using VLLM at {s.vllm_endpoint} with model {s.vllm_model}")

        # モデル別のchat_template_kwargs調整
        extra_body = None
        model_lower = s.vllm_model.lower()
        if "qwen3" in model_lower:
            # Qwen3: <think>ブロック抑制で高速化
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        elif "gpt-oss" in model_lower:
            # gpt-oss (harmony format): reasoning_effort を low/medium/high で指定
            extra_body = {"chat_template_kwargs": {"reasoning_effort": s.vllm_reasoning_effort}}
        # Gemma 系は chat_template_kwargs 不要（thinking モード非搭載）

        return ChatOpenAI(
            base_url=s.vllm_endpoint,
            api_key=s.vllm_api_key,
            model=s.vllm_model,
            temperature=vllm_temperature,
            max_tokens=vllm_max_tokens,
            timeout=s.vllm_timeout,
            extra_body=extra_body,
        )

    else:
        # Use Azure OpenAI (default)
        from langchain_openai import AzureChatOpenAI

        if not all([s.azure_openai_chat_deployment, s.azure_openai_api_version,
                    s.azure_openai_endpoint, s.azure_openai_api_key]):
            logger.error("Azure OpenAI configuration incomplete")
            raise ValueError("Azure OpenAI environment variables are required when LLM_PROVIDER=azure_openai")

        azure_temperature = temperature if temperature is not None else 0.0
        azure_max_tokens = max_tokens if max_tokens is not None else 4096

        logger.info(f"Using Azure OpenAI deployment: {s.azure_openai_chat_deployment}")

        return AzureChatOpenAI(
            azure_deployment=s.azure_openai_chat_deployment,
            openai_api_version=s.azure_openai_api_version,
            azure_endpoint=s.azure_openai_endpoint,
            api_key=s.azure_openai_api_key,
            temperature=azure_temperature,
            max_tokens=azure_max_tokens
        )


def create_embeddings() -> Any:
    """Create an embeddings instance based on EMBEDDING_PROVIDER.

    Returns:
        Either AzureOpenAIEmbeddings or OpenAIEmbeddings (vLLM) instance.
    """
    s = get_settings()

    if s.embedding_provider == "vllm":
        from langchain_openai import OpenAIEmbeddings

        if not s.vllm_embedding_endpoint:
            logger.error("VLLM_EMBEDDING_ENDPOINT not configured")
            raise ValueError("VLLM_EMBEDDING_ENDPOINT is required when EMBEDDING_PROVIDER=vllm")

        logger.info(f"Using vLLM embeddings at {s.vllm_embedding_endpoint} model={s.vllm_embedding_model}")

        return OpenAIEmbeddings(
            base_url=s.vllm_embedding_endpoint,
            api_key=s.vllm_embedding_api_key,
            model=s.vllm_embedding_model,
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )

    # default: Azure OpenAI
    from langchain_openai import AzureOpenAIEmbeddings

    if not all([s.azure_openai_embedding_deployment, s.azure_openai_api_version,
                s.azure_openai_endpoint, s.azure_openai_api_key]):
        logger.error("Azure OpenAI embedding configuration incomplete")
        raise ValueError("Azure OpenAI embedding environment variables are required when EMBEDDING_PROVIDER=azure_openai")

    logger.info(f"Using Azure OpenAI embedding deployment: {s.azure_openai_embedding_deployment}")

    return AzureOpenAIEmbeddings(
        azure_deployment=s.azure_openai_embedding_deployment,
        openai_api_version=s.azure_openai_api_version,
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_api_key,
    )


def get_llm_provider_info() -> dict:
    """
    Get information about the current LLM provider configuration.

    Returns:
        Dictionary with provider information
    """
    s = get_settings()

    if s.llm_provider == "vllm":
        return {
            "provider": "VLLM",
            "endpoint": s.vllm_endpoint or "Not configured",
            "model": s.vllm_model or "Not configured",
            "status": "VLLM Mode" if s.vllm_endpoint else "VLLM not configured"
        }
    else:
        return {
            "provider": "Azure OpenAI",
            "endpoint": s.azure_openai_endpoint or "Not configured",
            "model": s.azure_openai_chat_deployment or "Not configured",
            "status": "Azure OpenAI Mode" if s.azure_openai_chat_deployment else "Azure not configured"
        }
