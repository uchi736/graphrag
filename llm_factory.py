"""
llm_factory.py
==============
LLM Factory Pattern for switching between Azure OpenAI and VLLM
"""

import os
from typing import Optional, Any
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


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

    # Check which provider to use
    llm_provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()

    if llm_provider == "vllm":
        # Use VLLM
        from vllm_client import VLLMChatClient

        endpoint = os.getenv("VLLM_ENDPOINT")
        if not endpoint:
            logger.error("VLLM_ENDPOINT not configured")
            raise ValueError("VLLM_ENDPOINT environment variable is required when LLM_PROVIDER=vllm")

        # Get VLLM-specific settings
        vllm_temperature = temperature if temperature is not None else float(os.getenv("VLLM_TEMPERATURE", "0.0"))
        vllm_top_p = float(os.getenv("VLLM_TOP_P", "0.7"))
        vllm_top_k = int(os.getenv("VLLM_TOP_K", "5"))
        vllm_min_p = float(os.getenv("VLLM_MIN_P", "0.0"))
        vllm_max_tokens = max_tokens if max_tokens is not None else int(os.getenv("VLLM_MAX_TOKENS", "4096"))
        vllm_reasoning_effort = os.getenv("VLLM_REASONING_EFFORT", "medium")
        vllm_timeout = int(os.getenv("VLLM_TIMEOUT", "60"))

        logger.info(f"Using VLLM at {endpoint}")

        return VLLMChatClient(
            endpoint=endpoint,
            temperature=vllm_temperature,
            top_p=vllm_top_p,
            top_k=vllm_top_k,
            min_p=vllm_min_p,
            max_tokens=vllm_max_tokens,
            reasoning_effort=vllm_reasoning_effort,
            timeout=vllm_timeout
        )

    else:
        # Use Azure OpenAI (default)
        from langchain_openai import AzureChatOpenAI

        # Get Azure OpenAI settings
        azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not all([azure_deployment, azure_api_version, azure_endpoint, azure_api_key]):
            logger.error("Azure OpenAI configuration incomplete")
            raise ValueError("Azure OpenAI environment variables are required when LLM_PROVIDER=azure_openai")

        # Use temperature override or default to 0
        azure_temperature = temperature if temperature is not None else 0.0
        azure_max_tokens = max_tokens if max_tokens is not None else 4096

        logger.info(f"Using Azure OpenAI deployment: {azure_deployment}")

        return AzureChatOpenAI(
            azure_deployment=azure_deployment,
            openai_api_version=azure_api_version,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            temperature=azure_temperature,
            max_tokens=azure_max_tokens
        )


def create_standard_llm(
    model: str = "gpt-4o-mini",
    temperature: Optional[float] = None
) -> Any:
    """
    Create a standard LLM instance (for graphrag.py compatibility).

    Args:
        model: Model name (used only for Azure/OpenAI, ignored for VLLM)
        temperature: Override temperature setting

    Returns:
        Either ChatOpenAI or VLLMClient instance
    """

    # Check which provider to use
    llm_provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()

    if llm_provider == "vllm":
        # Use VLLM (ignore model parameter as VLLM serves single model)
        from vllm_client import VLLMClient

        endpoint = os.getenv("VLLM_ENDPOINT")
        if not endpoint:
            logger.error("VLLM_ENDPOINT not configured")
            raise ValueError("VLLM_ENDPOINT environment variable is required when LLM_PROVIDER=vllm")

        # Get VLLM-specific settings
        vllm_temperature = temperature if temperature is not None else float(os.getenv("VLLM_TEMPERATURE", "0.0"))
        vllm_top_p = float(os.getenv("VLLM_TOP_P", "0.7"))
        vllm_top_k = int(os.getenv("VLLM_TOP_K", "5"))
        vllm_min_p = float(os.getenv("VLLM_MIN_P", "0.0"))
        vllm_max_tokens = int(os.getenv("VLLM_MAX_TOKENS", "4096"))
        vllm_reasoning_effort = os.getenv("VLLM_REASONING_EFFORT", "medium")
        vllm_timeout = int(os.getenv("VLLM_TIMEOUT", "60"))

        logger.info(f"Using VLLM at {endpoint} (model param ignored)")

        return VLLMClient(
            endpoint=endpoint,
            temperature=vllm_temperature,
            top_p=vllm_top_p,
            top_k=vllm_top_k,
            min_p=vllm_min_p,
            max_tokens=vllm_max_tokens,
            reasoning_effort=vllm_reasoning_effort,
            timeout=vllm_timeout
        )

    else:
        # Use standard OpenAI/Azure OpenAI
        # Check if we should use Azure or regular OpenAI
        if os.getenv("AZURE_OPENAI_API_KEY"):
            # Use Azure OpenAI with ChatOpenAI interface
            from langchain_openai import ChatOpenAI

            # For graphrag.py, we use the base ChatOpenAI with azure configuration
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
            azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

            # Build the base URL for Azure
            base_url = f"{azure_endpoint}/openai/deployments/{azure_deployment}"

            logger.info(f"Using Azure OpenAI via ChatOpenAI interface: {model}")

            return ChatOpenAI(
                model=model,  # This is used for logging, actual model is determined by deployment
                temperature=temperature if temperature is not None else 0.0,
                base_url=base_url,
                api_key=azure_api_key,
                default_headers={"api-version": azure_api_version}
            )
        else:
            # Use regular OpenAI
            from langchain_openai import ChatOpenAI

            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error("No OpenAI API key configured")
                raise ValueError("OPENAI_API_KEY environment variable is required")

            logger.info(f"Using OpenAI model: {model}")

            return ChatOpenAI(
                model=model,
                temperature=temperature if temperature is not None else 0.0,
                api_key=openai_api_key
            )


def get_llm_provider_info() -> dict:
    """
    Get information about the current LLM provider configuration.

    Returns:
        Dictionary with provider information
    """
    llm_provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()

    if llm_provider == "vllm":
        return {
            "provider": "VLLM",
            "endpoint": os.getenv("VLLM_ENDPOINT", "Not configured"),
            "model": "VLLM-hosted model",
            "status": "✅ VLLM Mode" if os.getenv("VLLM_ENDPOINT") else "❌ VLLM not configured"
        }
    else:
        azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "Not configured")
        return {
            "provider": "Azure OpenAI",
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "Not configured"),
            "model": azure_deployment,
            "status": "✅ Azure OpenAI Mode" if azure_deployment != "Not configured" else "❌ Azure not configured"
        }