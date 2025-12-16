#!/usr/bin/env python3
"""
test_vllm.py
============
VLLM接続テストスクリプト
"""

import os
import sys
import time
import requests
from statistics import mean, stdev
from dotenv import load_dotenv
from typing import List, Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm_client import VLLMClient, VLLMChatClient
from llm_factory import create_chat_llm, get_llm_provider_info


def check_vllm_health(endpoint: str) -> bool:
    """VLLMサーバーのヘルスチェック"""
    try:
        response = requests.get(f"{endpoint}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def test_vllm_connection():
    """VLLM接続テスト"""
    # 設定を読み込み
    load_dotenv()

    print("=" * 60)
    print("VLLM Connection Test")
    print("=" * 60)

    # Check provider configuration
    provider_info = get_llm_provider_info()
    print(f"Provider: {provider_info['provider']}")
    print(f"Status: {provider_info['status']}")

    if provider_info['provider'] == 'VLLM':
        print(f"Endpoint: {provider_info['endpoint']}")

        # Health check
        vllm_endpoint = os.getenv("VLLM_ENDPOINT")
        if vllm_endpoint:
            health = check_vllm_health(vllm_endpoint)
            print(f"Health Check: {'✅ Server is responding' if health else '❌ Server not responding'}")
    else:
        print(f"Model: {provider_info['model']}")
        print(f"Endpoint: {provider_info['endpoint']}")

    print("-" * 60)

    try:
        # Use factory to create LLM
        client = create_chat_llm(temperature=0.0)

        # テストプロンプト
        test_prompts = [
            "こんにちは。簡単に自己紹介してください。",
            "1から5までの数字を日本語で書いてください。",
            "Pythonとは何ですか？簡潔に説明してください。"
        ]

        for i, test_prompt in enumerate(test_prompts, 1):
            print(f"\n[Test {i}]: {test_prompt}")
            print("[Sending request...]")

            start_time = time.time()

            # Execute
            if hasattr(client, 'invoke'):
                response = client.invoke(test_prompt)
                # Extract content if it's an AIMessage
                if hasattr(response, 'content'):
                    response = response.content
            else:
                response = str(client(test_prompt))

            end_time = time.time()
            latency = end_time - start_time

            print(f"✅ Success! (Latency: {latency:.2f}s)")
            print(f"[Response]: {response[:200]}{'...' if len(response) > 200 else ''}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_vllm(num_tests: int = 3):
    """パフォーマンステスト"""
    load_dotenv()

    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    try:
        client = create_chat_llm(temperature=0.0)

        test_prompts = [
            "1から10までを数えてください。",
            "機械学習の基本概念を50文字以内で説明してください。",
            "「こんにちは」を5つの言語で書いてください。"
        ]

        latencies = []

        for prompt in test_prompts:
            print(f"\nPrompt: {prompt[:50]}...")
            prompt_latencies = []

            for i in range(num_tests):
                start = time.time()

                if hasattr(client, 'invoke'):
                    response = client.invoke(prompt)
                    if hasattr(response, 'content'):
                        response = response.content
                else:
                    response = str(client(prompt))

                end = time.time()
                latency = end - start
                prompt_latencies.append(latency)
                latencies.append(latency)

                print(f"  Test {i+1}: {latency:.2f}s")

            print(f"  Average: {mean(prompt_latencies):.2f}s")

        print("\n" + "-" * 60)
        print(f"Overall Average Latency: {mean(latencies):.2f}s")
        if len(latencies) > 1:
            print(f"Standard Deviation: {stdev(latencies):.2f}s")
        print(f"Min: {min(latencies):.2f}s, Max: {max(latencies):.2f}s")

    except Exception as e:
        print(f"\n❌ Benchmark Error: {e}")
        import traceback
        traceback.print_exc()


def test_factory_switching():
    """Factory pattern switching test"""
    load_dotenv()

    print("\n" + "=" * 60)
    print("Factory Pattern Test")
    print("=" * 60)

    original_provider = os.environ.get("LLM_PROVIDER", "azure_openai")

    # Test Azure OpenAI mode
    os.environ["LLM_PROVIDER"] = "azure_openai"
    info = get_llm_provider_info()
    print(f"\nAzure OpenAI Mode:")
    print(f"  Provider: {info['provider']}")
    print(f"  Model: {info['model']}")

    # Test VLLM mode
    os.environ["LLM_PROVIDER"] = "vllm"
    info = get_llm_provider_info()
    print(f"\nVLLM Mode:")
    print(f"  Provider: {info['provider']}")
    print(f"  Endpoint: {info['endpoint']}")

    # Restore original
    os.environ["LLM_PROVIDER"] = original_provider
    print(f"\nRestored to: {original_provider}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLLM Integration Test")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--factory", action="store_true", help="Test factory pattern switching")
    parser.add_argument("--num-tests", type=int, default=3, help="Number of benchmark tests")

    args = parser.parse_args()

    if args.factory:
        test_factory_switching()
    elif args.benchmark:
        benchmark_vllm(args.num_tests)
    else:
        success = test_vllm_connection()

        if success:
            print("\n✨ LLM integration is working correctly!")
        else:
            print("\n⚠️  Please check your LLM configuration")
            print("\nTroubleshooting tips:")
            print("1. Check LLM_PROVIDER environment variable (azure_openai or vllm)")
            print("2. For VLLM: Ensure VLLM_ENDPOINT is set correctly")
            print("3. For Azure: Ensure all AZURE_OPENAI_* variables are set")
            print("4. Check network connectivity to the endpoint")