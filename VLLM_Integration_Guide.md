# VLLM統合実装ガイド

## 概要
このガイドは、VLLMサーバーを経由してLLM（Large Language Model）を呼び出すための実装手順を詳細に説明します。OpenAI APIの代わりにVLLMサーバーを使用することで、オープンソースモデルをセルフホスティングして利用できます。

## 目次
1. [必要な前提条件](#1-必要な前提条件)
2. [VLLMサーバーのセットアップ](#2-vllmサーバーのセットアップ)
3. [クライアント実装](#3-クライアント実装)
4. [設定ファイルの準備](#4-設定ファイルの準備)
5. [LangChainとの統合](#5-langchainとの統合)
6. [テストと検証](#6-テストと検証)
7. [トラブルシューティング](#7-トラブルシューティング)

---

## 1. 必要な前提条件

### 必要なパッケージ
```bash
pip install requests langchain-core python-dotenv
```

### VLLMサーバー側の要件
- VLLMがインストールされたサーバー
- 対応するLLMモデル（例：Qwen2.5-32B-Instruct）
- GPU（モデルサイズに応じたVRAM）

---

## 2. VLLMサーバーのセットアップ

### VLLMサーバーの起動コマンド例
```bash
vllm serve Qwen/Qwen2.5-32B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9
```

### トンネリング（外部アクセス用）
ローカルサーバーを外部からアクセス可能にする場合：

```bash
# localtunnelを使用
npx localtunnel --port 8000

# またはngrokを使用
ngrok http 8000
```

---

## 3. クライアント実装

### 3.1 VLLMクライアントクラスの実装

`vllm_client.py`を作成：

```python
"""
vllm_client.py
==============
VLLMサーバー用のカスタムLLMクライアント
modelパラメータを送信しない専用実装
"""

import requests
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import logging

logger = logging.getLogger(__name__)


class VLLMClient(LLM):
    """VLLM専用のLLMクライアント（modelパラメータを送信しない）"""

    endpoint: str
    temperature: float = 0.0
    top_p: float = 0.7
    top_k: int = 5
    min_p: float = 0.0
    reasoning_effort: str = "medium"
    timeout: int = 60

    @property
    def _llm_type(self) -> str:
        """Return identifier of LLM type."""
        return "vllm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        VLLMを呼び出す（動作確認済みコードと同じ形式）
        重要: modelパラメータを含めない
        """

        url = f"{self.endpoint}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer no-key"  # VLLMはトークン不要だが形式として必要
        }

        # modelパラメータを含めない（これが重要！）
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "reasoning_effort": self.reasoning_effort,
        }

        # stopワードが指定されている場合
        if stop:
            payload["stop"] = stop

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()

            if "choices" in data and data["choices"]:
                content = data["choices"][0].get("message", {}).get("content", "")
                return content

            logger.warning("Empty response from VLLM")
            return ""

        except requests.exceptions.Timeout:
            logger.error(f"VLLM request timeout after {self.timeout} seconds")
            raise Exception(f"VLLM request timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"VLLM API request failed: {e}")
            raise Exception(f"VLLM API call failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calling VLLM: {e}")
            raise Exception(f"VLLM API call failed: {e}")

    def invoke(self, input: Any, **kwargs) -> str:
        """
        LangChain互換のinvokeメソッド
        """
        if isinstance(input, str):
            return self._call(input)
        elif hasattr(input, 'content'):
            return self._call(input.content)
        else:
            return self._call(str(input))

    async def ainvoke(self, input: Any, **kwargs) -> str:
        """
        非同期版のinvokeメソッド（同期版を呼び出す）
        """
        return self.invoke(input, **kwargs)
```

### 3.2 重要なポイント

1. **modelパラメータを送信しない**
   - VLLMサーバーは単一モデルを実行するため、modelパラメータは不要
   - OpenAI APIとの主な違いの一つ

2. **エンドポイントの形式**
   - `{base_url}/v1/chat/completions` の形式を使用
   - OpenAI API互換のエンドポイント

3. **認証ヘッダー**
   - `Authorization: Bearer no-key` を含める
   - VLLMは実際には認証不要だが、API形式の互換性のため必要

---

## 4. 設定ファイルの準備

### 4.1 環境変数ファイル（.env）

```env
# VLLM設定
USE_VLLM=true
VLLM_ENDPOINT=https://your-tunnel.loca.lt/v1
VLLM_MODEL_NAME=qwen-2.5-32b-instruct
VLLM_REASONING_EFFORT=medium

# その他の設定
TEMPERATURE=0.0
TOP_P=0.7
TOP_K=5
MIN_P=0.0
```

### 4.2 設定クラスの実装

```python
"""
config.py
=========
設定管理クラス
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

@dataclass
class Config:
    """設定管理クラス"""

    # VLLM設定
    use_vllm: bool = False
    vllm_endpoint: Optional[str] = None
    vllm_model_name: str = ""
    vllm_reasoning_effort: str = ""

    # LLMパラメータ
    temperature: float = 0.0
    top_p: float = 0.7
    top_k: int = 5
    min_p: float = 0.0

    def __post_init__(self):
        """環境変数から設定を読み込み"""
        load_dotenv()

        # VLLM設定
        self.use_vllm = os.getenv("USE_VLLM", "false").lower() == "true"
        self.vllm_endpoint = os.getenv("VLLM_ENDPOINT")
        self.vllm_model_name = os.getenv("VLLM_MODEL_NAME", "gpt-oss")
        self.vllm_reasoning_effort = os.getenv("VLLM_REASONING_EFFORT", "medium")

        # LLMパラメータ
        self.temperature = float(os.getenv("TEMPERATURE", "0.0"))
        self.top_p = float(os.getenv("TOP_P", "0.7"))
        self.top_k = int(os.getenv("TOP_K", "5"))
        self.min_p = float(os.getenv("MIN_P", "0.0"))
```

---

## 5. LangChainとの統合

### 5.1 基本的な使用例

```python
from config import Config
from vllm_client import VLLMClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 設定を読み込み
config = Config()

# VLLMクライアントを初期化
llm = VLLMClient(
    endpoint=config.vllm_endpoint,
    temperature=config.temperature,
    top_p=config.top_p,
    top_k=config.top_k,
    min_p=config.min_p,
    reasoning_effort=config.vllm_reasoning_effort,
    timeout=60
)

# LangChainのチェーンを構築
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは親切なアシスタントです。"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

# 実行
response = chain.invoke({"input": "こんにちは！"})
print(response)
```

### 5.2 RAGシステムでの統合例

```python
class RAGSystem:
    def __init__(self, config: Config):
        self.config = config

        # LLMの選択
        if config.use_vllm:
            # VLLMを使用
            from vllm_client import VLLMClient
            self.llm = VLLMClient(
                endpoint=config.vllm_endpoint,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                min_p=config.min_p,
                reasoning_effort=config.vllm_reasoning_effort
            )
        else:
            # OpenAI APIを使用
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=config.temperature
            )

    def generate_answer(self, query: str, context: str) -> str:
        """RAGによる回答生成"""
        prompt = ChatPromptTemplate.from_template(
            "Context: {context}\n\n"
            "Question: {query}\n\n"
            "Answer:"
        )

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "context": context})
```

---

## 6. テストと検証

### 6.1 接続テストスクリプト

```python
#!/usr/bin/env python3
"""
test_vllm.py
============
VLLM接続テスト
"""

from dotenv import load_dotenv
from config import Config
from vllm_client import VLLMClient

def test_vllm_connection():
    """VLLM接続テスト"""
    # 設定を読み込み
    load_dotenv()
    cfg = Config()

    print("=" * 60)
    print("VLLM Connection Test")
    print("=" * 60)
    print(f"Endpoint: {cfg.vllm_endpoint}")
    print("-" * 60)

    try:
        # VLLMClientを初期化
        client = VLLMClient(
            endpoint=cfg.vllm_endpoint,
            temperature=0.0,
            top_p=0.7,
            top_k=5,
            min_p=0.0,
            reasoning_effort="medium",
            timeout=30
        )

        # テストプロンプト
        test_prompt = "こんにちは。簡単に自己紹介してください。"
        print(f"\n[Test Prompt]: {test_prompt}")
        print("[Sending request...]")

        # 実行
        response = client.invoke(test_prompt)

        print(f"\n✅ Success!")
        print(f"[Response]: {response[:200]}...")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_vllm_connection()

    if success:
        print("\n✨ VLLM is working correctly!")
    else:
        print("\n⚠️  Please check your VLLM server and configuration")
```

### 6.2 パフォーマンステスト

```python
import time
from statistics import mean, stdev

def benchmark_vllm(client: VLLMClient, num_tests: int = 5):
    """パフォーマンステスト"""
    test_prompts = [
        "1から10までを数えてください。",
        "Pythonとは何ですか？",
        "機械学習の基本概念を説明してください。",
    ]

    latencies = []

    for prompt in test_prompts:
        for _ in range(num_tests):
            start = time.time()
            response = client.invoke(prompt)
            end = time.time()

            latency = end - start
            latencies.append(latency)
            print(f"Prompt: {prompt[:30]}... | Latency: {latency:.2f}s")

    print(f"\n平均レイテンシ: {mean(latencies):.2f}s")
    print(f"標準偏差: {stdev(latencies):.2f}s")
```

---

## 7. トラブルシューティング

### 7.1 よくある問題と解決方法

#### 問題1: "model is required" エラー
**原因**: OpenAI API形式でmodelパラメータを送信している
**解決**: VLLMClientを使用し、modelパラメータを送信しない

#### 問題2: タイムアウトエラー
**原因**:
- VLLMサーバーが起動していない
- ネットワーク接続の問題
- モデルの初回ロードに時間がかかる

**解決**:
```python
# タイムアウトを増やす
client = VLLMClient(
    endpoint=endpoint,
    timeout=120  # 2分に増やす
)
```

#### 問題3: 接続拒否エラー
**原因**: エンドポイントURLが正しくない
**解決**:
- URLの形式を確認（末尾に`/v1`を含める）
- トンネルが有効か確認

### 7.2 デバッグ用ログ設定

```python
import logging

# デバッグログを有効化
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# リクエスト内容を確認
import requests
from requests import PreparedRequest

def debug_request(req: PreparedRequest):
    print(f"URL: {req.url}")
    print(f"Headers: {req.headers}")
    print(f"Body: {req.body}")
```

### 7.3 ヘルスチェック実装

```python
def check_vllm_health(endpoint: str) -> bool:
    """VLLMサーバーのヘルスチェック"""
    try:
        response = requests.get(f"{endpoint}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# 使用例
if not check_vllm_health(config.vllm_endpoint):
    print("Warning: VLLM server is not responding")
```

---

## 8. ベストプラクティス

### 8.1 エラーハンドリング
```python
def safe_llm_call(client: VLLMClient, prompt: str, max_retries: int = 3):
    """リトライ機能付きの安全なLLM呼び出し"""
    for attempt in range(max_retries):
        try:
            return client.invoke(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # 指数バックオフ
```

### 8.2 コネクションプーリング
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# セッションを再利用
session = requests.Session()
retry = Retry(total=3, backoff_factor=0.3)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)
```

### 8.3 キャッシング
```python
from functools import lru_cache
import hashlib

class CachedVLLMClient(VLLMClient):
    @lru_cache(maxsize=128)
    def _cached_call(self, prompt_hash: str, prompt: str) -> str:
        return super()._call(prompt)

    def _call(self, prompt: str, **kwargs) -> str:
        # プロンプトのハッシュを計算
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return self._cached_call(prompt_hash, prompt)
```

---

## まとめ

このガイドに従うことで、VLLMサーバーを使用したLLM統合を実装できます。重要なポイント：

1. **modelパラメータを送信しない** - VLLMサーバーの特徴
2. **適切なエンドポイント形式** - `/v1`を含める
3. **エラーハンドリング** - タイムアウトとリトライ
4. **LangChain互換性** - 標準的なインターフェース実装

質問や問題がある場合は、VLLMのドキュメントを参照するか、このガイドのトラブルシューティングセクションを確認してください。