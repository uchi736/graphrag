"""
vllm_client.py
==============
VLLMサーバー用のカスタムLLMクライアント
modelパラメータを送信しない専用実装
"""

import requests
from typing import Any, List, Optional, Dict
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage
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
    max_tokens: int = 4096
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
            "max_tokens": self.max_tokens,
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

    def invoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> str:
        """
        LangChain互換のinvokeメソッド
        """
        if isinstance(input, str):
            return self._call(input, **kwargs)
        elif isinstance(input, dict):
            # MessagesPlaceholder などからの入力に対応
            if "messages" in input:
                # メッセージリストから最後のユーザーメッセージを取得
                messages = input["messages"]
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, "content"):
                        return self._call(last_message.content, **kwargs)
                    elif isinstance(last_message, dict) and "content" in last_message:
                        return self._call(last_message["content"], **kwargs)
            # 通常のプロンプトテンプレートからの入力
            elif len(input) == 1:
                # 単一のキーバリューペアの場合、値を使用
                return self._call(str(list(input.values())[0]), **kwargs)
            else:
                # 複数のキーがある場合は文字列化
                return self._call(str(input), **kwargs)
        elif hasattr(input, 'content'):
            return self._call(input.content, **kwargs)
        else:
            return self._call(str(input), **kwargs)

    async def ainvoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> str:
        """
        非同期版のinvokeメソッド（同期版を呼び出す）
        """
        return self.invoke(input, config, **kwargs)

    def predict(self, text: str, *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        LangChain互換のpredictメソッド
        """
        return self._call(text, stop=stop, **kwargs)

    async def apredict(self, text: str, *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        非同期版のpredictメソッド
        """
        return self.predict(text, stop=stop, **kwargs)


class VLLMChatClient(VLLMClient):
    """
    ChatModelインターフェース互換のVLLMクライアント
    AzureChatOpenAI の代替として使用可能
    """

    def invoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> Any:
        """
        ChatModel互換のinvokeメソッド
        AIMessageオブジェクトを返すために、簡易的なラッパーを提供
        """
        # VLLMClient.invoke を直接呼び出し、文字列レスポンスを取得
        response_text = super().invoke(input, config, **kwargs)

        # LangChain標準のAIMessageで返す（contentがNoneでも空文字に）
        return AIMessage(content=response_text or "")

    async def ainvoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> Any:
        """
        非同期版のChatModel互換invokeメソッド
        """
        # 非同期でも二重ラップを避けるために、同期 invoke をそのまま呼ぶ
        return self.invoke(input, config, **kwargs)