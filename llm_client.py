from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI, OpenAIError

from config import settings


"""
异步:AsyncOpenAI 与 FastAPI 的 async 端点天然配合,一个进程可并发处理大量请求
超时与重试:在构造时统一配置,SDK 自动处理;不需要手写 retry 循环
统一异常:所有 SDK 错误包装成 LLMError,调用方(HTTP 层)只处理一种异常
空内容防御:content is None 时给出明确错误,而不是让调用方拿到 None 再猜
模型特有参数(如 DeepSeek 的 thinking 配置),后续按需给 chat() 增加参数
"""


class LLMError(Exception):
    """业务层统一异常。调用方只认识这个，不认识 SDK 异常。"""


class LLMClient:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_url,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """非流式调用，返回完整回答。"""
        kwargs: dict = {}
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            resp = await self._client.chat.completions.create(
                model=model or settings.deepseek_model,
                messages=messages,
                stream=False,
                **kwargs,
            )
        except OpenAIError as exc:
            raise LLMError(f"LLM 调用失败：{exc}") from exc

        content = resp.choices[0].message.content
        if content is None:
            raise LLMError("模型返回了空内容")
        return content

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """流式调用，逐段产出文本（配合 FastAPI StreamingResponse 使用）。"""
        try:
            stream = await self._client.chat.completions.create(
                model=model or settings.deepseek_model,
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except OpenAIError as exc:
            raise LLMError(f"LLM 流式调用失败：{exc}") from exc


llm_client = LLMClient()