from __future__ import annotations
import json
from typing import AsyncIterator, Any, Callable

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


def _parse_json(content: str) -> dict:
    """解析模型输出的 JSON,容错处理围栏和杂散文本"""
    content = content.strip("` ").removeprefix('JSON').removeprefix('json')
    
    try:
        content_json = json.loads(content)
    except json.JSONDecodeError as exc:
        start = content.find("{")
        end = content.rfind("}")

        if start != -1 and end != -1 and start < end:
            content_cut = content[start : end + 1]
        else:
            raise LLMError(f"解析失败:{exc}" + " JSON：" + content[:200])

        try:
            content_json = json.loads(content_cut)
        except json.JSONDecodeError as exc:
            raise LLMError(f"解析失败:{exc}" + " JSON：" + content[:200])
    return content_json


class LLMClient:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_url,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    async def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | None = None
    ) -> Any:
        """负责发请求 + 统一异常包装, 返回 message"""
        # 整理参数
        kwargs: dict = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_format is not None:
            kwargs["response_format"] = response_format
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
    
        try:
            resp = await self._client.chat.completions.create(
                model=model or settings.deepseek_model,
                messages=messages,
                stream=False,
                **kwargs,
            )
        except OpenAIError as exc:
            #OpenAIError 包装为 LLMError
            raise LLMError(f"LLM 调用失败：{exc}") from exc

        return resp.choices[0].message

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """非流式调用，返回完整回答。"""
        msg = await self._complete(
            messages=messages, 
            model=model, 
            temperature=temperature,
        )

        content = msg.content
        if content is None:
            raise LLMError("模型返回了空内容")
        return content

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict:
        """prompt 里必须出现 json 字样"""
        msg = await self._complete(
            messages=messages,
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        content = msg.content
        if content is None:
            raise LLMError("模型返回了空内容")
        return _parse_json(content)


    async def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        *,
        model: str | None = None,
        tool_choice: str | None = None,
    ) -> tuple[str | None, list | None]:
        """单次工具调用:模型可能直接回答,也可能提议调用工具。

        返回 (content, tool_calls):
        - tool_calls 为 None → 模型直接回答了,content 是回答文本;
        - tool_calls 有值 → 模型提议调用工具,content 通常为 None,
          需要调用方执行工具后把结果以 role="tool" 消息回传。
        """
        msg = await self._complete(
            messages=messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )

        return (msg.content, msg.tool_calls)


    async def run_tool_loop(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_impls: dict[str, Callable[..., object]],
        *,
        model: str | None = None,
        max_iterations: int = 5,
    ) -> str:
        """
        执行过程:
        - 每轮调用 chat_with_tools,携带完整对话历史;
        - 模型返回 tool_calls 时,执行 tool_impls 中对应的函数,
        把结果以 role="tool" 消息回传给模型,再进入下一轮;
        - 模型不再调用工具时,返回它的最终回答。
    
        参数:
            messages: 初始对话历史,函数内部会复制一份,不会修改调用方的列表。
            tools: 工具 schema 列表,原样传给模型(JSON Schema 格式)。
            tool_impls: 工具名 → 实现函数的映射,模型提议调用的工具必须在这里。
            model: 可选,覆盖默认模型。
            max_iterations: 最多请求轮数,防止模型反复调用工具陷入死循环。
    
        返回:
            模型的最终回答文本。
    
        异常:
            LLMError: 工具不存在、模型既没调工具也没返回内容、
                    或超过 max_iterations 轮仍未结束。
        """
        history = list(messages)

        for _ in range(max_iterations):
            content, tool_calls = await self.chat_with_tools(history, tools, model=model)

            if not tool_calls: # 没有 tool_calls 时，直接回答
                if not content:
                    raise LLMError("既没调工具也没说话")
                else:
                    return content
            else: # 有 tool_calls 则，调用工具
                history.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                })

                # 按照大模型返回的 tool_calls 逐个执行工具
                for tc in tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments) # '{"city": "北京"}' → {"city": "北京"}

                    impl = tool_impls.get(name)
                    if impl is None:
                        raise LLMError(f"没有名为 {name} 的工具")

                    result = impl(**args) # 等价于 get_weather(city="北京")

                    result_text = (
                        result if isinstance(result, str)
                        else json.dumps(result, ensure_ascii=False)
                    )

                    history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })

        # 最后一次尝试没有结果
        raise LLMError(f"工具调用超过 {max_iterations} 轮仍未结束")
    
                        
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