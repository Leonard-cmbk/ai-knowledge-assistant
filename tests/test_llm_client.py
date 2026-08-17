from unittest.mock import AsyncMock, MagicMock, patch

from openai import OpenAIError

import pytest, json

from llm_client import LLMClient, LLMError, _parse_json


def make_tool_call(call_id: str, name: str, args: dict):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args, ensure_ascii=False)
    return tc
    

def make_fake_weather(seen_cities):
    def fake_weather(city: str) -> str:
        seen_cities.append(city)
        table = {"北京": "晴,25°C", "上海": "多云,28°C"}
        return table.get(city, f"暂无 {city} 的天气数据")
    return fake_weather


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名,如 北京"}},
            "required": ["city"],
        },
    },
}


@pytest.mark.asyncio
async def test_chat_returns_content():
    client = LLMClient()

    fake_resp = AsyncMock()
    fake_resp.choices[0].message.content = "hello there"

    target = client._client.chat.completions
    with patch.object(target, "create", new=AsyncMock(return_value=fake_resp)):
        result = await client.chat([{"role": "user", "content": "hi"}])

    assert result == "hello there"


@pytest.mark.asyncio
async def test_chat_empty_content_raises():
    client = LLMClient()

    fake_resp = AsyncMock()
    fake_resp.choices[0].message.content = None

    target = client._client.chat.completions
    with patch.object(target, "create", new=AsyncMock(return_value=fake_resp)):
        with pytest.raises(LLMError):
            await client.chat([{"role": "user", "content": "hi"}])


def test_parse_json_normal_return():
    # 正常 JSON 字符串 → 返回 dict
    assert _parse_json('{"name": "张三"}') == {"name": "张三"}


def test_parse_json_strips_fence():
    # 模型用 Markdown 围栏包住 JSON
    content = '```json\n{"name": "张三"}\n```'
    assert _parse_json(content) == {"name": "张三"}


def test_parse_json_with_surrounding_text():
    # 模型在 JSON 前后夹带废话
    content = '好的，结果如下：{"name": "张三"}'
    assert _parse_json(content) == {"name": "张三"}


def test_parse_json_invalid_raises():
    # 完全不是 JSON → 抛业务异常
    with pytest.raises(LLMError):
        _parse_json("不能识别为 JSON")


@pytest.mark.asyncio
async def test_chat_json_correct_content():
    client = LLMClient()

    fake_resp = AsyncMock()
    fake_resp.choices[0].message.content = '{"name": "张三"}'

    target = client._client.chat.completions
    with patch.object(target, "create", new=AsyncMock(return_value=fake_resp)) as create:
        resp = await client.chat_json([{"role": "user", "content": "hi"}])

    assert resp == {"name": "张三"}
    # 断言:请求是否带了 JSON mode 开关
    assert create.await_args.kwargs['response_format'] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_chat_json_empty_content_raise():
    client = LLMClient()

    fake_resp = AsyncMock()
    fake_resp.choices[0].message.content = None

    target = client._client.chat.completions
    with patch.object(target, "create", new=AsyncMock(return_value=fake_resp)):
        with pytest.raises(LLMError):
            await client.chat_json([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_chat_with_tools_response_direct():
    client = LLMClient()

    fake_resp = AsyncMock()
    fake_resp.choices[0].message.content = '北京今天晴,25°C'
    fake_resp.choices[0].message.tool_calls = None

    target = client._client.chat.completions
    with patch.object(target, "create", new=AsyncMock(return_value=fake_resp)):
        msg = await client.chat_with_tools(
            [{"role": "user", "content": "hi"}], 
            [{"type": "function", "function": {"name": "get_weather"}}]
        )

    assert msg == ("北京今天晴,25°C", None)


@pytest.mark.asyncio
async def test_chat_with_tools_response_use_tool():
    client = LLMClient()

    fake_resp = AsyncMock()
    tool_call = make_tool_call("call_1", "get_weather", {"city": "北京"})

    fake_resp.choices[0].message.content = None
    fake_resp.choices[0].message.tool_calls = [tool_call]

    target = client._client.chat.completions
    with patch.object(target, "create", new=AsyncMock(return_value=fake_resp)) as create:
        msg = await client.chat_with_tools(
            [{"role": "user", "content": "hi"}],
            [{"type": "function", "function": {"name": "get_weather"}}]
        )

    assert msg[0] is None
    assert msg[1][0].function.name == "get_weather"
    # 判断参数 tool 是否合规
    assert create.await_args.kwargs['tools'] == [{
        "type": "function", 
        "function": {
            "name": "get_weather"
        }
    }]


@pytest.mark.asyncio
async def test_run_tool_loop():
    """测试正常流程:工具被调用、历史正确组装、返回最终回答。"""
    client = LLMClient()

    tool_call = make_tool_call("call_1", "get_weather", {"city": "北京"})
    
    seen_cities = []

    fake_chat = AsyncMock(side_effect=[
        (None, [tool_call]),        # 第 1 轮:模型提议调工具
        ("北京今天晴,25°C", None),   # 第 2 轮:模型给出最终回答
    ])

    with patch.object(client, "chat_with_tools", fake_chat):
        result = await client.run_tool_loop(
            [{"role": "user", "content": "北京天气?"}],
            tools=[WEATHER_TOOL],
            tool_impls={"get_weather": make_fake_weather(seen_cities)}
        )

    second_messages = fake_chat.await_args_list[1].args[0]   # 第 2 轮传入的 history
    tool_msgs = [m for m in second_messages if m["role"] == "tool"]

    assert result == "北京今天晴,25°C"
    assert seen_cities == ["北京"]      # 工具确实被调用了
    assert tool_msgs == [{"role": "tool", "tool_call_id": "call_1", "content": "晴,25°C"}]


@pytest.mark.asyncio
async def test_run_tool_loop_multi_call():
    """
    测试 「一轮多 tool_calls」
    """
    client = LLMClient()
    
    tool_call_bj = make_tool_call("call_1", "get_weather", {"city": "北京"})
    tool_call_sh = make_tool_call("call_2", "get_weather", {"city": "上海"})

    seen_cities = []

    fake_chat = AsyncMock(side_effect=[
        (None, [tool_call_bj, tool_call_sh]),        # 第 1 轮:模型提议调工具
        ("北京和上海天气都不错", None),   # 第 2 轮:模型给出最终回答
    ])

    with patch.object(client, "chat_with_tools", fake_chat):
        result = await client.run_tool_loop(
            [{"role": "user", "content": "北京和上海今天的天气怎么样？"}],
            tools=[WEATHER_TOOL],
            tool_impls={"get_weather": make_fake_weather(seen_cities)}
        )

    second_messages = fake_chat.await_args_list[1].args[0]   # 第 2 轮传入的 history
    tool_msgs = [m for m in second_messages if m["role"] == "tool"]

    assert result == "北京和上海天气都不错"
    assert seen_cities == ["北京", "上海"]          # 两个都被执行,且顺序保持
    assert tool_msgs == [
        {"role": "tool", "tool_call_id": "call_1", "content": "晴,25°C"},
        {"role": "tool", "tool_call_id": "call_2", "content": "多云,28°C"},
    ]                                              # 两条 tool 消息,id 一一配对


@pytest.mark.asyncio
async def test_run_tool_loop_over():
    """
    测试超过最大循环次数
    """
    client = LLMClient()

    tool_call = make_tool_call("call_1", "get_weather", {"city": "北京"})
    seen_cities = []

    with patch.object(client, "chat_with_tools", AsyncMock(return_value=(None, [tool_call]))):
        with pytest.raises(LLMError, match="超过 5 轮"):
            await client.run_tool_loop(
                [{"role": "user", "content": "北京天气?"}],
                tools=[WEATHER_TOOL],
                tool_impls={"get_weather": make_fake_weather(seen_cities)}
            )


# 流式单测(stream_chat)
async def make_fake_stream(contents):
    for content in contents:
        chunk = MagicMock()
        chunk.choices[0].delta.content = content
        yield chunk


@pytest.mark.asyncio
async def test_stream_chat_yields_only_non_empty_deltas():
    client = LLMClient()

    fake_stream = make_fake_stream(["你", "好", None, "！", ""])
    target = client._client.chat.completions
    with patch.object(target, "create", new=AsyncMock(return_value=fake_stream)):
        collected = [
            text
            async for text in client.stream_chat([{"role": "user", "content": "hi"}])
        ]

    assert collected == ["你", "好", "！"]


@pytest.mark.asyncio
async def test_stream_chat_wraps_sdk_error():
    client = LLMClient()

    target = client._client.chat.completions
    with patch.object(target, "create", new=AsyncMock(side_effect=OpenAIError("boom"))):
        with pytest.raises(LLMError):
            async for _ in client.stream_chat([{"role": "user", "content": "hi"}]):#messages中内容不重要
                pass
