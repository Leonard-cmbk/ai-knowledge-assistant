from unittest.mock import AsyncMock, MagicMock, patch

from openai import OpenAIError

import pytest

from llm_client import LLMClient, LLMError


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
            async for _ in client.stream_chat([{"role": "user", "content": "hi"}]):
                pass
