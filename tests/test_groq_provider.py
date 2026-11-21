"""What the Groq provider sends, and how it counts what Groq reports back.

The usage figures are the shape Groq actually returns for gpt-oss models: completion_tokens
already includes reasoning_tokens, and total_tokens is prompt plus completion.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas import UnifiedChatRequest
from app.providers.groq_provider import GroqProvider


USAGE = {
    "prompt_tokens": 74,
    "completion_tokens": 49,
    "total_tokens": 123,
    "completion_tokens_details": {"reasoning_tokens": 36},
}


class MockStreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class MockStreamingResponse:
    def __init__(self, lines):
        self.lines = lines
        self.raise_for_status = MagicMock()

    async def aiter_lines(self):
        for line in self.lines:
            yield line


def make_request(**fields) -> UnifiedChatRequest:
    return UnifiedChatRequest(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "Hello"}],
        **fields,
    )


def mock_client() -> MagicMock:
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


async def run_chat(request: UnifiedChatRequest):
    response = MagicMock()
    response.json.return_value = {
        "model": "openai/gpt-oss-20b",
        "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
        "usage": USAGE,
    }
    client = mock_client()
    client.post = AsyncMock(return_value=response)
    with patch("app.providers.groq_provider.httpx.AsyncClient", return_value=client):
        result = await GroqProvider(api_key="test-key").chat(request)
    return result, client.post.await_args.kwargs["json"]


async def run_stream(request: UnifiedChatRequest):
    stream = MockStreamingResponse(
        [
            "data: " + json.dumps({"choices": [{"delta": {"content": "Hi"}}]}),
            "data: " + json.dumps({"choices": [], "usage": USAGE}),
            "data: [DONE]",
        ]
    )
    client = mock_client()
    client.stream = MagicMock(return_value=MockStreamContext(stream))
    provider = GroqProvider(api_key="test-key")
    with patch("app.providers.groq_provider.httpx.AsyncClient", return_value=client):
        chunks = [chunk async for chunk in provider.chat_stream(request)]
    return provider, chunks, client.stream.call_args.kwargs["json"]


@pytest.mark.asyncio
async def test_chat_sends_the_callers_ceiling_and_temperature():
    """Budget is reserved at max_tokens, so the model must be told to stop there."""
    _result, sent = await run_chat(make_request(max_tokens=300, temperature=0.2))

    assert sent["max_completion_tokens"] == 300
    assert sent["temperature"] == 0.2


@pytest.mark.asyncio
async def test_streaming_sends_the_callers_ceiling():
    _provider, chunks, sent = await run_stream(make_request(max_tokens=60, stream=True))

    assert chunks == ["Hi"]
    assert sent["max_completion_tokens"] == 60


def test_unset_generation_options_are_left_to_groq():
    provider = GroqProvider(api_key="test-key")

    options = provider._build_generation_options(
        make_request(max_tokens=None, temperature=None)
    )

    assert options == {}


@pytest.mark.asyncio
async def test_chat_counts_reasoning_tokens_once():
    result, _sent = await run_chat(make_request())

    assert result.input_tokens == 74
    # 49, not 49 + 36: the reasoning tokens are already inside completion_tokens.
    assert result.output_tokens == 49
    assert result.input_tokens + result.output_tokens == USAGE["total_tokens"]


@pytest.mark.asyncio
async def test_streaming_counts_reasoning_tokens_once():
    provider, _chunks, _sent = await run_stream(make_request(stream=True))

    assert provider.last_stream_usage == {"input_tokens": 74, "output_tokens": 49}
