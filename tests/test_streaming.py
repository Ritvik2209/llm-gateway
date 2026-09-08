import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas import UnifiedChatRequest
from app.providers.ollama_provider import OllamaProvider


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


@pytest.mark.asyncio
async def test_ollama_chat_stream_yields_chunks_and_accumulates_text():
    stream_response = MockStreamingResponse(
        [
            json.dumps({"message": {"content": "Hel"}, "done": False}),
            json.dumps({"message": {"content": "lo"}, "done": False}),
            json.dumps({"message": {"content": "!"}, "done": False}),
            json.dumps({"done": True}),
        ]
    )
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock(return_value=MockStreamContext(stream_response))

    provider = OllamaProvider(base_url="http://ollama.test")
    request = UnifiedChatRequest(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say hello"},
        ],
        stream=True,
    )

    chunks = []
    with patch(
        "app.providers.ollama_provider.httpx.AsyncClient",
        return_value=mock_client,
    ):
        async for chunk in provider.chat_stream(request):
            chunks.append(chunk)

    assert chunks == ["Hel", "lo", "!"]
    assert "".join(chunks) == "Hello!"
    stream_response.raise_for_status.assert_called_once()
    mock_client.stream.assert_called_once_with(
        "POST",
        "http://ollama.test/api/chat",
        json={
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Say hello"},
            ],
            "stream": True,
        },
    )
