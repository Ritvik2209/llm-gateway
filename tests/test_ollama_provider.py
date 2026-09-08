from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from app.models.schemas import UnifiedChatRequest
from app.providers.ollama_provider import OllamaProvider


@pytest.mark.asyncio
async def test_ollama_chat_maps_successful_response():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model": "llama3.2",
        "message": {"role": "assistant", "content": "Hello from Ollama"},
        "done_reason": "stop",
        "prompt_eval_count": 12,
        "eval_count": 7,
    }
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)

    provider = OllamaProvider()
    request = UnifiedChatRequest(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say hello"},
        ],
    )

    with patch(
        "app.providers.ollama_provider.httpx.AsyncClient",
        return_value=mock_client,
    ):
        response = await provider.chat(request)

    assert response.id
    assert response.model == "llama3.2"
    assert response.content == "Hello from Ollama"
    assert response.input_tokens == 12
    assert response.output_tokens == 7
    assert response.provider == "ollama"
    assert response.finish_reason == "stop"

    mock_response.raise_for_status.assert_called_once()
    mock_client.post.assert_awaited_once_with(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Say hello"},
            ],
            "stream": False,
        },
    )


@pytest.mark.asyncio
async def test_ollama_chat_connection_failure_raises_exception():
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    provider = OllamaProvider()
    request = UnifiedChatRequest(
        model="llama3.2",
        messages=[{"role": "user", "content": "Hello"}],
    )

    with patch(
        "app.providers.ollama_provider.httpx.AsyncClient",
        return_value=mock_client,
    ):
        with pytest.raises(RuntimeError, match="Ollama chat request failed"):
            await provider.chat(request)
