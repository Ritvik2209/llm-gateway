import pytest

from app.models.schemas import UnifiedChatRequest
from app.providers.base import LLMProvider


def test_unified_chat_request_validates_with_messages():
    request = UnifiedChatRequest(
        model="test-model",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    )

    assert request.model == "test-model"
    assert len(request.messages) == 2
    assert request.messages[0].role == "system"
    assert request.temperature == 0.7
    assert request.max_tokens == 1024
    assert request.stream is False


def test_llm_provider_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        LLMProvider()
