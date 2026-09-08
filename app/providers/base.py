"""Abstract provider interface for LLM API integrations."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from app.models.schemas import UnifiedChatRequest, UnifiedChatResponse


class LLMProvider(ABC):
    """Base class for provider integrations.

    All provider implementations must translate the UnifiedChatRequest into
    their provider-specific API format, call the provider, and translate the
    response back into UnifiedChatResponse.
    """

    provider_name: str

    @abstractmethod
    async def chat(self, request: UnifiedChatRequest) -> UnifiedChatResponse:
        """Send a chat request to the provider and return a unified response."""

    @abstractmethod
    async def chat_stream(
        self, request: UnifiedChatRequest
    ) -> AsyncGenerator[str, None]:
        """Send a chat request and yield incremental text chunks."""
