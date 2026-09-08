"""Mock provider implementation for tests and health checks."""

import asyncio
from typing import AsyncGenerator
from uuid import uuid4

from app.models.schemas import UnifiedChatRequest, UnifiedChatResponse
from app.providers.base import LLMProvider


class MockProvider(LLMProvider):
    """Provider implementation with configurable latency and failure behavior."""

    provider_name = "mock"

    def __init__(
        self,
        should_fail: bool = False,
        latency_seconds: float = 0.001,
    ) -> None:
        self.should_fail = should_fail
        self.latency_seconds = latency_seconds

    async def chat(self, request: UnifiedChatRequest) -> UnifiedChatResponse:
        if self.should_fail:
            raise RuntimeError("Mock provider simulated failure")

        await asyncio.sleep(self.latency_seconds)
        return UnifiedChatResponse(
            id=str(uuid4()),
            model=request.model,
            content="mock response",
            input_tokens=10,
            output_tokens=5,
            provider=self.provider_name,
            finish_reason="stop",
        )

    async def chat_stream(
        self, request: UnifiedChatRequest
    ) -> AsyncGenerator[str, None]:
        if self.should_fail:
            raise RuntimeError("Mock provider simulated failure")

        await asyncio.sleep(self.latency_seconds)
        yield "mock "
        yield "response"
