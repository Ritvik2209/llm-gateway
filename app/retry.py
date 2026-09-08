"""Retry helpers for provider calls."""

import asyncio

from app.models.schemas import UnifiedChatRequest, UnifiedChatResponse
from app.providers.base import LLMProvider


async def call_with_retry(
    provider: LLMProvider,
    request: UnifiedChatRequest,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> UnifiedChatResponse:
    """Call provider.chat with exponential backoff before surfacing failure."""
    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await provider.chat(request)
        except Exception as exc:
            last_exception = exc
            if attempt == max_retries:
                break
            await asyncio.sleep(base_delay * (2**attempt))

    assert last_exception is not None
    raise last_exception
