"""Retry helpers for provider calls."""

import asyncio

from app.models.schemas import UnifiedChatRequest, UnifiedChatResponse
from app.providers.base import LLMProvider
from app.providers.errors import ProviderError


async def call_with_retry(
    provider: LLMProvider,
    request: UnifiedChatRequest,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> UnifiedChatResponse:
    """Call ``provider.chat`` with exponential backoff, retrying only where it can help.

    Retrying is not free: the default schedule spends 0.5 + 1 + 2 seconds of user-visible
    latency before giving up, and every attempt adds load to a provider that may already
    be struggling. So a retry is only attempted when the failure could plausibly clear:

    * A non-retryable failure — rejected credentials, a malformed request — is raised
      immediately. No amount of backoff will change the outcome, and the delay only
      postpones the caller's error or the fallback behind it.
    * A rate limit that the provider says will outlast our backoff budget is also raised
      immediately. Retrying into a daily quota that resets in 24 hours cannot succeed,
      and failing over to another provider is strictly better than waiting.
    * Anything unclassified is treated as retryable, preserving the previous behaviour for
      providers that have not adopted the error taxonomy.
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await provider.chat(request)
        except Exception as exc:
            last_exception = exc

            if isinstance(exc, ProviderError) and not exc.retryable:
                raise

            if attempt == max_retries:
                break

            delay = base_delay * (2**attempt)

            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None and retry_after > delay:
                # The provider has told us it will not be ready inside our budget, so
                # sleeping here would burn latency and still fail.
                raise

            await asyncio.sleep(delay)

    assert last_exception is not None
    raise last_exception
