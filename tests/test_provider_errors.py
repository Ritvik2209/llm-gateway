"""Retry and fallback behaviour for classified provider failures.

The taxonomy exists so the request path can answer two questions without parsing an
error message: can a retry help, and is the provider at fault. These tests pin both.
"""

import httpx
import pytest

from app.models.schemas import ChatMessage, UnifiedChatRequest, UnifiedChatResponse
from app.providers.base import LLMProvider
from app.providers.errors import (
    ProviderAuthError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
    error_class_for_status,
    parse_retry_after,
)
from app.retry import call_with_retry


REQUEST = UnifiedChatRequest(
    model="mock-model",
    messages=[ChatMessage(role="user", content="hi")],
)


class CountingProvider(LLMProvider):
    """Fails with a given exception a fixed number of times, then succeeds."""

    provider_name = "counting"

    def __init__(self, exception: Exception, fail_times: int = 99) -> None:
        self.exception = exception
        self.fail_times = fail_times
        self.attempts = 0

    async def chat(self, request):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise self.exception
        return UnifiedChatResponse(
            id="ok",
            model=request.model,
            content="ok",
            input_tokens=1,
            output_tokens=1,
            provider=self.provider_name,
            finish_reason="stop",
        )

    async def chat_stream(self, request):
        yield "ok"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status_code, expected",
    [
        (429, ProviderRateLimited),
        (401, ProviderAuthError),
        (403, ProviderAuthError),
        (500, ProviderUnavailable),
        (503, ProviderUnavailable),
        (400, ProviderRequestRejected),
        (404, ProviderRequestRejected),
        (413, ProviderRequestRejected),
    ],
)
def test_http_status_maps_to_the_right_failure_class(status_code, expected):
    assert error_class_for_status(status_code) is expected


def test_only_transient_failures_are_retryable():
    assert ProviderTimeout("t").retryable is True
    assert ProviderUnavailable("u").retryable is True
    assert ProviderRateLimited("r").retryable is True
    assert ProviderAuthError("a").retryable is False
    assert ProviderRequestRejected("b").retryable is False


def test_a_rejected_request_is_not_the_provider_fault():
    """One malformed request pattern must not be able to open a healthy circuit."""
    assert ProviderRequestRejected("bad input").provider_at_fault is False
    assert ProviderAuthError("bad key").provider_at_fault is True
    assert ProviderUnavailable("down").provider_at_fault is True


def test_parse_retry_after_reads_seconds_and_tolerates_junk():
    assert parse_retry_after(httpx.Headers({"retry-after": "86.4"})) == 86.4
    assert parse_retry_after(httpx.Headers({})) is None
    # An HTTP-date Retry-After is not worth parsing; the backoff schedule takes over.
    assert parse_retry_after(httpx.Headers({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})) is None


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retryable_failures_are_retried_to_the_limit():
    provider = CountingProvider(ProviderUnavailable("down"))

    with pytest.raises(ProviderUnavailable):
        await call_with_retry(provider, REQUEST, max_retries=3, base_delay=0.0)

    assert provider.attempts == 4


@pytest.mark.asyncio
async def test_a_retryable_failure_that_clears_is_recovered():
    provider = CountingProvider(ProviderTimeout("slow"), fail_times=2)

    response = await call_with_retry(provider, REQUEST, max_retries=3, base_delay=0.0)

    assert response.content == "ok"
    assert provider.attempts == 3


@pytest.mark.asyncio
async def test_auth_failures_are_not_retried_at_all():
    """Previously this burned four attempts and 3.5s of backoff before failing."""
    provider = CountingProvider(ProviderAuthError("bad key"))

    with pytest.raises(ProviderAuthError):
        await call_with_retry(provider, REQUEST, max_retries=3, base_delay=0.0)

    assert provider.attempts == 1


@pytest.mark.asyncio
async def test_rejected_requests_are_not_retried_at_all():
    provider = CountingProvider(ProviderRequestRejected("malformed"))

    with pytest.raises(ProviderRequestRejected):
        await call_with_retry(provider, REQUEST, max_retries=3, base_delay=0.0)

    assert provider.attempts == 1


@pytest.mark.asyncio
async def test_a_rate_limit_inside_the_backoff_budget_is_retried():
    provider = CountingProvider(
        ProviderRateLimited("slow down", retry_after=0.01),
        fail_times=1,
    )

    response = await call_with_retry(provider, REQUEST, max_retries=3, base_delay=1.0)

    assert response.content == "ok"
    assert provider.attempts == 2


@pytest.mark.asyncio
async def test_a_rate_limit_longer_than_the_backoff_budget_stops_immediately():
    """A daily quota resets in hours, so retrying inside one request cannot succeed.

    Observed in practice: Groq answered 'requests per day (RPD): Limit 1000, Used 1000'
    with a retry hint far beyond any sane request budget. Failing over at once beats
    spending the backoff and failing anyway.
    """
    provider = CountingProvider(ProviderRateLimited("daily quota", retry_after=86.4))

    with pytest.raises(ProviderRateLimited):
        await call_with_retry(provider, REQUEST, max_retries=3, base_delay=0.5)

    assert provider.attempts == 1


@pytest.mark.asyncio
async def test_unclassified_failures_stay_retryable():
    """Providers that have not adopted the taxonomy keep their previous behaviour."""
    provider = CountingProvider(RuntimeError("something odd"))

    with pytest.raises(RuntimeError):
        await call_with_retry(provider, REQUEST, max_retries=3, base_delay=0.0)

    assert provider.attempts == 4
