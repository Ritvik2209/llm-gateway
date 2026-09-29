import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.circuit_breaker import CircuitBreaker
from app.health import HealthMonitor
from app.models.catalog import ModelCatalog
from app.models.schemas import UnifiedChatRequest
from app.providers.base import LLMProvider
from app.providers.errors import (
    ProviderAuthError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from app.providers.groq_provider import GroqProvider
from app.providers.ollama_provider import OllamaProvider
from tests.fakes import FakeRedis


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


# ---------------------------------------------------------------------------
# Streaming failures and the resilience layer
#
# The circuit breaker and health monitor were fed only by `call_chat_with_fallback`,
# which serves the non-streaming path. Streaming bypassed both, so a provider that
# failed only under streaming load stayed "healthy" with a closed circuit and kept
# receiving traffic. These tests pin the streaming path to the same contract the
# non-streaming path already had.
# ---------------------------------------------------------------------------


class StubStreamProvider(LLMProvider):
    """A provider whose stream fails at a controllable point.

    `fail_after=None` streams cleanly; `0` fails before the first chunk, so the client
    receives nothing; `n > 0` fails after n chunks, which is the case where the client
    already holds a 200 and a partial body.
    """

    provider_name = "mock"

    def __init__(
        self,
        fail_after: int | None = None,
        error: Exception | None = None,
        chunks: tuple[str, ...] = ("one ", "two ", "three"),
    ) -> None:
        self.fail_after = fail_after
        self.error = error if error is not None else ProviderUnavailable("stream died")
        self.chunks = chunks
        self.last_stream_usage = {"input_tokens": 10, "output_tokens": 5}

    async def chat(self, request):
        raise AssertionError("these tests must exercise the streaming path only")

    async def chat_stream(self, request):
        if self.fail_after == 0:
            raise self.error
        for index, chunk in enumerate(self.chunks, start=1):
            yield chunk
            if self.fail_after is not None and index >= self.fail_after:
                raise self.error


STREAM_TEAMS_CONFIG = {
    "streaming-key": {
        "team_id": "streaming-resilience",
        "allowed_models": ["mock-model"],
        "allowed_providers": ["mock"],
        "provider_priority": ["mock"],
        "requests_per_minute": 1000,
        "monthly_budget_usd": 10.0,
    },
}

STREAM_CATALOG = ModelCatalog(providers={"mock": {"mock-model"}})

STREAM_REQUEST = {
    "model": "mock-model",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": True,
    "max_tokens": 10,
}


def create_streaming_client(monkeypatch, provider, raise_server_exceptions=True):
    """Build a TestClient wired to `provider`, with a fresh breaker and health monitor.

    Not used as a context manager, matching the rest of the suite: that keeps the lifespan
    handler from starting the background health task or contacting a real provider.
    """
    circuit_breaker = CircuitBreaker()
    health_monitor = HealthMonitor()

    monkeypatch.setattr(main_module.app.state, "teams_config", STREAM_TEAMS_CONFIG)
    monkeypatch.setattr(main_module.app.state, "redis_client", FakeRedis(decode_responses=True))
    monkeypatch.setattr(main_module.app.state, "circuit_breaker", circuit_breaker)
    monkeypatch.setattr(main_module.app.state, "model_catalog", STREAM_CATALOG)
    monkeypatch.setattr(main_module, "health_monitor", health_monitor)
    monkeypatch.setattr(main_module, "mock_provider", provider)

    client = TestClient(main_module.app, raise_server_exceptions=raise_server_exceptions)
    return client, circuit_breaker, health_monitor


def post_stream(client):
    return client.post(
        "/v1/chat",
        headers={"Authorization": "Bearer streaming-key"},
        json=STREAM_REQUEST,
    )


def test_streaming_failure_before_first_chunk_feeds_breaker_and_health(monkeypatch):
    client, circuit_breaker, health_monitor = create_streaming_client(
        monkeypatch, StubStreamProvider(fail_after=0)
    )

    with pytest.raises(ProviderUnavailable):
        post_stream(client)

    assert circuit_breaker.get_state("mock") == "closed"
    assert circuit_breaker._get_provider_state("mock")["failure_count"] == 1
    assert health_monitor.provider_health[("mock", "mock-model")].consecutive_failures == 1
    assert health_monitor.get_status("mock") == "degraded"


def test_streaming_failure_mid_stream_still_feeds_breaker_and_health(monkeypatch, caplog):
    """A stream that dies after partial content is still a failed request.

    The client cannot be told so: the 200 and the first chunks are already on the wire,
    and HTTP has no way to retract a status line once the body has begun. The gateway's
    own record is therefore the only trace, which is precisely why the breaker and health
    monitor have to be fed here.

    The truncated body is deliberately not asserted. `TestClient` discards it when the
    ASGI app raises mid-body, so what a real caller would hold - the bytes already sent -
    is not observable through this transport. The logged chunk count is, and it is what
    distinguishes this case from a failure before the first chunk.
    """
    client, circuit_breaker, health_monitor = create_streaming_client(
        monkeypatch, StubStreamProvider(fail_after=2), raise_server_exceptions=False
    )

    with caplog.at_level(logging.WARNING, logger="app.main"):
        response = post_stream(client)

    assert response.status_code == 200
    assert "after 2 chunk(s)" in caplog.text
    assert circuit_breaker._get_provider_state("mock")["failure_count"] == 1
    assert health_monitor.get_status("mock") == "degraded"


def test_repeated_streaming_failures_open_the_circuit(monkeypatch):
    """The headline of this change: streaming traffic can now open a breaker.

    Four consecutive failures is the configured threshold, so the fourth streaming
    failure removes the provider from rotation exactly as a non-streaming one would.
    """
    client, circuit_breaker, health_monitor = create_streaming_client(
        monkeypatch, StubStreamProvider(fail_after=0)
    )

    for _ in range(CircuitBreaker().failure_threshold):
        with pytest.raises(ProviderUnavailable):
            post_stream(client)

    assert circuit_breaker.get_state("mock") == "open"
    assert circuit_breaker.can_attempt("mock") is False
    assert health_monitor.get_status("mock") == "down"


def test_streaming_success_records_success_and_time_to_first_chunk(monkeypatch):
    client, circuit_breaker, health_monitor = create_streaming_client(
        monkeypatch, StubStreamProvider()
    )

    response = post_stream(client)

    assert response.status_code == 200
    assert response.text == "one two three"
    assert circuit_breaker.get_state("mock") == "closed"
    assert circuit_breaker._get_provider_state("mock")["failure_count"] == 0
    assert health_monitor.get_status("mock") == "healthy"

    # One latency sample, and it is time to first chunk rather than total stream
    # duration: the stub sleeps nowhere, so both are small, but only the first is a
    # measure of provider responsiveness rather than of output length.
    assert len(health_monitor.provider_health[("mock", "mock-model")].recent_latencies) == 1


def test_streaming_failure_not_the_providers_fault_spares_the_circuit(monkeypatch):
    """A request the provider rejected on its merits must not count against it.

    `ProviderRequestRejected` carries `provider_at_fault = False`, so one malformed
    request pattern cannot open a working provider's circuit and remove it from rotation
    for every other team. The streaming path now reads that flag; before this change it
    had no typed error to read it from.
    """
    client, circuit_breaker, health_monitor = create_streaming_client(
        monkeypatch,
        StubStreamProvider(
            fail_after=0,
            error=ProviderRequestRejected("context length exceeded"),
        ),
    )

    with pytest.raises(ProviderRequestRejected):
        post_stream(client)

    assert circuit_breaker._get_provider_state("mock")["failure_count"] == 0
    assert circuit_breaker.get_state("mock") == "closed"
    assert ("mock", "mock-model") not in health_monitor.provider_health


# ---------------------------------------------------------------------------
# Streaming error taxonomy
#
# Both streaming implementations raised bare RuntimeError, so every streaming failure
# looked identical: the request path could not tell a rejected credential from a
# momentary outage, and could not read `provider_at_fault` to decide whether the
# failure belonged against the provider's health. They now map status onto the same
# taxonomy their non-streaming counterparts already used.
# ---------------------------------------------------------------------------


class UnreadStreamResponse:
    """A streamed response whose `raise_for_status` fails and whose body is unread.

    `.text` raises here exactly as httpx does before a streamed response is consumed, so
    a failure handler that reaches for the body fails this test rather than production.
    That is not hypothetical: the non-streaming handlers do include `response.text` in
    their message, and copying them verbatim into the streaming path would raise
    ResponseNotRead from inside the error handler.
    """

    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}

    @property
    def text(self):
        raise httpx.ResponseNotRead()

    def raise_for_status(self):
        raise httpx.HTTPStatusError(
            f"HTTP {self.status_code}",
            request=httpx.Request("POST", "http://provider.test/stream"),
            response=self,
        )

    async def aiter_lines(self):
        yield ""


def stream_client_for(response):
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock(return_value=MockStreamContext(response))
    return mock_client


STREAM_IMPLEMENTATIONS = [
    pytest.param(
        "app.providers.groq_provider.httpx.AsyncClient",
        GroqProvider(api_key="test-key", base_url="http://groq.test"),
        "openai/gpt-oss-20b",
        id="groq",
    ),
    pytest.param(
        "app.providers.ollama_provider.httpx.AsyncClient",
        OllamaProvider(base_url="http://ollama.test"),
        "llama3.2",
        id="ollama",
    ),
]


@pytest.mark.parametrize("client_path, provider, model", STREAM_IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "status_code, expected_error",
    [
        (429, ProviderRateLimited),
        (401, ProviderAuthError),
        (403, ProviderAuthError),
        (400, ProviderRequestRejected),
        (500, ProviderUnavailable),
        (503, ProviderUnavailable),
    ],
)
@pytest.mark.asyncio
async def test_chat_stream_maps_status_onto_the_error_taxonomy(
    client_path, provider, model, status_code, expected_error
):
    request = UnifiedChatRequest(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    mock_client = stream_client_for(UnreadStreamResponse(status_code))

    with patch(client_path, return_value=mock_client):
        with pytest.raises(expected_error) as raised:
            async for _ in provider.chat_stream(request):
                pass

    assert raised.value.status_code == status_code


@pytest.mark.parametrize("client_path, provider, model", STREAM_IMPLEMENTATIONS)
@pytest.mark.asyncio
async def test_chat_stream_rate_limit_carries_retry_after(client_path, provider, model):
    """The provider's own estimate, not the retry loop's guess.

    A per-day quota will not clear inside a request's backoff budget while a per-minute
    one might, and only `retry_after` distinguishes them.
    """
    request = UnifiedChatRequest(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    mock_client = stream_client_for(
        UnreadStreamResponse(429, headers={"retry-after": "42"})
    )

    with patch(client_path, return_value=mock_client):
        with pytest.raises(ProviderRateLimited) as raised:
            async for _ in provider.chat_stream(request):
                pass

    assert raised.value.retry_after == 42.0
    assert raised.value.retryable is True


@pytest.mark.parametrize("client_path, provider, model", STREAM_IMPLEMENTATIONS)
@pytest.mark.asyncio
async def test_chat_stream_timeout_is_typed_as_a_timeout(client_path, provider, model):
    """httpx.TimeoutException subclasses RequestError, so handler order is load-bearing.

    Catching RequestError first would classify every timeout as ProviderUnavailable and
    lose the distinction.
    """
    request = UnifiedChatRequest(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock(side_effect=httpx.ReadTimeout("too slow"))

    with patch(client_path, return_value=mock_client):
        with pytest.raises(ProviderTimeout):
            async for _ in provider.chat_stream(request):
                pass


def test_a_disclaimer_is_emitted_as_a_final_chunk(monkeypatch):
    """A stream cannot be rewritten, so the disclaimer is an extra chunk at the end.

    The non-streaming path splices it into the body; here it can only be appended after
    the model's own output, which is the same policy applied the only way the transport
    allows.
    """
    client, _breaker, _health = create_streaming_client(monkeypatch, StubStreamProvider())
    monkeypatch.setitem(
        STREAM_TEAMS_CONFIG["streaming-key"],
        "response_disclaimer",
        "AI generated.",
    )

    response = post_stream(client)

    assert response.status_code == 200
    assert response.text == "one two three\n\nAI generated."


def test_a_failed_stream_carries_no_disclaimer(monkeypatch):
    """Nothing finished, so there is no complete answer to qualify.

    The disclaimer is emitted after the success path's accounting, which a failed stream
    never reaches — so this falls out of where it sits rather than needing its own guard.
    """
    client, _breaker, _health = create_streaming_client(
        monkeypatch, StubStreamProvider(fail_after=2), raise_server_exceptions=False
    )
    monkeypatch.setitem(
        STREAM_TEAMS_CONFIG["streaming-key"],
        "response_disclaimer",
        "AI generated.",
    )

    response = post_stream(client)

    assert "AI generated." not in response.text
