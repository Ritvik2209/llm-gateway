"""End-to-end integration tests driving the full FastAPI request path.

These differ from the unit tests in this directory: rather than calling helpers such as
`select_provider` or `check_budget` directly, every test here issues a real HTTP request through
`TestClient` and asserts on what the whole stack does - auth, model authorization, rate limiting,
budget checks, provider routing/fallback, circuit breaking, spend accounting and metrics.

Conventions follow the existing suite (see `test_enrichment.py`): plain test functions with
`monkeypatch`, in-memory team configs rather than the real `config/teams.yaml`, and no fixtures or
`conftest.py`, since the project has neither.

Every test uses the mock provider only, so the suite is deterministic, offline and free.

NOTE on the Redis fake: `fakeredis` is used for storage, so budget accounting exercises genuine
Redis `INCRBYFLOAT`/`GET` semantics (values round-trip as strings). However, the rate limiter runs a
Lua script via `EVAL` (`app/rate_limiter.py:12-24`), and fakeredis can only execute Lua when `lupa`
is installed - it is not installed and is not in `requirements.txt`, so `EVAL` raises
`unknown command 'eval'`. `_FakeRedis` therefore subclasses fakeredis and emulates only that one
script in Python, mirroring the hand-rolled fake already used in `test_rate_limiter.py`.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app import budget as budget_module
from app import main as main_module
from app.circuit_breaker import CircuitBreaker
from app.config import MODEL_PRICING
from app.models.schemas import UnifiedChatResponse
from app.retry import call_with_retry as real_call_with_retry


CHAT_REQUEST = {
    "model": "mock-model",
    "messages": [{"role": "user", "content": "hello"}],
}

# Each scenario gets its own api key and team_id so Redis budget/rate-limit keys never collide,
# letting these tests run in any order, repeatedly, alongside the existing unit tests.
TEAMS_CONFIG = {
    "integration-lifecycle-key": {
        "team_id": "integration-lifecycle",
        "allowed_models": ["mock-model"],
        "allowed_providers": ["mock"],
        "provider_priority": ["mock"],
        "requests_per_minute": 1000,
        "monthly_budget_usd": 10.0,
    },
    "integration-fallback-key": {
        "team_id": "integration-fallback",
        "allowed_models": ["mock-model"],
        # NOTE: `allowed_providers` is listed here for realism but is never consulted when routing.
        # `get_provider_candidates` (app/main.py:179-181) filters `provider_priority` against the
        # globally registered providers dict only. See the "allowed_providers is never enforced in
        # routing" known limitation in LOAD_TEST_RESULTS.md. Routing below is driven purely by
        # `provider_priority`.
        "allowed_providers": ["mock", "ollama"],
        "provider_priority": ["mock", "ollama"],
        "requests_per_minute": 1000,
        "monthly_budget_usd": 10.0,
    },
    "integration-circuit-key": {
        "team_id": "integration-circuit",
        "allowed_models": ["mock-model"],
        "allowed_providers": ["mock"],
        "provider_priority": ["mock"],
        "requests_per_minute": 1000,
        "monthly_budget_usd": 10.0,
    },
    "integration-budget-key": {
        "team_id": "integration-budget",
        "allowed_models": ["mock-model"],
        "allowed_providers": ["mock"],
        "provider_priority": ["mock"],
        "requests_per_minute": 1000,
        "monthly_budget_usd": 0.001,
    },
    "integration-ratelimit-key": {
        "team_id": "integration-ratelimit",
        "allowed_models": ["mock-model"],
        "allowed_providers": ["mock"],
        "provider_priority": ["mock"],
        "requests_per_minute": 2,
        "monthly_budget_usd": 10.0,
    },
}

# Mock responses report 10 input and 5 output tokens (app/providers/mock_provider.py:29-35).
MOCK_INPUT_TOKENS = 10
MOCK_OUTPUT_TOKENS = 5

# NOTE: the shipped pricing table prices "mock-model" at 0.0/0.0 (app/config.py:19-22), so a live
# mock request costs exactly $0 and can never move a budget. Tests that need spend to accumulate
# monkeypatch a priced entry, following the pattern already used in test_budget.py.
PRICED_MOCK_MODEL = {
    **MODEL_PRICING,
    "mock-model": {"input_price_per_1k": 0.02, "output_price_per_1k": 0.04},
}
# 10/1000 * 0.02 + 5/1000 * 0.04
PRICED_COST_PER_REQUEST = 0.0004


class _FakeRedis(fakeredis.aioredis.FakeRedis):
    """fakeredis with a Python emulation of the rate limiter's Lua sliding window."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rate_limit_windows: dict[str, list[tuple[float, str]]] = {}

    async def eval(self, script, numkeys, *args):
        key, now, window_start, limit, member, _window_seconds = args
        now = float(now)
        window_start = float(window_start)
        limit = int(limit)

        entries = [
            (score, existing_member)
            for score, existing_member in self._rate_limit_windows.get(key, [])
            if score > window_start
        ]
        self._rate_limit_windows[key] = entries

        if len(entries) < limit:
            entries.append((now, member))
            return [1, limit - len(entries)]

        return [0, 0]


async def _immediate_retry(provider, request, max_retries=3, base_delay=0.5):
    """`call_with_retry` with the backoff removed.

    The real default backoff is 0.5s doubling per attempt, so one exhausted provider sleeps
    0.5 + 1 + 2 = 3.5 seconds (app/retry.py:10-22). Retry semantics are preserved - still four
    attempts, still exactly one `record_failure` per request - only the wall-clock delay is dropped.
    """
    return await real_call_with_retry(
        provider,
        request,
        max_retries=max_retries,
        base_delay=0.0,
    )


def create_test_client(monkeypatch):
    """Build a TestClient with an isolated teams config, Redis and circuit breaker.

    `TestClient` is deliberately not used as a context manager, matching `test_enrichment.py`: that
    keeps the lifespan handler from running, so no background health-check task starts and no real
    provider is ever contacted.
    """
    redis_client = _FakeRedis(decode_responses=True)
    circuit_breaker = CircuitBreaker()

    monkeypatch.setattr(main_module.app.state, "teams_config", TEAMS_CONFIG)
    monkeypatch.setattr(main_module.app.state, "redis_client", redis_client)
    monkeypatch.setattr(main_module.app.state, "circuit_breaker", circuit_breaker)
    monkeypatch.setattr(main_module, "call_with_retry", _immediate_retry)

    return TestClient(main_module.app), redis_client, circuit_breaker


def read_spend(redis_client, team_id: str) -> float:
    """Read a team's recorded month-to-date spend straight out of Redis."""
    return asyncio.run(budget_module.get_current_spend(team_id, redis_client))


def read_fallback_counter(team_id: str, from_provider: str, to_provider: str) -> float:
    """Read gateway_fallback_triggered_total from the prometheus_client registry in-process."""
    value = REGISTRY.get_sample_value(
        "gateway_fallback_triggered_total",
        {
            "team_id": team_id,
            "from_provider": from_provider,
            "to_provider": to_provider,
        },
    )
    # The sample does not exist at all until the counter is first incremented.
    return 0.0 if value is None else value


class StubOllamaProvider:
    """Stand-in for the real Ollama provider so fallback never touches the network."""

    provider_name = "ollama"

    async def chat(self, request):
        return UnifiedChatResponse(
            id="stub-ollama-response",
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
# 1. Full request lifecycle
# ---------------------------------------------------------------------------


def test_full_request_lifecycle_returns_response_and_records_spend(monkeypatch):
    """A valid request returns a well-formed response and charges the team's budget."""
    monkeypatch.setattr(budget_module, "MODEL_PRICING", PRICED_MOCK_MODEL)
    client, redis_client, _circuit_breaker = create_test_client(monkeypatch)

    response = client.post(
        "/v1/chat",
        headers={"Authorization": "Bearer integration-lifecycle-key"},
        json=CHAT_REQUEST,
    )

    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        "id",
        "model",
        "content",
        "input_tokens",
        "output_tokens",
        "provider",
        "finish_reason",
    }
    assert body["id"]
    assert body["model"] == "mock-model"
    assert body["content"] == "mock response"
    assert body["input_tokens"] == MOCK_INPUT_TOKENS
    assert body["output_tokens"] == MOCK_OUTPUT_TOKENS
    assert body["finish_reason"] == "stop"

    # The reported provider is the one that actually served the call, not merely the team's first
    # preference - scenario 2 exercises the case where those two differ.
    assert body["provider"] == "mock"

    spend = read_spend(redis_client, "integration-lifecycle")
    assert spend == pytest.approx(PRICED_COST_PER_REQUEST)


def test_full_request_lifecycle_costs_nothing_with_real_mock_pricing(monkeypatch):
    """Documents current behavior: a mock request is free under the shipped pricing table.

    NOTE: this is not an obviously-correct design. `mock-model` is priced at 0.0/0.0 in
    `app/config.py:19-22`, so real mock traffic accrues no spend and can never trip a budget cap.
    That is reasonable for a test/demo provider, but it does mean budget enforcement is unreachable
    for any team routed to `mock` - which is precisely why the budget scenario below has to
    monkeypatch a price in. Asserted here so the behavior is recorded rather than assumed.
    """
    client, redis_client, _circuit_breaker = create_test_client(monkeypatch)

    response = client.post(
        "/v1/chat",
        headers={"Authorization": "Bearer integration-lifecycle-key"},
        json=CHAT_REQUEST,
    )

    assert response.status_code == 200
    assert read_spend(redis_client, "integration-lifecycle") == 0.0


# ---------------------------------------------------------------------------
# 2. Automatic fallback
# ---------------------------------------------------------------------------


def test_failing_first_provider_falls_back_transparently(monkeypatch):
    """When the first-priority provider fails, the next one serves the request silently."""
    client, _redis_client, circuit_breaker = create_test_client(monkeypatch)
    monkeypatch.setattr(main_module, "ollama_provider", StubOllamaProvider())

    counter_before = read_fallback_counter("integration-fallback", "mock", "ollama")

    main_module.mock_provider.should_fail = True
    try:
        response = client.post(
            "/v1/chat",
            headers={"Authorization": "Bearer integration-fallback-key"},
            json=CHAT_REQUEST,
        )
    finally:
        # Explicit teardown: mock_provider is a module-level singleton shared by every test.
        main_module.mock_provider.should_fail = False

    # The client sees a plain success; nothing in the response signals the first provider failed.
    assert response.status_code == 200
    assert response.json()["provider"] == "ollama"

    counter_after = read_fallback_counter("integration-fallback", "mock", "ollama")
    assert counter_after == counter_before + 1

    # One failed request records a single failure, well below the open threshold, so the circuit
    # stays closed. Scenario 3 covers what happens once the threshold is exceeded.
    assert circuit_breaker.get_state("mock") == "closed"
    assert main_module.mock_provider.should_fail is False


# ---------------------------------------------------------------------------
# 3. Circuit breaker state transitions
# ---------------------------------------------------------------------------


def test_circuit_breaker_opens_then_half_opens_then_closes(monkeypatch):
    """Drive closed -> open -> half_open -> closed through real HTTP requests.

    The cooldown is rewound by editing `opened_at` directly rather than sleeping out the real
    `cooldown_seconds` (30 by default). Sleeping would make this one test longer than the entire
    rest of the suite for no extra coverage, and `test_circuit_breaker.py` already establishes
    timestamp manipulation as this project's convention for the same transition. The threshold and
    cooldown are read off the instance rather than hardcoded, so this tracks the real constants.
    """
    client, _redis_client, circuit_breaker = create_test_client(monkeypatch)

    assert circuit_breaker.get_state("mock") == "closed"

    main_module.mock_provider.should_fail = True
    try:
        for _ in range(circuit_breaker.failure_threshold):
            failed_response = client.post(
                "/v1/chat",
                headers={"Authorization": "Bearer integration-circuit-key"},
                json=CHAT_REQUEST,
            )
            # NOTE: an exhausted provider surfaces as 503, not 502/500. `call_chat_with_fallback`
            # runs out of candidates and `select_provider` raises "No healthy providers available"
            # (app/main.py:158-165), so an upstream provider failure is reported to the client as a
            # gateway-availability problem. Documented rather than judged.
            assert failed_response.status_code == 503

        assert circuit_breaker.get_state("mock") == "open"

        # While open, the provider is not attempted at all.
        open_response = client.post(
            "/v1/chat",
            headers={"Authorization": "Bearer integration-circuit-key"},
            json=CHAT_REQUEST,
        )
        assert open_response.status_code == 503
        assert "No healthy providers available" in open_response.json()["detail"]
        assert circuit_breaker.get_state("mock") == "open"

        # Rewind past the cooldown; the next attempt check should promote it to half_open.
        circuit_breaker.provider_states["mock"]["opened_at"] = datetime.now(
            timezone.utc
        ) - timedelta(seconds=circuit_breaker.cooldown_seconds + 1)

        assert circuit_breaker.can_attempt("mock") is True
        assert circuit_breaker.get_state("mock") == "half_open"

        main_module.mock_provider.should_fail = False
        recovery_response = client.post(
            "/v1/chat",
            headers={"Authorization": "Bearer integration-circuit-key"},
            json=CHAT_REQUEST,
        )
    finally:
        main_module.mock_provider.should_fail = False

    assert recovery_response.status_code == 200
    assert recovery_response.json()["provider"] == "mock"
    assert circuit_breaker.get_state("mock") == "closed"
    assert circuit_breaker.provider_states["mock"]["failure_count"] == 0


# ---------------------------------------------------------------------------
# 4. Budget enforcement
# ---------------------------------------------------------------------------


def test_budget_warning_then_hard_stop_at_cap(monkeypatch):
    """Spend crossing 80% sets X-Budget-Warning; reaching 100% returns 402.

    NOTE: `check_budget` compares spend accrued *before* the current request (app/main.py:299-317
    calls it ahead of the provider call), so the request that pushes a team over its cap still
    succeeds and only the following one is rejected. A team can therefore overspend its cap by one
    request. Asserted below as observed behavior, and flagged because a "sensible" reading of a hard
    cap would refuse the request that crosses it.
    """
    monkeypatch.setattr(budget_module, "MODEL_PRICING", PRICED_MOCK_MODEL)
    client, redis_client, _circuit_breaker = create_test_client(monkeypatch)

    headers = {"Authorization": "Bearer integration-budget-key"}
    budget = TEAMS_CONFIG["integration-budget-key"]["monthly_budget_usd"]

    # Spend before request: 0.0 -> below the 0.0008 warning threshold.
    first = client.post("/v1/chat", headers=headers, json=CHAT_REQUEST)
    assert first.status_code == 200
    assert "X-Budget-Warning" not in first.headers

    # Spend before request: 0.0004 -> still below the warning threshold.
    second = client.post("/v1/chat", headers=headers, json=CHAT_REQUEST)
    assert second.status_code == 200
    assert "X-Budget-Warning" not in second.headers

    # Spend before request: 0.0008 == 80% of 0.001 -> warns, and is still served.
    third = client.post("/v1/chat", headers=headers, json=CHAT_REQUEST)
    assert third.status_code == 200
    assert third.headers.get("X-Budget-Warning") == "true"

    # Spend before request: 0.0012 >= the 0.001 cap -> rejected.
    fourth = client.post("/v1/chat", headers=headers, json=CHAT_REQUEST)
    assert fourth.status_code == 402
    assert "budget" in fourth.json()["detail"].lower()

    # The cap was overshot by exactly one request's worth of spend, per the NOTE above.
    final_spend = read_spend(redis_client, "integration-budget")
    assert final_spend == pytest.approx(3 * PRICED_COST_PER_REQUEST)
    assert final_spend > budget


# ---------------------------------------------------------------------------
# 5. Rate limiting
# ---------------------------------------------------------------------------


def test_requests_beyond_rate_limit_are_rejected_with_429(monkeypatch):
    """A team limited to 2 req/min is served twice, then throttled within the same window."""
    client, _redis_client, _circuit_breaker = create_test_client(monkeypatch)
    headers = {"Authorization": "Bearer integration-ratelimit-key"}
    limit = TEAMS_CONFIG["integration-ratelimit-key"]["requests_per_minute"]

    allowed_responses = [
        client.post("/v1/chat", headers=headers, json=CHAT_REQUEST) for _ in range(limit)
    ]
    assert [response.status_code for response in allowed_responses] == [200] * limit

    for _ in range(2):
        throttled = client.post("/v1/chat", headers=headers, json=CHAT_REQUEST)
        assert throttled.status_code == 429
        assert throttled.headers.get("Retry-After") == "60"
        assert "rate limit" in throttled.json()["detail"].lower()
