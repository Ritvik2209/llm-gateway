"""Tests for demo mode: the outage switch, the demo routes, and the config they rely on.

The outage switch is only worth having if the rest of the gateway cannot tell it from a
real outage, so the central test here drives the real fallback path through it — retry,
circuit breaker and all — rather than asserting on the switch alone.
"""

from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config as config_module
from app import main as main_module
from app.circuit_breaker import CircuitBreaker
from app.demo import build_demo_router
from app.health import HealthMonitor
from app.models.catalog import ModelCatalog
from app.models.schemas import ChatMessage, UnifiedChatRequest, UnifiedChatResponse
from app.providers.base import LLMProvider
from app.providers.errors import ProviderUnavailable
from app.providers.mock_provider import MockProvider
from app.providers.outage import OutageSwitch
from app.retry import call_with_retry as real_call_with_retry


REQUEST = UnifiedChatRequest(
    model="chat-demo",
    messages=[ChatMessage(role="user", content="hello")],
)

DEMO_TEAM = {
    "team_id": "demo-under-test",
    "allowed_models": ["chat-demo"],
    "allowed_providers": ["groq", "mock"],
    "provider_priority": ["groq", "mock"],
}

DEMO_CATALOG = ModelCatalog(
    providers={"groq": {"openai/gpt-oss-20b"}, "mock": {"mock-model"}},
    tiers={"chat-demo": {"groq": "openai/gpt-oss-20b", "mock": "mock-model"}},
)


class FakeClock:
    """A monotonic clock the test moves by hand, so expiry needs no sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class RecordingProvider(LLMProvider):
    """Stands in for Groq, and counts every call that actually reaches it."""

    provider_name = "groq"

    def __init__(self) -> None:
        self.calls = 0
        self.last_stream_usage = {"input_tokens": 3, "output_tokens": 4}

    async def chat(self, request):
        self.calls += 1
        return UnifiedChatResponse(
            id="groq-response",
            model=request.model,
            content="real answer",
            input_tokens=10,
            output_tokens=5,
            provider=self.provider_name,
            finish_reason="stop",
        )

    async def chat_stream(self, request):
        self.calls += 1
        yield "real "
        yield "answer"


async def _no_backoff(provider, request, max_retries=3, base_delay=0.5):
    """Real retry semantics, without the 3.5 s of real backoff per exhausted provider."""
    return await real_call_with_retry(
        provider, request, max_retries=max_retries, base_delay=0.0
    )


# ---------------------------------------------------------------------------
# The switch itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calls_pass_through_when_no_outage_is_running():
    inner = RecordingProvider()
    switch = OutageSwitch(inner, clock=FakeClock())

    response = await switch.chat(REQUEST)

    assert response.content == "real answer"
    assert inner.calls == 1


@pytest.mark.asyncio
async def test_an_outage_fails_without_reaching_the_real_provider():
    """Failing before the call is what keeps a demo outage from spending Groq quota."""
    inner = RecordingProvider()
    switch = OutageSwitch(inner, clock=FakeClock())
    switch.start_outage(60)

    with pytest.raises(ProviderUnavailable):
        await switch.chat(REQUEST)

    assert inner.calls == 0


@pytest.mark.asyncio
async def test_an_outage_ends_on_its_own():
    clock = FakeClock()
    inner = RecordingProvider()
    switch = OutageSwitch(inner, clock=clock)
    switch.start_outage(60)

    clock.now += 60.5

    assert switch.outage_remaining_seconds() == 0
    assert (await switch.chat(REQUEST)).content == "real answer"


def test_a_running_outage_is_not_extended():
    """Otherwise repeated presses could hold the provider down indefinitely."""
    clock = FakeClock()
    switch = OutageSwitch(RecordingProvider(), clock=clock)

    assert switch.start_outage(60) is True
    clock.now += 30
    assert switch.start_outage(60) is False
    assert switch.outage_remaining_seconds() == pytest.approx(30)


@pytest.mark.asyncio
async def test_streaming_fails_during_an_outage_and_streams_otherwise():
    clock = FakeClock()
    switch = OutageSwitch(RecordingProvider(), clock=clock)

    assert [chunk async for chunk in switch.chat_stream(REQUEST)] == ["real ", "answer"]

    switch.start_outage(60)
    with pytest.raises(ProviderUnavailable):
        async for _ in switch.chat_stream(REQUEST):
            pass


def test_the_wrapper_keeps_the_provider_name_and_forwards_other_attributes():
    """The streaming path reads last_stream_usage off whatever provider served it."""
    switch = OutageSwitch(RecordingProvider())

    assert switch.provider_name == "groq"
    assert switch.last_stream_usage == {"input_tokens": 3, "output_tokens": 4}


# ---------------------------------------------------------------------------
# The real resilience path, driven through the switch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_outage_falls_back_opens_the_circuit_and_recovers(monkeypatch):
    """The sequence a demo visitor sees, end to end through call_chat_with_fallback.

    Fallback while Groq is down, the circuit opening after the failure threshold, and the
    circuit closing again on the first trial request once the outage is over.
    """
    monkeypatch.setattr(main_module, "call_with_retry", _no_backoff)
    clock = FakeClock()
    inner = RecordingProvider()
    switch = OutageSwitch(inner, clock=clock)
    providers = {"groq": switch, "mock": MockProvider()}
    circuit_breaker = CircuitBreaker()
    monitor = HealthMonitor()

    async def ask():
        return await main_module.call_chat_with_fallback(
            DEMO_TEAM,
            REQUEST,
            monitor,
            circuit_breaker,
            providers,
            model_catalog=DEMO_CATALOG,
        )

    switch.start_outage(60)
    for _ in range(circuit_breaker.failure_threshold):
        response = await ask()
        assert response.provider == "mock"
        assert response.model == "mock-model"

    assert circuit_breaker.get_state("groq") == "open"
    # With the circuit open, Groq is skipped outright rather than attempted and failed.
    assert (await ask()).provider == "mock"

    clock.now += 61
    circuit_breaker.provider_states["groq"]["opened_at"] -= timedelta(
        seconds=circuit_breaker.cooldown_seconds + 1
    )

    response = await ask()
    assert response.provider == "groq"
    assert response.model == "openai/gpt-oss-20b"
    assert circuit_breaker.get_state("groq") == "closed"
    assert inner.calls == 1


# ---------------------------------------------------------------------------
# Demo routes
# ---------------------------------------------------------------------------


def build_demo_client(teams_config):
    clock = FakeClock()
    switch = OutageSwitch(RecordingProvider(), clock=clock)
    app = FastAPI()
    app.state.teams_config = teams_config
    app.state.circuit_breaker = CircuitBreaker()
    app.include_router(
        build_demo_router(switch, outage_seconds=60, grafana_url="https://grafana.test")
    )
    return TestClient(app), switch, app.state.circuit_breaker, clock


PUBLISHED_TEAMS = {
    "public-key": {
        **DEMO_TEAM,
        "team_id": "demo-public",
        "requests_per_minute": 20,
        "tokens_per_minute": 4000,
        "monthly_budget_usd": 1.0,
        "public_demo": True,
    },
    "private-key": {**DEMO_TEAM, "team_id": "private-team", "public_demo": False},
    # Refused at load time; excluded here too, so a config that skipped validation still
    # cannot publish an admin key.
    "admin-key": {**DEMO_TEAM, "team_id": "admin", "public_demo": True, "is_admin": True},
}


def test_only_public_non_admin_teams_are_published():
    client, *_ = build_demo_client(PUBLISHED_TEAMS)

    teams = client.get("/demo/teams").json()["teams"]

    assert [team["team_id"] for team in teams] == ["demo-public"]
    assert teams[0]["api_key"] == "public-key"
    assert teams[0]["provider_chain"] == ["groq", "mock"]
    assert teams[0]["requests_per_minute"] == 20


def test_the_outage_endpoint_starts_one_outage_at_a_time():
    client, switch, _breaker, clock = build_demo_client(PUBLISHED_TEAMS)

    first = client.post("/demo/outage").json()
    clock.now += 10
    second = client.post("/demo/outage").json()

    assert first["started"] is True
    assert first["outage_remaining_seconds"] == 60
    assert second["started"] is False
    assert second["outage_remaining_seconds"] == 50
    assert switch.outage_remaining_seconds() == 50


def test_status_reports_an_open_circuit_and_the_time_to_its_trial_request():
    client, _switch, breaker, _clock = build_demo_client(PUBLISHED_TEAMS)

    closed = client.get("/demo/status").json()
    for _ in range(breaker.failure_threshold):
        breaker.record_failure("groq")
    opened = client.get("/demo/status").json()

    assert closed["circuit_state"] == "closed"
    assert closed["circuit_retry_in_seconds"] is None
    assert closed["grafana_url"] == "https://grafana.test"
    assert opened["circuit_state"] == "open"
    assert 0 < opened["circuit_retry_in_seconds"] <= breaker.cooldown_seconds


def test_the_page_is_served_at_the_root():
    client, *_ = build_demo_client(PUBLISHED_TEAMS)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "LLM Gateway demo" in response.text


def test_demo_routes_and_the_switch_exist_only_in_demo_mode():
    """An unauthenticated way to take a provider down must not ship by default."""
    paths = {route.path for route in main_module.app.routes}

    assert ("/demo/outage" in paths) is config_module.DEMO_MODE
    assert isinstance(main_module.groq_provider, OutageSwitch) is config_module.DEMO_MODE


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_the_public_demo_flag_is_loaded(tmp_path):
    path = tmp_path / "teams.yaml"
    path.write_text(
        "teams:\n"
        '  - team_id: "demo"\n'
        '    api_key: "demo-key"\n'
        "    public_demo: true\n"
        '  - team_id: "other"\n'
        '    api_key: "other-key"\n',
        encoding="utf-8",
    )

    teams = config_module.load_teams_config(str(path))

    assert teams["demo-key"]["public_demo"] is True
    assert teams["other-key"]["public_demo"] is False


def test_a_public_admin_team_is_rejected_at_load(tmp_path):
    path = tmp_path / "teams.yaml"
    path.write_text(
        "teams:\n"
        '  - team_id: "oops"\n'
        '    api_key: "oops-key"\n'
        "    public_demo: true\n"
        "    is_admin: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot be an admin key"):
        config_module.load_teams_config(str(path))
