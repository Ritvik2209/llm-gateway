"""Tests for demo mode: the outage switch, the demo routes, and the config they rely on.

The outage switch is only worth having if the rest of the gateway cannot tell it from a
real outage, so the central test here drives the real fallback path through it — retry,
circuit breaker and all — rather than asserting on the switch alone.
"""

from datetime import timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config as config_module
from app import main as main_module
from app.circuit_breaker import CircuitBreaker
from app.demo import build_demo_router, demo_provider_names, public_demo_teams
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


@pytest.mark.asyncio
async def test_a_three_provider_chain_falls_back_one_step_per_outage(monkeypatch):
    """groq -> ollama -> mock, each provider asked for its own model in the tier."""
    monkeypatch.setattr(main_module, "call_with_retry", _no_backoff)
    clock = FakeClock()
    groq = OutageSwitch(recording_provider("groq"), clock=clock)
    ollama = OutageSwitch(recording_provider("ollama"), clock=clock)
    providers = {"groq": groq, "ollama": ollama, "mock": MockProvider()}
    team = {
        **DEMO_TEAM,
        "allowed_providers": ["groq", "ollama", "mock"],
        "provider_priority": ["groq", "ollama", "mock"],
    }
    catalog = ModelCatalog(
        providers={
            "groq": {"openai/gpt-oss-20b"},
            "ollama": {"llama3.2"},
            "mock": {"mock-model"},
        },
        tiers={
            "chat-demo": {
                "groq": "openai/gpt-oss-20b",
                "ollama": "llama3.2",
                "mock": "mock-model",
            }
        },
    )
    circuit_breaker = CircuitBreaker()

    async def served_by():
        response = await main_module.call_chat_with_fallback(
            team, REQUEST, HealthMonitor(), circuit_breaker, providers, model_catalog=catalog
        )
        return response.provider, response.model

    assert await served_by() == ("groq", "openai/gpt-oss-20b")
    groq.start_outage(60)
    assert await served_by() == ("ollama", "llama3.2")
    ollama.start_outage(60)
    assert await served_by() == ("mock", "mock-model")


# ---------------------------------------------------------------------------
# Demo routes
# ---------------------------------------------------------------------------


def recording_provider(name: str) -> RecordingProvider:
    provider = RecordingProvider()
    provider.provider_name = name
    return provider


def build_demo_client(teams_config):
    clock = FakeClock()
    switches = {
        name: OutageSwitch(recording_provider(name), clock=clock)
        for name in ("groq", "ollama")
    }
    app = FastAPI()
    app.state.teams_config = teams_config
    app.state.model_catalog = ROUTING_CATALOG
    app.state.circuit_breaker = CircuitBreaker()
    app.include_router(
        build_demo_router(switches, outage_seconds=60, grafana_url="https://grafana.test")
    )
    return TestClient(app), switches, app.state.circuit_breaker, clock


# chat-demo maps Ollama as well, as config/models.yaml does, so the routing test can show
# that a team whose chain omits Ollama is never told it would go there.
ROUTING_CATALOG = ModelCatalog(
    providers={
        "groq": {"openai/gpt-oss-20b"},
        "ollama": {"llama3.2"},
        "mock": {"mock-model"},
    },
    tiers={
        "chat-demo": {
            "groq": "openai/gpt-oss-20b",
            "ollama": "llama3.2",
            "mock": "mock-model",
        }
    },
)


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

WITH_OLLAMA_TEAM = {
    **PUBLISHED_TEAMS,
    "ollama-key": {
        **DEMO_TEAM,
        "team_id": "demo-ollama",
        "allowed_providers": ["ollama", "mock"],
        "provider_priority": ["ollama", "mock"],
        "public_demo": True,
    },
}


def test_only_public_non_admin_teams_are_published():
    client, *_ = build_demo_client(PUBLISHED_TEAMS)

    teams = client.get("/demo/teams").json()["teams"]

    assert [team["team_id"] for team in teams] == ["demo-public"]
    assert teams[0]["api_key"] == "public-key"
    assert teams[0]["provider_chain"] == ["groq", "mock"]
    assert teams[0]["requests_per_minute"] == 20


def test_each_model_shows_where_it_goes_through_this_teams_chain():
    """A tier resolves per provider, but only across providers the team may reach."""
    teams_config = {
        "public-key": {
            **PUBLISHED_TEAMS["public-key"],
            "allowed_models": ["chat-demo", "mock-model", "no-such-model"],
        }
    }
    client, *_ = build_demo_client(teams_config)

    models = client.get("/demo/teams").json()["teams"][0]["models"]
    routes = {model["name"]: model["routes"] for model in models}

    assert routes["chat-demo"] == [
        {"provider": "groq", "model": "openai/gpt-oss-20b"},
        {"provider": "mock", "model": "mock-model"},
    ]
    assert routes["mock-model"] == [{"provider": "mock", "model": "mock-model"}]
    assert routes["no-such-model"] == []


def test_the_outage_endpoint_starts_one_outage_at_a_time():
    client, switches, _breaker, clock = build_demo_client(PUBLISHED_TEAMS)

    first = client.post("/demo/outage/groq").json()
    clock.now += 10
    second = client.post("/demo/outage/groq").json()

    assert first["started"] is True
    assert first["providers"][0]["outage_remaining_seconds"] == 60
    assert second["started"] is False
    assert second["providers"][0]["outage_remaining_seconds"] == 50
    assert switches["groq"].outage_remaining_seconds() == 50


def test_status_reports_an_open_circuit_and_the_time_to_its_trial_request():
    client, _switches, breaker, _clock = build_demo_client(PUBLISHED_TEAMS)

    closed = client.get("/demo/status").json()
    for _ in range(breaker.failure_threshold):
        breaker.record_failure("groq")
    opened = client.get("/demo/status").json()

    assert closed["grafana_url"] == "https://grafana.test"
    assert closed["providers"][0]["circuit_state"] == "closed"
    assert closed["providers"][0]["circuit_retry_in_seconds"] is None
    assert opened["providers"][0]["circuit_state"] == "open"
    retry_in = opened["providers"][0]["circuit_retry_in_seconds"]
    assert 0 < retry_in <= breaker.cooldown_seconds


def test_only_providers_a_demo_team_uses_are_shown_or_breakable():
    """A server without Ollama in any demo chain must not offer to take Ollama down."""
    client, switches, _breaker, _clock = build_demo_client(PUBLISHED_TEAMS)

    names = [p["name"] for p in client.get("/demo/status").json()["providers"]]
    response = client.post("/demo/outage/ollama")

    assert names == ["groq"]
    assert response.status_code == 404
    assert switches["ollama"].outage_remaining_seconds() == 0


def test_each_demo_provider_has_its_own_outage():
    client, switches, _breaker, _clock = build_demo_client(WITH_OLLAMA_TEAM)

    names = [p["name"] for p in client.get("/demo/status").json()["providers"]]
    response = client.post("/demo/outage/ollama").json()

    assert names == ["groq", "ollama"]
    assert response["started"] is True
    assert switches["ollama"].outage_remaining_seconds() == 60
    assert switches["groq"].outage_remaining_seconds() == 0


def test_mock_can_never_be_taken_down():
    """Mock ends every demo chain; breaking it would replace fallback with 503s."""
    client, *_ = build_demo_client(WITH_OLLAMA_TEAM)

    assert client.post("/demo/outage/mock").status_code == 404


def test_the_page_is_served_at_the_root():
    client, *_ = build_demo_client(PUBLISHED_TEAMS)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "LLM Gateway demo" in response.text


def test_demo_routes_and_the_switches_exist_only_in_demo_mode():
    """An unauthenticated way to take a provider down must not ship by default."""
    paths = {route.path for route in main_module.app.routes}

    assert ("/demo/outage/{provider_name}" in paths) is config_module.DEMO_MODE
    assert isinstance(main_module.groq_provider, OutageSwitch) is config_module.DEMO_MODE
    assert isinstance(main_module.ollama_provider, OutageSwitch) is config_module.DEMO_MODE


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


# The two committed teams files a demo instance can run: the server's template and the full
# laptop demo. Both are loaded as written, so a regex or YAML mistake fails here rather than
# when the gateway starts.
REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_TEAMS = REPO_ROOT / "deploy" / "teams.example.yaml"
LAPTOP_TEAMS = REPO_ROOT / "config" / "teams.demo.yaml"
MODEL_CATALOG = REPO_ROOT / "config" / "models.yaml"


@pytest.mark.parametrize("teams_path", [SERVER_TEAMS, LAPTOP_TEAMS], ids=["server", "laptop"])
def test_every_model_a_demo_page_offers_has_somewhere_to_go(teams_path):
    """A model with no route through its team's chain would be a dropdown entry that 400s."""
    teams_config = config_module.load_teams_config(str(teams_path))
    catalog = config_module.load_model_catalog(str(MODEL_CATALOG))

    published = public_demo_teams(teams_config, catalog)

    assert published
    for team in published:
        for model in team["models"]:
            assert model["routes"], f'{team["team_id"]} offers {model["name"]} with no route'


def test_the_laptop_demo_is_the_server_demo_plus_ollama():
    """Same page, same server teams; the laptop adds teams and an Ollama outage button."""
    server = config_module.load_teams_config(str(SERVER_TEAMS))
    laptop = config_module.load_teams_config(str(LAPTOP_TEAMS))
    switches = dict.fromkeys(["groq", "ollama"])

    server_public = {key: team for key, team in server.items() if team["public_demo"]}

    assert server_public
    assert {key: laptop.get(key) for key in server_public} == server_public
    assert demo_provider_names(server, switches) == ["groq"]
    assert demo_provider_names(laptop, switches) == ["groq", "ollama"]
    assert [team["team_id"] for team in public_demo_teams(laptop)] == [
        "demo-public",
        "demo-policy",
        "demo-chain",
        "demo-ollama",
        "team-alpha",
        "team-beta",
    ]
