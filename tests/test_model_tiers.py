"""Logical model tiers.

A tier is what makes a fallback chain able to serve a fallback. Without one, the providers
in a chain serve disjoint model sets, so every provider after the first rejects the request
and the chain does nothing — which was the state of this gateway before tiers existed.
"""

import pytest
from fastapi.testclient import TestClient

from app import budget as budget_module
from app import config as config_module
from app import main as main_module
from app.models.catalog import ModelCatalog
from app.models.schemas import ChatMessage, UnifiedChatResponse
from app.providers.errors import ProviderUnavailable
from tests.fakes import FakeRedis


CATALOG = ModelCatalog(
    providers={
        "groq": {"openai/gpt-oss-20b"},
        "ollama": {"llama3.2"},
        "mock": {"mock-model"},
    },
    tiers={
        "chat-general": {"groq": "openai/gpt-oss-20b", "ollama": "llama3.2"},
        "chat-local": {"ollama": "llama3.2"},
    },
)

TEAMS = {
    "tier-key": {
        "team_id": "tier-team",
        "allowed_models": ["chat-general", "chat-local"],
        "allowed_providers": ["groq", "ollama"],
        "provider_priority": ["groq", "ollama"],
        "requests_per_minute": 1000,
        "tokens_per_minute": 0,
        "monthly_budget_usd": 100.0,
    },
}

PRICED = {
    **config_module.MODEL_PRICING,
    # Deliberately unequal, so a worst-case reservation is distinguishable from a
    # first-candidate or cheapest-candidate one.
    "openai/gpt-oss-20b": {"input_price_per_1k": 0.01, "output_price_per_1k": 0.02},
    "llama3.2": {"input_price_per_1k": 0.0, "output_price_per_1k": 0.0},
}

REQUEST = {"model": "chat-general", "messages": [{"role": "user", "content": "hello"}]}


class RecordingProvider:
    """Records the model name it was actually asked for."""

    def __init__(self, provider_name: str, fails: bool = False) -> None:
        self.provider_name = provider_name
        self.fails = fails
        self.models_requested: list[str] = []

    async def chat(self, request):
        self.models_requested.append(request.model)
        if self.fails:
            raise ProviderUnavailable(f"{self.provider_name} is down")
        return UnifiedChatResponse(
            id=f"{self.provider_name}-response",
            model=request.model,
            content="ok",
            input_tokens=10,
            output_tokens=5,
            provider=self.provider_name,
            finish_reason="stop",
        )

    async def chat_stream(self, request):
        self.models_requested.append(request.model)
        yield "ok"


async def _no_backoff(provider, request, max_retries=3, base_delay=0.5):
    from app.retry import call_with_retry as real

    return await real(provider, request, max_retries=max_retries, base_delay=0.0)


def build(monkeypatch, groq_fails=False, ollama_fails=False):
    groq = RecordingProvider("groq", fails=groq_fails)
    ollama = RecordingProvider("ollama", fails=ollama_fails)

    monkeypatch.setattr(main_module.app.state, "teams_config", TEAMS)
    monkeypatch.setattr(main_module.app.state, "model_catalog", CATALOG)
    monkeypatch.setattr(
        main_module.app.state, "redis_client", FakeRedis(decode_responses=True)
    )
    monkeypatch.setattr(main_module.app.state, "circuit_breaker", __import__(
        "app.circuit_breaker", fromlist=["CircuitBreaker"]
    ).CircuitBreaker())
    monkeypatch.setattr(main_module, "groq_provider", groq)
    monkeypatch.setattr(main_module, "ollama_provider", ollama)
    monkeypatch.setattr(main_module, "call_with_retry", _no_backoff)
    monkeypatch.setattr(budget_module, "MODEL_PRICING", PRICED)

    return TestClient(main_module.app), groq, ollama


HEADERS = {"Authorization": "Bearer tier-key"}


# ---------------------------------------------------------------------------
# Catalog resolution
# ---------------------------------------------------------------------------


def test_a_tier_resolves_to_a_different_model_per_provider():
    assert CATALOG.resolve("chat-general") == {
        "groq": "openai/gpt-oss-20b",
        "ollama": "llama3.2",
    }


def test_a_physical_model_resolves_to_providers_that_serve_it():
    """The pre-tier path must keep working unchanged."""
    assert CATALOG.resolve("llama3.2") == {"ollama": "llama3.2"}
    assert CATALOG.resolve("openai/gpt-oss-20b") == {"groq": "openai/gpt-oss-20b"}


def test_an_unknown_model_resolves_to_nothing():
    assert CATALOG.resolve("no-such-model") == {}


def test_a_tier_naming_an_unknown_provider_is_rejected_on_load(tmp_path):
    """A bad tier would otherwise surface at request time as an unexplained provider 404."""
    path = tmp_path / "models.yaml"
    path.write_text(
        'providers:\n  mock:\n    - "mock-model"\ntiers:\n  t:\n    nope: "mock-model"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown provider"):
        config_module.load_model_catalog(str(path))


def test_a_tier_mapping_a_model_the_provider_does_not_serve_is_rejected(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        'providers:\n  mock:\n    - "mock-model"\ntiers:\n  t:\n    mock: "other"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not serve"):
        config_module.load_model_catalog(str(path))


def test_a_tier_cannot_shadow_a_provider_name(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        'providers:\n  mock:\n    - "mock-model"\ntiers:\n  mock:\n    mock: "mock-model"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="collides"):
        config_module.load_model_catalog(str(path))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_each_provider_is_asked_for_its_own_physical_model(monkeypatch):
    client, groq, ollama = build(monkeypatch)

    response = client.post("/v1/chat", headers=HEADERS, json=REQUEST)

    assert response.status_code == 200
    # First in the team's priority order, asked for the model it actually serves.
    assert groq.models_requested == ["openai/gpt-oss-20b"]
    assert ollama.models_requested == []
    assert response.json()["provider"] == "groq"
    assert response.json()["model"] == "openai/gpt-oss-20b"


def test_a_tier_falls_back_to_a_provider_serving_a_different_model(monkeypatch):
    """The whole point: one logical name, two providers, two physical models."""
    client, groq, ollama = build(monkeypatch, groq_fails=True)

    response = client.post("/v1/chat", headers=HEADERS, json=REQUEST)

    assert response.status_code == 200
    assert groq.models_requested == ["openai/gpt-oss-20b"] * 4  # retried, then gave up
    assert ollama.models_requested == ["llama3.2"]
    assert response.json()["provider"] == "ollama"
    assert response.json()["model"] == "llama3.2"


def test_a_tier_limited_to_one_provider_does_not_reach_the_others(monkeypatch):
    client, groq, ollama = build(monkeypatch)

    response = client.post(
        "/v1/chat", headers=HEADERS, json={**REQUEST, "model": "chat-local"}
    )

    assert response.status_code == 200
    assert groq.models_requested == []
    assert ollama.models_requested == ["llama3.2"]


def test_a_tier_outside_the_team_allowlist_is_refused(monkeypatch):
    client, _groq, _ollama = build(monkeypatch)

    response = client.post(
        "/v1/chat", headers=HEADERS, json={**REQUEST, "model": "chat-test"}
    )

    assert response.status_code == 403


def test_a_tier_cannot_reach_a_provider_the_team_is_not_allowed(monkeypatch):
    """A tier declares capability, not authorization: the allowlist still governs."""
    client, groq, ollama = build(monkeypatch)
    monkeypatch.setitem(
        main_module.app.state.teams_config,
        "tier-key",
        {**TEAMS["tier-key"], "allowed_providers": ["ollama"], "provider_priority": ["groq", "ollama"]},
    )

    response = client.post("/v1/chat", headers=HEADERS, json=REQUEST)

    assert response.status_code == 200
    assert groq.models_requested == []
    assert ollama.models_requested == ["llama3.2"]


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_a_tier_reserves_the_most_expensive_candidate(monkeypatch):
    """Reserving the cheapest would let a fallback to a pricier provider breach the cap."""
    messages = [ChatMessage(role="user", content="hello")]

    reservation = budget_module.estimate_max_cost(
        model="chat-general",
        messages=messages,
        max_output_tokens=100,
        candidate_models=CATALOG.physical_models_for("chat-general"),
    )

    # llama3.2 is free; gpt-oss-20b is not. The reservation must assume the paid one.
    expected = budget_module.calculate_cost("openai/gpt-oss-20b", 2, 100)
    assert reservation == pytest.approx(expected)
    assert reservation > 0


def test_cost_is_charged_against_the_model_that_actually_served_it(monkeypatch):
    """Groq fails over to the free local model, so the request must end up costing zero."""
    client, _groq, ollama = build(monkeypatch, groq_fails=True)

    response = client.post("/v1/chat", headers=HEADERS, json=REQUEST)

    assert response.status_code == 200
    assert response.json()["model"] == "llama3.2"

    import asyncio

    spend = asyncio.run(
        budget_module.get_current_spend("tier-team", main_module.app.state.redis_client)
    )
    # Reserved at the paid model's price, reconciled down to the free one it ran on.
    assert spend == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Fallback accounting
# ---------------------------------------------------------------------------


def read_fallback(team_id: str, from_provider: str, to_provider: str) -> float:
    from prometheus_client import REGISTRY

    value = REGISTRY.get_sample_value(
        "gateway_fallback_triggered_total",
        {
            "team_id": team_id,
            "from_provider": from_provider,
            "to_provider": to_provider,
        },
    )
    return 0.0 if value is None else value


def test_capability_routing_is_not_counted_as_a_fallback(monkeypatch):
    """Being served elsewhere because the preferred provider cannot host the model is
    routine routing, not a failover.

    The team prefers groq and asks for llama3.2, which only ollama serves. groq is healthy
    and was never called, so counting this as a groq-to-ollama fallback would make the
    metric fire on ordinary traffic and hide real outages.
    """
    client, groq, ollama = build(monkeypatch)
    before = read_fallback("tier-team", "groq", "ollama")

    response = client.post(
        "/v1/chat", headers=HEADERS, json={**REQUEST, "model": "chat-local"}
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "ollama"
    assert groq.models_requested == []
    assert read_fallback("tier-team", "groq", "ollama") == before


def test_a_real_failure_is_counted_as_a_fallback(monkeypatch):
    """The preferred provider could serve it and did not, so this is a genuine failover."""
    client, groq, ollama = build(monkeypatch, groq_fails=True)
    before = read_fallback("tier-team", "groq", "ollama")

    response = client.post("/v1/chat", headers=HEADERS, json=REQUEST)

    assert response.status_code == 200
    assert response.json()["provider"] == "ollama"
    assert groq.models_requested  # it was attempted, and failed
    assert read_fallback("tier-team", "groq", "ollama") == before + 1
