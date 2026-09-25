import pytest
from fastapi import HTTPException, status

from app.circuit_breaker import CircuitBreaker
from app.health import HealthMonitor, ProviderHealth
from app.main import select_provider
from app.models.schemas import UnifiedChatResponse
from app.providers.base import LLMProvider


class DummyProvider(LLMProvider):
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name

    async def chat(self, request):
        return UnifiedChatResponse(
            id=f"{self.provider_name}-response",
            model=request.model,
            content="ok",
            input_tokens=0,
            output_tokens=0,
            provider=self.provider_name,
            finish_reason="stop",
        )

    async def chat_stream(self, request):
        yield "ok"


def make_providers() -> dict[str, LLMProvider]:
    return {
        "ollama": DummyProvider("ollama"),
        "mock": DummyProvider("mock"),
    }


def make_team_config() -> dict:
    # allowed_providers is the authorization boundary and is enforced in routing, so a
    # team config that omits it authorizes nothing. It must list every provider the
    # priority chain is allowed to reach.
    return {
        "allowed_providers": ["ollama", "mock"],
        "provider_priority": ["ollama", "mock"],
    }


def set_status(monitor: HealthMonitor, provider_name: str, provider_status: str) -> None:
    monitor.provider_health[provider_name] = ProviderHealth(status=provider_status)


def test_selects_first_provider_when_circuit_is_closed():
    monitor = HealthMonitor()
    providers = make_providers()
    circuit_breaker = CircuitBreaker()

    selected_provider = select_provider(
        make_team_config(),
        monitor,
        providers,
        circuit_breaker,
    )

    assert selected_provider.provider_name == "ollama"


def test_health_status_does_not_affect_provider_selection():
    monitor = HealthMonitor()
    providers = make_providers()
    set_status(monitor, "ollama", "down")
    set_status(monitor, "mock", "healthy")
    circuit_breaker = CircuitBreaker()

    selected_provider = select_provider(
        make_team_config(),
        monitor,
        providers,
        circuit_breaker,
    )

    assert selected_provider.provider_name == "ollama"


def test_skips_provider_when_circuit_breaker_disallows_attempt():
    monitor = HealthMonitor()
    providers = make_providers()
    circuit_breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30)
    set_status(monitor, "ollama", "healthy")
    set_status(monitor, "mock", "healthy")
    circuit_breaker.record_failure("ollama")

    selected_provider = select_provider(
        make_team_config(),
        monitor,
        providers,
        circuit_breaker,
    )

    assert selected_provider.provider_name == "mock"


def test_selects_unknown_health_provider_when_circuit_is_closed():
    monitor = HealthMonitor()
    providers = make_providers()
    set_status(monitor, "mock", "healthy")
    circuit_breaker = CircuitBreaker()

    selected_provider = select_provider(
        make_team_config(),
        monitor,
        providers,
        circuit_breaker,
    )

    assert selected_provider.provider_name == "ollama"


def test_raises_503_when_all_priority_provider_circuits_are_open():
    monitor = HealthMonitor()
    providers = make_providers()
    circuit_breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30)
    circuit_breaker.record_failure("ollama")
    circuit_breaker.record_failure("mock")

    with pytest.raises(HTTPException) as exc_info:
        select_provider(make_team_config(), monitor, providers, circuit_breaker)

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "ollama, mock" in exc_info.value.detail


def test_provider_outside_the_allowlist_is_never_selected():
    """provider_priority must not grant access to a provider the team cannot use."""
    monitor = HealthMonitor()
    providers = make_providers()
    circuit_breaker = CircuitBreaker()
    team_config = {
        "allowed_providers": ["mock"],
        "provider_priority": ["ollama", "mock"],
    }

    selected_provider = select_provider(
        team_config,
        monitor,
        providers,
        circuit_breaker,
    )

    assert selected_provider.provider_name == "mock"


def test_empty_allowlist_denies_every_provider():
    """Enforcement is fail-closed: an incomplete config must not grant access."""
    monitor = HealthMonitor()
    providers = make_providers()
    circuit_breaker = CircuitBreaker()
    team_config = {"provider_priority": ["ollama", "mock"]}

    with pytest.raises(HTTPException) as exc_info:
        select_provider(team_config, monitor, providers, circuit_breaker)

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "none" in exc_info.value.detail
