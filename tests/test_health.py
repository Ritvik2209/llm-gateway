import pytest

from app.circuit_breaker import CircuitBreaker
from app.health import HealthMonitor
from app.providers.mock_provider import MockProvider


class CapturingMockProvider(MockProvider):
    def __init__(self, should_fail=False, latency_seconds=0):
        super().__init__(should_fail=should_fail, latency_seconds=latency_seconds)
        self.requests = []

    async def chat(self, request):
        self.requests.append(request)
        return await super().chat(request)


@pytest.mark.asyncio
async def test_healthy_provider_is_marked_healthy_after_check():
    monitor = HealthMonitor()
    provider = CapturingMockProvider(should_fail=False, latency_seconds=0)
    monitor.register_provider(provider, health_check_model="mock-model")

    await monitor.check_provider_health("mock")

    assert monitor.get_status("mock") == "healthy"
    assert monitor.provider_health[("mock", "mock-model")].consecutive_failures == 0
    assert monitor.provider_health[("mock", "mock-model")].error_rate == 0.0
    assert provider.requests[0].model == "mock-model"


@pytest.mark.asyncio
async def test_failing_provider_degrades_then_goes_down():
    monitor = HealthMonitor()
    provider = MockProvider(should_fail=True, latency_seconds=0)
    monitor.register_provider(provider, health_check_model="mock-model")

    await monitor.check_provider_health("mock")
    await monitor.check_provider_health("mock")

    assert monitor.get_status("mock") == "degraded"
    assert monitor.provider_health[("mock", "mock-model")].consecutive_failures == 2

    await monitor.check_provider_health("mock")
    await monitor.check_provider_health("mock")

    assert monitor.get_status("mock") == "down"
    assert monitor.provider_health[("mock", "mock-model")].consecutive_failures == 4


def test_get_status_returns_unknown_for_never_checked_provider():
    monitor = HealthMonitor()

    assert monitor.get_status("missing") == "unknown"


def test_should_probe_when_provider_has_never_been_seen():
    """With no health data there is no cheaper signal, so a probe is worth its cost."""
    monitor = HealthMonitor()
    circuit_breaker = CircuitBreaker()

    assert monitor.should_probe("mock", circuit_breaker) is True


def test_should_not_probe_a_closed_provider_with_health_data():
    """A provider serving real traffic reports its own health for free.

    Probing it anyway is what exhausted a Groq free-tier daily quota: a fixed-interval
    loop against a metered dependency consumes the capacity it exists to protect.
    """
    monitor = HealthMonitor()
    circuit_breaker = CircuitBreaker()
    monitor.record_request_outcome("mock", "mock-model", succeeded=True, latency=0.01)

    assert monitor.should_probe("mock", circuit_breaker) is False


def test_should_probe_while_the_circuit_is_not_closed():
    """An open circuit withholds real traffic, so a probe is the only recovery signal."""
    monitor = HealthMonitor()
    circuit_breaker = CircuitBreaker(failure_threshold=1)
    monitor.record_request_outcome("mock", "mock-model", succeeded=True, latency=0.01)
    circuit_breaker.record_failure("mock")

    assert circuit_breaker.get_state("mock") == "open"
    assert monitor.should_probe("mock", circuit_breaker) is True


def test_passive_outcomes_drive_status_like_probes_do():
    """Real request outcomes must move status on the same thresholds as probes."""
    monitor = HealthMonitor()

    monitor.record_request_outcome("mock", "mock-model", succeeded=True, latency=0.01)
    assert monitor.get_status("mock") == "healthy"

    monitor.record_request_outcome("mock", "mock-model", succeeded=False)
    assert monitor.get_status("mock") == "degraded"

    monitor.record_request_outcome("mock", "mock-model", succeeded=False)
    assert monitor.get_status("mock") == "degraded"

    monitor.record_request_outcome("mock", "mock-model", succeeded=False)
    monitor.record_request_outcome("mock", "mock-model", succeeded=False)
    assert monitor.get_status("mock") == "down"


def test_a_successful_outcome_clears_consecutive_failures():
    monitor = HealthMonitor()
    for _ in range(4):
        monitor.record_request_outcome("mock", "mock-model", succeeded=False)
    assert monitor.get_status("mock") == "down"

    monitor.record_request_outcome("mock", "mock-model", succeeded=True, latency=0.02)

    assert monitor.provider_health[("mock", "mock-model")].consecutive_failures == 0
    assert monitor.provider_health[("mock", "mock-model")].recent_latencies == [0.02]


def test_passive_outcomes_cost_no_provider_requests():
    """The point of the change: health data accrues without spending provider quota."""
    monitor = HealthMonitor()
    provider = CapturingMockProvider(should_fail=False, latency_seconds=0)
    monitor.register_provider(provider, health_check_model="mock-model")

    for _ in range(50):
        monitor.record_request_outcome("mock", "mock-model", succeeded=True, latency=0.01)

    assert monitor.get_status("mock") == "healthy"
    assert provider.requests == []


def test_health_status_is_published_as_a_metric():
    """Health lived only in the /admin/health JSON, so no dashboard or alert could see it."""
    from prometheus_client import REGISTRY

    monitor = HealthMonitor()
    provider = CapturingMockProvider(should_fail=False, latency_seconds=0)
    monitor.register_provider(provider, health_check_model="mock-model")

    def state() -> float | None:
        return REGISTRY.get_sample_value(
            "gateway_provider_health",
            {"provider_name": "mock", "model": "mock-model"},
        )

    # Registered but never observed reads -1, which a dashboard renders differently from
    # both "healthy" and an absent series.
    assert state() == -1

    monitor.record_request_outcome("mock", "mock-model", succeeded=True, latency=0.01)
    assert state() == 0

    for _ in range(2):
        monitor.record_request_outcome("mock", "mock-model", succeeded=False)
    assert state() == 1

    for _ in range(2):
        monitor.record_request_outcome("mock", "mock-model", succeeded=False)
    assert state() == 2


# ---------------------------------------------------------------------------
# Per provider-model granularity
#
# Health was keyed on the provider alone, which was sufficient while each provider
# served exactly one model here. Logical model tiers broke that assumption: a provider
# can be quota-exhausted on one model and serving another normally, and conflating them
# marks the whole provider down, diverting traffic that would have succeeded.
# ---------------------------------------------------------------------------


def test_one_failing_model_does_not_condemn_the_others():
    """The case tiers made reachable, and the reason this granularity exists."""
    monitor = HealthMonitor()

    for _ in range(4):
        monitor.record_request_outcome("groq", "openai/gpt-oss-20b", succeeded=False)
    monitor.record_request_outcome("groq", "llama-3.3-70b", succeeded=True, latency=0.4)

    assert monitor.get_status("groq", "openai/gpt-oss-20b") == "down"
    assert monitor.get_status("groq", "llama-3.3-70b") == "healthy"


def test_a_provider_is_degraded_not_down_while_one_model_still_works():
    """Reporting a provider down while it serves most of its traffic would be wrong.

    "Down" has to mean unusable. A provider failing one model of several is impaired,
    and the aggregate says so without overstating it.
    """
    monitor = HealthMonitor()

    for _ in range(4):
        monitor.record_request_outcome("groq", "openai/gpt-oss-20b", succeeded=False)
    monitor.record_request_outcome("groq", "llama-3.3-70b", succeeded=True, latency=0.4)

    assert monitor.get_status("groq") == "degraded"


def test_a_provider_is_down_only_when_every_model_is_down():
    monitor = HealthMonitor()

    for model in ("openai/gpt-oss-20b", "llama-3.3-70b"):
        for _ in range(4):
            monitor.record_request_outcome("groq", model, succeeded=False)

    assert monitor.get_status("groq") == "down"


def test_unknown_models_do_not_drag_the_aggregate_down():
    """A registered but unobserved pair is absence of evidence, not evidence of failure."""
    monitor = HealthMonitor()
    provider = CapturingMockProvider(should_fail=False, latency_seconds=0)
    monitor.register_provider(provider, health_check_model="mock-model")

    monitor.record_request_outcome("mock", "mock-model", succeeded=True, latency=0.01)

    assert monitor.get_status("mock") == "healthy"


def test_models_for_lists_only_observed_pairs():
    monitor = HealthMonitor()
    monitor.record_request_outcome("groq", "openai/gpt-oss-20b", succeeded=True, latency=0.4)
    monitor.record_request_outcome("groq", "llama-3.3-70b", succeeded=True, latency=0.5)
    monitor.record_request_outcome("ollama", "llama3.2", succeeded=True, latency=2.0)

    assert sorted(monitor.models_for("groq")) == ["llama-3.3-70b", "openai/gpt-oss-20b"]
    assert monitor.models_for("ollama") == ["llama3.2"]
    assert monitor.models_for("mock") == []


def test_each_pair_publishes_its_own_metric_series():
    from prometheus_client import REGISTRY

    monitor = HealthMonitor()

    def state(model: str) -> float | None:
        return REGISTRY.get_sample_value(
            "gateway_provider_health",
            {"provider_name": "metrics-probe", "model": model},
        )

    monitor.record_request_outcome("metrics-probe", "model-a", succeeded=True, latency=0.1)
    for _ in range(4):
        monitor.record_request_outcome("metrics-probe", "model-b", succeeded=False)

    assert state("model-a") == 0
    assert state("model-b") == 2


def test_traffic_on_any_model_suppresses_a_probe():
    """Probing is quota spend, so any evidence the provider is reachable is enough.

    Keying this on the probe pair alone would probe a provider we already have a signal
    for - the exact failure that exhausted a Groq daily quota once already.
    """
    monitor = HealthMonitor()
    circuit_breaker = CircuitBreaker()
    provider = CapturingMockProvider(should_fail=False, latency_seconds=0)
    monitor.register_provider(provider, health_check_model="probe-model")

    monitor.record_request_outcome("mock", "some-other-model", succeeded=True, latency=0.01)

    assert monitor.should_probe("mock", circuit_breaker) is False
