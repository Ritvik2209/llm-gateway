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
    assert monitor.provider_health["mock"].consecutive_failures == 0
    assert monitor.provider_health["mock"].error_rate == 0.0
    assert provider.requests[0].model == "mock-model"


@pytest.mark.asyncio
async def test_failing_provider_degrades_then_goes_down():
    monitor = HealthMonitor()
    provider = MockProvider(should_fail=True, latency_seconds=0)
    monitor.register_provider(provider, health_check_model="mock-model")

    await monitor.check_provider_health("mock")
    await monitor.check_provider_health("mock")

    assert monitor.get_status("mock") == "degraded"
    assert monitor.provider_health["mock"].consecutive_failures == 2

    await monitor.check_provider_health("mock")
    await monitor.check_provider_health("mock")

    assert monitor.get_status("mock") == "down"
    assert monitor.provider_health["mock"].consecutive_failures == 4


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
    monitor.record_request_outcome("mock", succeeded=True, latency=0.01)

    assert monitor.should_probe("mock", circuit_breaker) is False


def test_should_probe_while_the_circuit_is_not_closed():
    """An open circuit withholds real traffic, so a probe is the only recovery signal."""
    monitor = HealthMonitor()
    circuit_breaker = CircuitBreaker(failure_threshold=1)
    monitor.record_request_outcome("mock", succeeded=True, latency=0.01)
    circuit_breaker.record_failure("mock")

    assert circuit_breaker.get_state("mock") == "open"
    assert monitor.should_probe("mock", circuit_breaker) is True


def test_passive_outcomes_drive_status_like_probes_do():
    """Real request outcomes must move status on the same thresholds as probes."""
    monitor = HealthMonitor()

    monitor.record_request_outcome("mock", succeeded=True, latency=0.01)
    assert monitor.get_status("mock") == "healthy"

    monitor.record_request_outcome("mock", succeeded=False)
    assert monitor.get_status("mock") == "degraded"

    monitor.record_request_outcome("mock", succeeded=False)
    assert monitor.get_status("mock") == "degraded"

    monitor.record_request_outcome("mock", succeeded=False)
    monitor.record_request_outcome("mock", succeeded=False)
    assert monitor.get_status("mock") == "down"


def test_a_successful_outcome_clears_consecutive_failures():
    monitor = HealthMonitor()
    for _ in range(4):
        monitor.record_request_outcome("mock", succeeded=False)
    assert monitor.get_status("mock") == "down"

    monitor.record_request_outcome("mock", succeeded=True, latency=0.02)

    assert monitor.provider_health["mock"].consecutive_failures == 0
    assert monitor.provider_health["mock"].recent_latencies == [0.02]


def test_passive_outcomes_cost_no_provider_requests():
    """The point of the change: health data accrues without spending provider quota."""
    monitor = HealthMonitor()
    provider = CapturingMockProvider(should_fail=False, latency_seconds=0)
    monitor.register_provider(provider, health_check_model="mock-model")

    for _ in range(50):
        monitor.record_request_outcome("mock", succeeded=True, latency=0.01)

    assert monitor.get_status("mock") == "healthy"
    assert provider.requests == []
