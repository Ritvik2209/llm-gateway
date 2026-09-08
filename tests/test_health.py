import pytest

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
