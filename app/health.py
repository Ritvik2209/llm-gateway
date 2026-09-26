"""Provider health monitoring infrastructure."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from time import perf_counter

from app.models.schemas import ChatMessage, UnifiedChatRequest
from app.providers.base import LLMProvider


logger = logging.getLogger(__name__)


@dataclass
class ProviderHealth:
    status: str = "unknown"
    last_check_time: datetime | None = None
    consecutive_failures: int = 0
    recent_latencies: list[float] = field(default_factory=list)
    error_rate: float = 0.0
    recent_results: list[bool] = field(default_factory=list)


class HealthMonitor:
    """Tracks provider health through fast, periodic chat probes."""

    def __init__(self, health_check_timeout_seconds: float = 5.0) -> None:
        self.provider_health: dict[str, ProviderHealth] = {}
        self.providers: dict[str, LLMProvider] = {}
        self.health_check_models: dict[str, str] = {}
        self.health_check_timeout_seconds = health_check_timeout_seconds

    def register_provider(
        self,
        provider: LLMProvider,
        health_check_model: str,
    ) -> None:
        self.providers[provider.provider_name] = provider
        self.health_check_models[provider.provider_name] = health_check_model

    async def check_provider_health(self, provider_name: str) -> ProviderHealth:
        provider = self.providers[provider_name]
        health_check_model = self.health_check_models[provider_name]
        health = self.provider_health.setdefault(provider_name, ProviderHealth())
        request = UnifiedChatRequest(
            model=health_check_model,
            messages=[ChatMessage(role="user", content="ping")],
        )

        start_time = perf_counter()
        try:
            await asyncio.wait_for(
                provider.chat(request),
                timeout=self.health_check_timeout_seconds,
            )
        except Exception as exc:
            logger.warning("Health check failed for %s: %s", provider_name, exc)
            self._apply_outcome(health, succeeded=False)
        else:
            self._apply_outcome(
                health,
                succeeded=True,
                latency=perf_counter() - start_time,
            )

        return health

    def record_request_outcome(
        self,
        provider_name: str,
        succeeded: bool,
        latency: float | None = None,
    ) -> None:
        """Fold a real request's outcome into provider health.

        Passive health checking. A real request is a strictly better health signal than a
        synthetic probe — it reflects the traffic that actually matters, and it costs no
        provider quota — so the gateway learns provider health as a side effect of serving
        requests rather than by spending requests to ask.
        """
        health = self.provider_health.setdefault(provider_name, ProviderHealth())
        self._apply_outcome(health, succeeded=succeeded, latency=latency)

    def should_probe(
        self,
        provider_name: str,
        circuit_breaker: Any = None,
    ) -> bool:
        """Decide whether a provider needs a synthetic probe.

        Probing is the expensive path: every probe spends real provider quota, and on a
        metered free tier a fixed-interval probe loop will exhaust the daily allowance on
        its own — a monitor that consumes the capacity it exists to protect. So a probe is
        only worth spending when there is no cheaper signal available:

        * the provider has never been seen, so there is no health data at all; or
        * its circuit is not closed, so we need to know when it recovers and real traffic
          is being withheld from it.

        A provider that is closed and serving traffic already reports its own health for
        free through ``record_request_outcome``.
        """
        health = self.provider_health.get(provider_name)
        if health is None or health.status == "unknown":
            return True

        if circuit_breaker is not None:
            return circuit_breaker.get_state(provider_name) != "closed"

        return False

    async def start_background_checks(
        self,
        interval_seconds: int = 30,
        circuit_breaker: Any = None,
    ) -> None:
        """Probe only the providers that need probing, on a fixed interval.

        Passing ``circuit_breaker`` enables the quota-aware policy in ``should_probe``.
        Omitting it falls back to probing every provider every interval, which is the
        original behaviour and is only appropriate against providers with no meaningful
        request quota.
        """
        while True:
            due = [
                provider_name
                for provider_name in self.providers
                if self.should_probe(provider_name, circuit_breaker)
            ]
            checks = [
                asyncio.create_task(self.check_provider_health(provider_name))
                for provider_name in due
            ]
            if checks:
                await asyncio.gather(*checks, return_exceptions=True)
            await asyncio.sleep(interval_seconds)

    def get_status(self, provider_name: str) -> str:
        health = self.provider_health.get(provider_name)
        if health is None:
            return "unknown"
        return health.status

    def _apply_outcome(
        self,
        health: ProviderHealth,
        succeeded: bool,
        latency: float | None = None,
    ) -> None:
        """Apply one observation, from a probe or from real traffic, identically."""
        self._record_result(health, succeeded=succeeded)

        if succeeded:
            health.consecutive_failures = 0
            if latency is not None:
                health.recent_latencies.append(latency)
                health.recent_latencies = health.recent_latencies[-20:]
        else:
            health.consecutive_failures += 1

        if health.consecutive_failures >= 4:
            health.status = "down"
        elif health.consecutive_failures >= 2:
            health.status = "degraded"
        else:
            health.status = "healthy" if health.error_rate < 0.5 else "degraded"

        health.last_check_time = datetime.now(timezone.utc)

    def _record_result(self, health: ProviderHealth, succeeded: bool) -> None:
        health.recent_results.append(succeeded)
        health.recent_results = health.recent_results[-20:]
        failures = health.recent_results.count(False)
        health.error_rate = failures / len(health.recent_results)
