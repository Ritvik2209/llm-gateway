"""Provider health monitoring infrastructure."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from time import perf_counter

from app.metrics import set_provider_health_state
from app.models.schemas import ChatMessage, UnifiedChatRequest
from app.providers.base import LLMProvider


logger = logging.getLogger(__name__)

# Worst-first, so aggregating a provider's models is a search down this list.
_STATUS_SEVERITY = ("down", "degraded", "healthy")


@dataclass
class ProviderHealth:
    status: str = "unknown"
    last_check_time: datetime | None = None
    consecutive_failures: int = 0
    recent_latencies: list[float] = field(default_factory=list)
    error_rate: float = 0.0
    recent_results: list[bool] = field(default_factory=list)


class HealthMonitor:
    """Tracks health for each provider-model pair the gateway has actually used.

    Health is keyed on the pair rather than on the provider because one provider serves
    several models and can fail for one while serving another perfectly well — a quota
    exhausted on a single model is the ordinary case, not a corner case. Keying on the
    provider alone conflates them, so one model's failures mark the whole provider down
    and divert traffic that would have succeeded.

    That conflation only became reachable when logical model tiers were introduced: before
    them a provider served one model here, so the pair and the provider were the same
    thing. This is the granularity the tier work made necessary.
    """

    def __init__(self, health_check_timeout_seconds: float = 5.0) -> None:
        self.provider_health: dict[tuple[str, str], ProviderHealth] = {}
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
        # Publish "unknown" immediately, so a pair that has never been observed shows as
        # -1 rather than as an absent series a dashboard would render as a gap.
        set_provider_health_state(provider.provider_name, health_check_model, "unknown")

    def models_for(self, provider_name: str) -> list[str]:
        """Models this provider has been observed serving, probed or real."""
        return [
            model
            for (candidate, model) in self.provider_health
            if candidate == provider_name
        ]

    async def check_provider_health(self, provider_name: str) -> ProviderHealth:
        """Probe one representative model for this provider.

        Deliberately not every model the provider serves. Probing is the expensive path,
        and the cost of covering every pair synthetically scales with the catalog while
        buying nothing for pairs that real traffic already reports on for free.
        """
        provider = self.providers[provider_name]
        health_check_model = self.health_check_models[provider_name]
        health = self.provider_health.setdefault(
            (provider_name, health_check_model), ProviderHealth()
        )
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
            logger.warning(
                "Health check failed for %s/%s: %s",
                provider_name,
                health_check_model,
                exc,
            )
            self._apply_outcome(
                health,
                succeeded=False,
                provider_name=provider_name,
                model=health_check_model,
            )
        else:
            self._apply_outcome(
                health,
                succeeded=True,
                latency=perf_counter() - start_time,
                provider_name=provider_name,
                model=health_check_model,
            )

        return health

    def record_request_outcome(
        self,
        provider_name: str,
        model: str,
        succeeded: bool,
        latency: float | None = None,
    ) -> None:
        """Fold a real request's outcome into the health of that provider-model pair.

        Passive health checking. A real request is a strictly better health signal than a
        synthetic probe — it reflects the traffic that actually matters, and it costs no
        provider quota — so the gateway learns health as a side effect of serving requests
        rather than by spending requests to ask. It is also the only way the non-probed
        pairs are covered at all.
        """
        health = self.provider_health.setdefault(
            (provider_name, model), ProviderHealth()
        )
        self._apply_outcome(
            health,
            succeeded=succeeded,
            latency=latency,
            provider_name=provider_name,
            model=model,
        )

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

        * no model of this provider has been observed, so there is no health data at all;
          or
        * the circuit is not closed, so we need to know when it recovers and real traffic
          is being withheld from it.

        A provider that is closed and serving traffic already reports its own health for
        free through ``record_request_outcome``.

        The question is asked across every model of the provider, not just the one that
        would be probed. Traffic on any model is evidence the provider is reachable, so
        keying this on the probe pair alone would spend quota probing a provider we
        already have a signal for - which is exactly the quota exhaustion the passive
        path exists to prevent.
        """
        if self.get_status(provider_name) == "unknown":
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

    def get_status(self, provider_name: str, model: str | None = None) -> str:
        """Health of one provider-model pair, or of the provider as a whole.

        The aggregate is not simply the worst pair. A provider counts as "down" only when
        every model it serves is down; one failing model among several working ones makes
        it degraded, because calling it down would misreport a provider that is still
        serving most of its traffic correctly.
        """
        if model is not None:
            health = self.provider_health.get((provider_name, model))
            return health.status if health is not None else "unknown"

        statuses = [
            health.status
            for (candidate, _), health in self.provider_health.items()
            if candidate == provider_name
        ]
        known = [status for status in statuses if status != "unknown"]
        if not known:
            return "unknown"
        if all(status == "down" for status in known):
            return "down"
        if any(status in ("down", "degraded") for status in known):
            return "degraded"
        return "healthy"

    def _apply_outcome(
        self,
        health: ProviderHealth,
        succeeded: bool,
        latency: float | None = None,
        provider_name: str | None = None,
        model: str | None = None,
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

        if provider_name is not None and model is not None:
            set_provider_health_state(provider_name, model, health.status)

    def _record_result(self, health: ProviderHealth, succeeded: bool) -> None:
        health.recent_results.append(succeeded)
        health.recent_results = health.recent_results[-20:]
        failures = health.recent_results.count(False)
        health.error_rate = failures / len(health.recent_results)
