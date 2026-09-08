"""Provider health monitoring infrastructure."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
            self._record_result(health, succeeded=False)
            health.consecutive_failures += 1
            if health.consecutive_failures >= 4:
                health.status = "down"
            elif health.consecutive_failures >= 2:
                health.status = "degraded"
            else:
                health.status = "healthy" if health.error_rate < 0.5 else "degraded"
        else:
            latency = perf_counter() - start_time
            health.recent_latencies.append(latency)
            health.recent_latencies = health.recent_latencies[-20:]
            self._record_result(health, succeeded=True)
            health.consecutive_failures = 0
            health.status = "healthy" if health.error_rate < 0.5 else "degraded"

        health.last_check_time = datetime.now(timezone.utc)
        return health

    async def start_background_checks(self, interval_seconds: int = 30) -> None:
        while True:
            checks = [
                asyncio.create_task(self.check_provider_health(provider_name))
                for provider_name in self.providers
            ]
            if checks:
                await asyncio.gather(*checks, return_exceptions=True)
            await asyncio.sleep(interval_seconds)

    def get_status(self, provider_name: str) -> str:
        health = self.provider_health.get(provider_name)
        if health is None:
            return "unknown"
        return health.status

    def _record_result(self, health: ProviderHealth, succeeded: bool) -> None:
        health.recent_results.append(succeeded)
        health.recent_results = health.recent_results[-20:]
        failures = health.recent_results.count(False)
        health.error_rate = failures / len(health.recent_results)
