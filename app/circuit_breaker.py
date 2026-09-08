"""Circuit breaker state tracking for provider calls."""

from datetime import datetime, timedelta, timezone
from typing import Any

from app.metrics import set_circuit_breaker_state


class CircuitBreaker:
    """Per-provider circuit breaker with closed, open, and half-open states."""

    def __init__(
        self,
        failure_threshold: int = 4,
        cooldown_seconds: int = 30,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.provider_states: dict[str, dict[str, Any]] = {}

    def record_success(self, provider_name: str) -> None:
        state = self._get_provider_state(provider_name)
        previous_state = state["state"]
        state["state"] = "closed"
        state["failure_count"] = 0
        state["opened_at"] = None
        if previous_state != state["state"]:
            set_circuit_breaker_state(provider_name, state["state"])

    def record_failure(self, provider_name: str) -> None:
        state = self._get_provider_state(provider_name)
        previous_state = state["state"]
        state["failure_count"] += 1
        if state["failure_count"] >= self.failure_threshold:
            state["state"] = "open"
            state["opened_at"] = datetime.now(timezone.utc)
        if previous_state != state["state"]:
            set_circuit_breaker_state(provider_name, state["state"])

    def can_attempt(self, provider_name: str) -> bool:
        state = self._get_provider_state(provider_name)
        if state["state"] == "closed":
            return True

        if state["state"] == "open":
            opened_at = state["opened_at"]
            if opened_at is None:
                return False

            cooldown_elapsed = datetime.now(timezone.utc) - opened_at
            if cooldown_elapsed >= timedelta(seconds=self.cooldown_seconds):
                state["state"] = "half_open"
                set_circuit_breaker_state(provider_name, state["state"])
                return True
            return False

        if state["state"] == "half_open":
            return True

        return True

    def get_state(self, provider_name: str) -> str:
        return self._get_provider_state(provider_name)["state"]

    def _get_provider_state(self, provider_name: str) -> dict[str, Any]:
        return self.provider_states.setdefault(
            provider_name,
            {"state": "closed", "failure_count": 0, "opened_at": None},
        )
