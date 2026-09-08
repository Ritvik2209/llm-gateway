"""Prometheus metrics for the gateway."""

from prometheus_client import Counter, Gauge, Histogram


REQUESTS_TOTAL = Counter(
    "gateway_requests_total",
    "Total gateway chat requests.",
    ["team_id", "model", "provider", "status"],
)

REQUEST_DURATION_SECONDS = Histogram(
    "gateway_request_duration_seconds",
    "Provider request latency in seconds.",
    ["team_id", "provider"],
)

ERRORS_TOTAL = Counter(
    "gateway_errors_total",
    "Total gateway errors.",
    ["team_id", "provider", "error_type"],
)

FALLBACK_TRIGGERED_TOTAL = Counter(
    "gateway_fallback_triggered_total",
    "Total fallback routing switches away from first-priority provider.",
    ["team_id", "from_provider", "to_provider"],
)

CIRCUIT_BREAKER_STATE = Gauge(
    "gateway_circuit_breaker_state",
    "Circuit breaker state: 0 closed, 1 half_open, 2 open.",
    ["provider_name"],
)

TOKENS_TOTAL = Counter(
    "gateway_tokens_total",
    "Total gateway tokens.",
    ["team_id", "provider", "token_type"],
)


CIRCUIT_BREAKER_STATE_VALUES = {
    "closed": 0,
    "half_open": 1,
    "open": 2,
}


def set_circuit_breaker_state(provider_name: str, state: str) -> None:
    """Update the circuit breaker gauge for a provider."""
    CIRCUIT_BREAKER_STATE.labels(provider_name=provider_name).set(
        CIRCUIT_BREAKER_STATE_VALUES[state]
    )
