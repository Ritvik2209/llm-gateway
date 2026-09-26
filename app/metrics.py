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

# Cost is a monotonic counter so that spend over any window — per day, per hour — is a
# query (`increase(gateway_cost_usd_total[1d])`) rather than a separate metric. Labelled
# by model as well as provider, because attributing spend to a model is what makes a
# cost-routing decision arguable rather than guessed.
COST_USD_TOTAL = Counter(
    "gateway_cost_usd_total",
    "Total gateway cost in USD.",
    ["team_id", "provider", "model"],
)

# The counter above resets when the process restarts, which is normal for Prometheus but
# useless for answering "how close is this team to its cap". These two gauges mirror the
# authoritative Redis state instead, so budget utilisation is spend / budget.
TEAM_SPEND_USD = Gauge(
    "gateway_team_spend_usd",
    "Team month-to-date spend in USD, as recorded in Redis.",
    ["team_id"],
)

TEAM_BUDGET_USD = Gauge(
    "gateway_team_budget_usd",
    "Team monthly budget cap in USD.",
    ["team_id"],
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
