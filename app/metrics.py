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

# The histogram above wraps the provider call only, so it cannot answer "how much latency
# does the gateway add" — during load testing it read 0.049s at P95 while clients saw
# 0.302s, the difference being queueing it cannot see. This one covers the whole handler,
# so gateway overhead is the difference between the two and the sub-10ms target becomes a
# query rather than something only an external load tool can measure.
#
# For a streaming request this measures time to the response being returned, which is time
# to first byte rather than time to completion.
REQUEST_TOTAL_DURATION_SECONDS = Histogram(
    "gateway_request_total_duration_seconds",
    "End-to-end gateway request latency in seconds, including all policy checks.",
    ["team_id", "status"],
)

# Overhead is computed per request and then aggregated, never as the difference between two
# aggregates. Subtracting percentiles is invalid — P95(end-to-end) minus P95(provider) is
# not the P95 of the difference, because the two percentiles describe different requests.
# Doing it that way produced a reading of -72ms, which is how the error announced itself.
#
# Requests rejected before any provider call contribute their whole duration here, which is
# correct: policy checks are gateway work.
GATEWAY_OVERHEAD_SECONDS = Histogram(
    "gateway_overhead_seconds",
    "Gateway time per request, excluding time spent inside provider calls.",
    ["team_id"],
    buckets=(
        0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
        float("inf"),
    ),
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

# Provider health existed only in the /admin/health JSON, so Prometheus — and therefore any
# dashboard or alert — could not see it. Encoded higher-is-worse like the circuit breaker
# gauge above, with -1 for a provider that has not been observed yet, so "not yet known" is
# distinguishable from "healthy" rather than both reading as zero.
PROVIDER_HEALTH_STATE = Gauge(
    "gateway_provider_health",
    "Provider health: -1 unknown, 0 healthy, 1 degraded, 2 down.",
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


PROVIDER_HEALTH_STATE_VALUES = {
    "unknown": -1,
    "healthy": 0,
    "degraded": 1,
    "down": 2,
}


def set_provider_health_state(provider_name: str, state: str) -> None:
    """Publish a provider's health so alerts and dashboards can read it."""
    PROVIDER_HEALTH_STATE.labels(provider_name=provider_name).set(
        PROVIDER_HEALTH_STATE_VALUES.get(state, -1)
    )


def set_circuit_breaker_state(provider_name: str, state: str) -> None:
    """Update the circuit breaker gauge for a provider."""
    CIRCUIT_BREAKER_STATE.labels(provider_name=provider_name).set(
        CIRCUIT_BREAKER_STATE_VALUES[state]
    )


# A rejected reload is the interesting event: it means the running config and the file on
# disk have diverged, and someone believes a change has taken effect when it has not.
CONFIG_RELOADS_TOTAL = Counter(
    "gateway_config_reloads_total",
    "Configuration reload attempts.",
    ["result"],
)

CONFIG_LOADED_TIMESTAMP = Gauge(
    "gateway_config_loaded_timestamp_seconds",
    "Unix timestamp of the configuration currently in effect.",
)
