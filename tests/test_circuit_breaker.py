from datetime import datetime, timedelta, timezone

from app.circuit_breaker import CircuitBreaker


def test_circuit_starts_closed_and_stays_closed_on_repeated_success():
    circuit_breaker = CircuitBreaker()

    assert circuit_breaker.get_state("mock") == "closed"

    circuit_breaker.record_success("mock")
    circuit_breaker.record_success("mock")

    assert circuit_breaker.get_state("mock") == "closed"


def test_circuit_opens_after_failure_threshold_failures():
    circuit_breaker = CircuitBreaker(failure_threshold=2)

    circuit_breaker.record_failure("mock")
    assert circuit_breaker.get_state("mock") == "closed"

    circuit_breaker.record_failure("mock")
    assert circuit_breaker.get_state("mock") == "open"


def test_can_attempt_returns_false_while_open_before_cooldown_passes():
    circuit_breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30)
    circuit_breaker.record_failure("mock")

    assert circuit_breaker.can_attempt("mock") is False


def test_can_attempt_transitions_open_to_half_open_after_cooldown():
    circuit_breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30)
    circuit_breaker.record_failure("mock")
    circuit_breaker.provider_states["mock"]["opened_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=31)
    )

    assert circuit_breaker.can_attempt("mock") is True
    assert circuit_breaker.get_state("mock") == "half_open"


def test_success_while_half_open_closes_circuit_again():
    circuit_breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30)
    circuit_breaker.record_failure("mock")
    circuit_breaker.provider_states["mock"]["opened_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=31)
    )
    circuit_breaker.can_attempt("mock")

    circuit_breaker.record_success("mock")

    assert circuit_breaker.get_state("mock") == "closed"
    assert circuit_breaker.provider_states["mock"]["failure_count"] == 0
