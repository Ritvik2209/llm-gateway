import asyncio

import pytest

from app import budget
from app.config import MODEL_PRICING
from app.models.schemas import ChatMessage


class FakeRedisClient:
    def __init__(self):
        self.values = {}

    async def get(self, key):
        return self.values.get(key)

    async def incrbyfloat(self, key, amount):
        new_value = float(self.values.get(key, 0.0)) + amount
        self.values[key] = str(new_value)
        return new_value


def test_calculate_cost_uses_model_pricing(monkeypatch):
    pricing = {
        **MODEL_PRICING,
        "test-model": {
            "input_price_per_1k": 1.0,
            "output_price_per_1k": 2.0,
        },
    }
    monkeypatch.setattr(budget, "MODEL_PRICING", pricing)

    cost = budget.calculate_cost(
        model="test-model",
        input_tokens=500,
        output_tokens=1000,
    )

    assert cost == 2.5


@pytest.mark.asyncio
async def test_get_current_spend_returns_zero_when_no_spend_recorded():
    redis_client = FakeRedisClient()

    spend = await budget.get_current_spend("team-alpha", redis_client)

    assert spend == 0.0


@pytest.mark.asyncio
async def test_add_spend_accumulates_across_calls():
    redis_client = FakeRedisClient()

    first_total = await budget.add_spend("team-alpha", 1.25, redis_client)
    second_total = await budget.add_spend("team-alpha", 2.75, redis_client)

    assert first_total == 1.25
    assert second_total == 4.0
    assert await budget.get_current_spend("team-alpha", redis_client) == 4.0


@pytest.mark.asyncio
async def test_reserve_budget_rejects_when_the_reservation_would_exceed_the_cap():
    redis_client = FakeRedisClient()
    await budget.add_spend("team-alpha", 4.5, redis_client)

    allowed, spend_before, is_warning = await budget.reserve_budget(
        team_id="team-alpha",
        reservation_usd=1.0,
        monthly_budget_usd=5.0,
        redis_client=redis_client,
    )

    assert allowed is False
    assert spend_before == 4.5
    assert is_warning is True
    # The rejected reservation must be compensated, not left holding budget.
    assert await budget.get_current_spend("team-alpha", redis_client) == 4.5


@pytest.mark.asyncio
async def test_reserve_budget_allows_a_reservation_that_exactly_fills_the_cap():
    redis_client = FakeRedisClient()
    await budget.add_spend("team-alpha", 4.0, redis_client)

    allowed, _spend_before, is_warning = await budget.reserve_budget(
        team_id="team-alpha",
        reservation_usd=1.0,
        monthly_budget_usd=5.0,
        redis_client=redis_client,
    )

    assert allowed is True
    assert is_warning is True
    assert await budget.get_current_spend("team-alpha", redis_client) == 5.0


@pytest.mark.asyncio
async def test_reserve_budget_warns_once_the_reservation_crosses_80_percent():
    redis_client = FakeRedisClient()
    await budget.add_spend("team-alpha", 3.5, redis_client)

    allowed, spend_before, is_warning = await budget.reserve_budget(
        team_id="team-alpha",
        reservation_usd=0.5,
        monthly_budget_usd=5.0,
        redis_client=redis_client,
    )

    assert allowed is True
    assert spend_before == 3.5
    # 3.5 alone is under the 4.0 threshold; the reservation is what crosses it, so the
    # warning is raised on the request that causes it rather than the one after.
    assert is_warning is True


@pytest.mark.asyncio
async def test_reserve_budget_does_not_warn_below_the_threshold():
    redis_client = FakeRedisClient()

    _allowed, _spend_before, is_warning = await budget.reserve_budget(
        team_id="team-alpha",
        reservation_usd=1.0,
        monthly_budget_usd=5.0,
        redis_client=redis_client,
    )

    assert is_warning is False


@pytest.mark.asyncio
async def test_concurrent_reservations_cannot_exceed_the_cap():
    """The property the previous read-then-check implementation could not hold.

    A cap of 5.0 has room for exactly two reservations of 2.0. Firing five at once must
    admit two and reject three, and must leave recorded spend at or below the cap.
    """
    redis_client = FakeRedisClient()

    results = await asyncio.gather(
        *[
            budget.reserve_budget(
                team_id="team-alpha",
                reservation_usd=2.0,
                monthly_budget_usd=5.0,
                redis_client=redis_client,
            )
            for _ in range(5)
        ]
    )

    admitted = [allowed for allowed, _spend, _warning in results]
    assert admitted.count(True) == 2
    assert admitted.count(False) == 3

    final_spend = await budget.get_current_spend("team-alpha", redis_client)
    assert final_spend == 4.0
    assert final_spend <= 5.0


@pytest.mark.asyncio
async def test_reconcile_spend_replaces_a_reservation_with_the_actual_cost():
    redis_client = FakeRedisClient()
    await budget.reserve_budget(
        team_id="team-alpha",
        reservation_usd=1.0,
        monthly_budget_usd=5.0,
        redis_client=redis_client,
    )

    total = await budget.reconcile_spend(
        team_id="team-alpha",
        reserved_usd=1.0,
        actual_usd=0.25,
        redis_client=redis_client,
    )

    # The worst-case reservation is replaced by what the provider actually reported.
    assert total == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_release_reservation_returns_the_full_amount():
    redis_client = FakeRedisClient()
    await budget.add_spend("team-alpha", 1.0, redis_client)
    await budget.reserve_budget(
        team_id="team-alpha",
        reservation_usd=2.0,
        monthly_budget_usd=5.0,
        redis_client=redis_client,
    )

    total = await budget.release_reservation(
        team_id="team-alpha",
        reserved_usd=2.0,
        redis_client=redis_client,
    )

    # A request that never reached a provider must leave spend untouched.
    assert total == pytest.approx(1.0)


def test_estimate_max_cost_prices_output_at_the_max_tokens_ceiling():
    pricing = {
        **MODEL_PRICING,
        "test-model": {"input_price_per_1k": 1.0, "output_price_per_1k": 2.0},
    }
    budget.MODEL_PRICING = pricing
    try:
        messages = [ChatMessage(role="user", content="x" * 400)]

        cost = budget.estimate_max_cost(
            model="test-model",
            messages=messages,
            max_output_tokens=500,
        )
    finally:
        budget.MODEL_PRICING = MODEL_PRICING

    # 400 chars / 4 = 100 estimated input tokens at $1/1k, plus the 500-token output
    # ceiling at $2/1k. Reserving the ceiling is what makes the cap a hard limit.
    assert cost == pytest.approx(100 / 1000 * 1.0 + 500 / 1000 * 2.0)


def test_estimate_input_tokens_never_returns_zero():
    assert budget.estimate_input_tokens([ChatMessage(role="user", content="")]) == 1
