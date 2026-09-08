import pytest

from app import budget
from app.config import MODEL_PRICING


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
async def test_check_budget_rejects_when_spend_meets_budget():
    redis_client = FakeRedisClient()
    await budget.add_spend("team-alpha", 5.0, redis_client)

    allowed, current_spend, is_warning = await budget.check_budget(
        team_id="team-alpha",
        monthly_budget_usd=5.0,
        redis_client=redis_client,
    )

    assert allowed is False
    assert current_spend == 5.0
    assert is_warning is True


@pytest.mark.asyncio
async def test_check_budget_warns_at_80_percent_threshold():
    redis_client = FakeRedisClient()
    await budget.add_spend("team-alpha", 4.0, redis_client)

    allowed, current_spend, is_warning = await budget.check_budget(
        team_id="team-alpha",
        monthly_budget_usd=5.0,
        redis_client=redis_client,
    )

    assert allowed is True
    assert current_spend == 4.0
    assert is_warning is True
