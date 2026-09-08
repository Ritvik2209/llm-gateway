"""Monthly budget tracking for team usage."""

from datetime import datetime, timezone

from app.config import MODEL_PRICING


def _current_year_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _budget_key(team_id: str) -> str:
    return f"budget:{team_id}:{_current_year_month()}"


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate request cost from model pricing and token usage."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        raise ValueError(f'No pricing configured for model "{model}".')

    input_cost = input_tokens / 1000 * pricing["input_price_per_1k"]
    output_cost = output_tokens / 1000 * pricing["output_price_per_1k"]
    return input_cost + output_cost


async def get_current_spend(team_id: str, redis_client) -> float:
    """Read the team's current month spend from Redis."""
    spend = await redis_client.get(_budget_key(team_id))
    if spend is None:
        return 0.0
    return float(spend)


async def add_spend(team_id: str, amount: float, redis_client) -> float:
    """Atomically add spend to the team's current month total."""
    return float(await redis_client.incrbyfloat(_budget_key(team_id), amount))


async def check_budget(
    team_id: str,
    monthly_budget_usd: float,
    redis_client,
) -> tuple[bool, float, bool]:
    """Return whether spend is below cap and whether warning threshold is met."""
    current_spend = await get_current_spend(team_id, redis_client)
    if current_spend >= monthly_budget_usd:
        return False, current_spend, True

    is_warning = current_spend >= 0.8 * monthly_budget_usd
    return True, current_spend, is_warning
