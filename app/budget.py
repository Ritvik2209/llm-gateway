"""Monthly budget tracking and pre-flight spend reservation for team usage."""

import math
from datetime import datetime, timezone
from typing import Iterable

from app.config import MODEL_PRICING


# Prompt tokens are estimated rather than counted: the gateway has no tokenizer for
# models it does not host, and a reservation only needs to be an upper bound on cost,
# not an exact figure. Reconciliation replaces the estimate with the provider's own
# reported usage once the call returns.
CHARS_PER_TOKEN_ESTIMATE = 4

WARNING_THRESHOLD_RATIO = 0.8

# Costs are floats, so a reservation that exactly fills a cap can land a fraction of a
# cent above it through binary rounding alone (0.0008 + 0.0002 > 0.001). This tolerance
# stops that from rejecting a legitimate request. The real fix is to hold money as
# integer micro-dollars rather than floats; that is a storage change, not a patch here.
BUDGET_EPSILON_USD = 1e-9


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


def estimate_input_tokens(messages: Iterable) -> int:
    """Approximate prompt tokens from message length."""
    total_characters = sum(len(message.content) for message in messages)
    return max(1, math.ceil(total_characters / CHARS_PER_TOKEN_ESTIMATE))


def estimate_max_cost(
    model: str,
    messages: Iterable,
    max_output_tokens: int | None,
) -> float:
    """Return an upper bound on what a request can cost.

    Output tokens are charged at the caller's ``max_tokens`` ceiling because that is the
    most the provider can bill for. Reserving the worst case is what makes the cap a
    hard limit: reserving a likely cost instead would let an unusually long completion
    carry a team past its budget.
    """
    return calculate_cost(
        model=model,
        input_tokens=estimate_input_tokens(messages),
        output_tokens=max(0, max_output_tokens or 0),
    )


async def get_current_spend(team_id: str, redis_client) -> float:
    """Read the team's current month spend from Redis."""
    spend = await redis_client.get(_budget_key(team_id))
    if spend is None:
        return 0.0
    return float(spend)


async def add_spend(team_id: str, amount: float, redis_client) -> float:
    """Atomically add spend to the team's current month total."""
    return float(await redis_client.incrbyfloat(_budget_key(team_id), amount))


async def reserve_budget(
    team_id: str,
    reservation_usd: float,
    monthly_budget_usd: float,
    redis_client,
) -> tuple[bool, float, bool]:
    """Atomically reserve a request's projected cost against the team's monthly cap.

    Returns ``(allowed, spend_before, is_warning)``.

    The reservation is taken *before* the provider call, so a request cannot be served
    unless its worst-case cost already fits inside the cap. That is what makes the cap
    a hard limit rather than a limit discovered one request late.

    ``INCRBYFLOAT`` is atomic and returns the post-increment total, so concurrent
    requests cannot both claim the same remaining headroom: each observes a distinct
    total, and only those landing at or below the cap proceed. A rejected request
    compensates by decrementing exactly what it added, so the stored value converges
    back — the compensating-transaction shape, applied to a counter.

    A concurrent *reader* can briefly observe a total above the cap, between a rejected
    request's increment and its compensation. That is visible to reporting but never to
    enforcement, since no request is admitted on the strength of that value. Collapsing
    the two round trips into one Lua script would remove even the transient, at the cost
    of moving the rule out of Python.
    """
    key = _budget_key(team_id)
    total_after_reservation = float(
        await redis_client.incrbyfloat(key, reservation_usd)
    )
    spend_before = total_after_reservation - reservation_usd

    if total_after_reservation > monthly_budget_usd + BUDGET_EPSILON_USD:
        await redis_client.incrbyfloat(key, -reservation_usd)
        return False, spend_before, True

    is_warning = (
        total_after_reservation >= WARNING_THRESHOLD_RATIO * monthly_budget_usd
    )
    return True, spend_before, is_warning


async def reconcile_spend(
    team_id: str,
    reserved_usd: float,
    actual_usd: float,
    redis_client,
) -> float:
    """Replace a reservation with the request's actual cost.

    The provider reports real token usage only after the call, so the reserved
    worst-case figure is adjusted by the difference. The delta is normally negative,
    since a completion rarely reaches the ``max_tokens`` ceiling the reservation
    assumed.
    """
    delta = actual_usd - reserved_usd
    if delta == 0:
        return await get_current_spend(team_id, redis_client)

    return float(await redis_client.incrbyfloat(_budget_key(team_id), delta))


async def release_reservation(
    team_id: str,
    reserved_usd: float,
    redis_client,
) -> float:
    """Return a reservation to the team's budget after a request fails.

    A request that never reached a provider, or failed at every provider, incurred no
    provider cost, so holding its reservation would leak budget on every failure.
    """
    return await reconcile_spend(
        team_id=team_id,
        reserved_usd=reserved_usd,
        actual_usd=0.0,
        redis_client=redis_client,
    )
