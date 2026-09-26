import pytest

from app import rate_limiter
from tests.fakes import FakeRedis


@pytest.mark.asyncio
async def test_requests_under_limit_are_allowed(monkeypatch):
    redis_client = FakeRedis(decode_responses=True)
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 1000.0)

    allowed, remaining, limit_kind = await rate_limiter.check_rate_limit(
        team_id="team-alpha",
        requests_per_minute=2,
        redis_client=redis_client,
    )

    assert allowed is True
    assert remaining == 1
    assert limit_kind == "ok"


@pytest.mark.asyncio
async def test_requests_at_or_over_limit_are_rejected(monkeypatch):
    redis_client = FakeRedis(decode_responses=True)
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 1000.0)

    first_allowed, first_remaining, _kind = await rate_limiter.check_rate_limit(
        team_id="team-alpha",
        requests_per_minute=1,
        redis_client=redis_client,
    )
    second_allowed, second_remaining, second_kind = await rate_limiter.check_rate_limit(
        team_id="team-alpha",
        requests_per_minute=1,
        redis_client=redis_client,
    )

    assert first_allowed is True
    assert first_remaining == 0
    assert second_allowed is False
    assert second_remaining == 0
    assert second_kind == "requests"


@pytest.mark.asyncio
async def test_requests_are_allowed_after_window_passes(monkeypatch):
    redis_client = FakeRedis(decode_responses=True)
    current_time = 1000.0
    monkeypatch.setattr(rate_limiter.time, "time", lambda: current_time)

    allowed, remaining, _kind = await rate_limiter.check_rate_limit(
        team_id="team-alpha",
        requests_per_minute=1,
        redis_client=redis_client,
    )
    assert allowed is True
    assert remaining == 0

    current_time = 1061.0
    allowed, remaining, _kind = await rate_limiter.check_rate_limit(
        team_id="team-alpha",
        requests_per_minute=1,
        redis_client=redis_client,
    )

    assert allowed is True
    assert remaining == 0


# ---------------------------------------------------------------------------
# Tokens per minute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_token_limit_disables_token_limiting(monkeypatch):
    """Teams configured before this limit existed must behave exactly as before."""
    redis_client = FakeRedis(decode_responses=True)
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 1000.0)

    for _ in range(5):
        allowed, _remaining, _kind = await rate_limiter.check_rate_limit(
            team_id="team-alpha",
            requests_per_minute=100,
            redis_client=redis_client,
            tokens_per_minute=0,
            estimated_tokens=1_000_000,
        )
        assert allowed is True


@pytest.mark.asyncio
async def test_token_limit_rejects_once_the_window_would_be_exceeded(monkeypatch):
    redis_client = FakeRedis(decode_responses=True)
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 1000.0)

    kwargs = dict(
        team_id="team-alpha",
        requests_per_minute=100,
        redis_client=redis_client,
        tokens_per_minute=1000,
    )

    first, _r, _k = await rate_limiter.check_rate_limit(**kwargs, estimated_tokens=600)
    assert first is True

    # 600 + 600 exceeds 1000, so the second is refused on tokens even though the request
    # limit has plenty of headroom.
    second, _r, second_kind = await rate_limiter.check_rate_limit(
        **kwargs, estimated_tokens=600
    )
    assert second is False
    assert second_kind == "tokens"

    # A smaller request still fits in the remaining 400.
    third, _r, _k = await rate_limiter.check_rate_limit(**kwargs, estimated_tokens=300)
    assert third is True


@pytest.mark.asyncio
async def test_a_token_rejection_does_not_consume_a_request_slot(monkeypatch):
    """Both limits are evaluated in one script precisely so this cannot happen.

    Checking them in sequence would admit the request against the request limit, then
    reject it on tokens — spending a slot on a request that was never served.
    """
    redis_client = FakeRedis(decode_responses=True)
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 1000.0)

    rejected, _r, kind = await rate_limiter.check_rate_limit(
        team_id="team-alpha",
        requests_per_minute=5,
        redis_client=redis_client,
        tokens_per_minute=100,
        estimated_tokens=500,
    )
    assert rejected is False
    assert kind == "tokens"

    assert await redis_client.zcard("ratelimit:team-alpha") == 0


@pytest.mark.asyncio
async def test_recorded_usage_counts_against_the_window(monkeypatch):
    """Output tokens are unknown at admission, so they are charged afterwards."""
    redis_client = FakeRedis(decode_responses=True)
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 1000.0)

    kwargs = dict(
        team_id="team-alpha",
        requests_per_minute=100,
        redis_client=redis_client,
        tokens_per_minute=1000,
    )

    allowed, _r, _k = await rate_limiter.check_rate_limit(**kwargs, estimated_tokens=100)
    assert allowed is True

    # The completion turned out to be far longer than the prompt.
    await rate_limiter.record_token_usage("team-alpha", 850, redis_client)

    # 100 + 850 = 950 used, so a 100-token request no longer fits.
    allowed, _r, kind = await rate_limiter.check_rate_limit(**kwargs, estimated_tokens=100)
    assert allowed is False
    assert kind == "tokens"


@pytest.mark.asyncio
async def test_recorded_usage_can_be_negative_when_the_estimate_overshot(monkeypatch):
    redis_client = FakeRedis(decode_responses=True)
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 1000.0)

    kwargs = dict(
        team_id="team-alpha",
        requests_per_minute=100,
        redis_client=redis_client,
        tokens_per_minute=1000,
    )

    await rate_limiter.check_rate_limit(**kwargs, estimated_tokens=900)
    await rate_limiter.record_token_usage("team-alpha", -800, redis_client)

    # 900 - 800 = 100 used, so there is room again.
    allowed, _r, _k = await rate_limiter.check_rate_limit(**kwargs, estimated_tokens=800)
    assert allowed is True


@pytest.mark.asyncio
async def test_token_usage_ages_out_of_the_window(monkeypatch):
    redis_client = FakeRedis(decode_responses=True)
    current_time = 1000.0
    monkeypatch.setattr(rate_limiter.time, "time", lambda: current_time)

    kwargs = dict(
        team_id="team-alpha",
        requests_per_minute=100,
        redis_client=redis_client,
        tokens_per_minute=1000,
    )

    allowed, _r, _k = await rate_limiter.check_rate_limit(**kwargs, estimated_tokens=900)
    assert allowed is True

    blocked, _r, kind = await rate_limiter.check_rate_limit(**kwargs, estimated_tokens=900)
    assert blocked is False
    assert kind == "tokens"

    current_time = 1061.0
    allowed, _r, _k = await rate_limiter.check_rate_limit(**kwargs, estimated_tokens=900)
    assert allowed is True


@pytest.mark.asyncio
async def test_recording_zero_tokens_is_a_no_op(monkeypatch):
    redis_client = FakeRedis(decode_responses=True)
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 1000.0)

    await rate_limiter.record_token_usage("team-alpha", 0, redis_client)

    assert await redis_client.zcard("ratelimit:tokens:team-alpha") == 0
