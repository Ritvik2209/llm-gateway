import pytest

from app import rate_limiter


class FakeRedisClient:
    def __init__(self):
        self.sorted_sets = {}

    async def eval(self, script, numkeys, *args):
        key, now, window_start, limit, member, _window_seconds = args
        now = float(now)
        window_start = float(window_start)
        limit = int(limit)

        entries = [
            (score, existing_member)
            for score, existing_member in self.sorted_sets.get(key, [])
            if score > window_start
        ]
        self.sorted_sets[key] = entries

        count = len(entries)
        if count < limit:
            entries.append((now, member))
            return [1, limit - count - 1]

        return [0, 0]


@pytest.mark.asyncio
async def test_requests_under_limit_are_allowed(monkeypatch):
    redis_client = FakeRedisClient()
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 1000.0)

    allowed, remaining = await rate_limiter.check_rate_limit(
        team_id="team-alpha",
        requests_per_minute=2,
        redis_client=redis_client,
    )

    assert allowed is True
    assert remaining == 1


@pytest.mark.asyncio
async def test_requests_at_or_over_limit_are_rejected(monkeypatch):
    redis_client = FakeRedisClient()
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 1000.0)

    first_allowed, first_remaining = await rate_limiter.check_rate_limit(
        team_id="team-alpha",
        requests_per_minute=1,
        redis_client=redis_client,
    )
    second_allowed, second_remaining = await rate_limiter.check_rate_limit(
        team_id="team-alpha",
        requests_per_minute=1,
        redis_client=redis_client,
    )

    assert first_allowed is True
    assert first_remaining == 0
    assert second_allowed is False
    assert second_remaining == 0


@pytest.mark.asyncio
async def test_requests_are_allowed_after_window_passes(monkeypatch):
    redis_client = FakeRedisClient()
    current_time = 1000.0
    monkeypatch.setattr(rate_limiter.time, "time", lambda: current_time)

    allowed, remaining = await rate_limiter.check_rate_limit(
        team_id="team-alpha",
        requests_per_minute=1,
        redis_client=redis_client,
    )
    assert allowed is True
    assert remaining == 0

    current_time = 1061.0
    allowed, remaining = await rate_limiter.check_rate_limit(
        team_id="team-alpha",
        requests_per_minute=1,
        redis_client=redis_client,
    )

    assert allowed is True
    assert remaining == 0
