"""Redis-backed sliding window rate limiting."""

import os
import time
from uuid import uuid4

import redis.asyncio as redis


RATE_LIMIT_WINDOW_SECONDS = 60

RATE_LIMIT_LUA_SCRIPT = """
redis.call("ZREMRANGEBYSCORE", KEYS[1], "-inf", ARGV[2])
local count = tonumber(redis.call("ZCARD", KEYS[1]))
local limit = tonumber(ARGV[3])

if count < limit then
    redis.call("ZADD", KEYS[1], ARGV[1], ARGV[4])
    redis.call("EXPIRE", KEYS[1], ARGV[5])
    return {1, limit - count - 1}
end

return {0, 0}
"""


def get_redis_client() -> redis.Redis:
    """Create an async Redis client using environment-based configuration."""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    return redis.Redis(host=host, port=port, db=db, decode_responses=True)


async def check_rate_limit(
    team_id: str,
    requests_per_minute: int,
    redis_client,
) -> tuple[bool, int]:
    """Return whether a team is allowed and how much quota remains."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    key = f"ratelimit:{team_id}"
    member = f"{now}:{uuid4()}"

    allowed, remaining = await redis_client.eval(
        RATE_LIMIT_LUA_SCRIPT,
        1,
        key,
        now,
        window_start,
        requests_per_minute,
        member,
        RATE_LIMIT_WINDOW_SECONDS,
    )

    return bool(allowed), int(remaining)
