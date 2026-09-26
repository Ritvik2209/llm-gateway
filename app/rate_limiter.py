"""Redis-backed sliding window rate limiting on requests and tokens."""

import os
import time
from uuid import uuid4

import redis.asyncio as redis


RATE_LIMIT_WINDOW_SECONDS = 60

# Both limits are evaluated in one script, and that is the point rather than an
# optimisation. Checking them separately means a request can consume a request slot, then
# be rejected on tokens — the slot is spent on a request that was never served, so the
# limiter leaks capacity on every partial admission. Admission across several dimensions
# has to be all-or-nothing.
#
# Tokens are held in their own sorted set, scored by timestamp like the request set, with
# the token count encoded in the member. Summing the window therefore means parsing the
# members, which is O(requests in window) — bounded by the request limit that was already
# checked above it.
RATE_LIMIT_LUA_SCRIPT = """
local now = ARGV[1]
local window_start = ARGV[2]
local request_limit = tonumber(ARGV[3])
local token_limit = tonumber(ARGV[4])
local tokens_requested = tonumber(ARGV[5])
local member = ARGV[6]
local window_seconds = ARGV[7]

redis.call("ZREMRANGEBYSCORE", KEYS[1], "-inf", window_start)
redis.call("ZREMRANGEBYSCORE", KEYS[2], "-inf", window_start)

local request_count = tonumber(redis.call("ZCARD", KEYS[1]))
if request_count >= request_limit then
    return {0, "requests", 0}
end

local token_sum = 0
if token_limit > 0 then
    local entries = redis.call("ZRANGE", KEYS[2], 0, -1)
    for i = 1, #entries do
        local encoded = string.match(entries[i], "^(-?%d+)")
        if encoded then
            token_sum = token_sum + tonumber(encoded)
        end
    end
    if token_sum + tokens_requested > token_limit then
        return {0, "tokens", 0}
    end
end

redis.call("ZADD", KEYS[1], now, member)
redis.call("EXPIRE", KEYS[1], window_seconds)

if tokens_requested ~= 0 then
    redis.call("ZADD", KEYS[2], now, tokens_requested .. ":" .. member)
    redis.call("EXPIRE", KEYS[2], window_seconds)
end

return {1, "ok", request_limit - request_count - 1}
"""


def get_redis_client() -> redis.Redis:
    """Create an async Redis client using environment-based configuration."""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    return redis.Redis(host=host, port=port, db=db, decode_responses=True)


def _request_key(team_id: str) -> str:
    return f"ratelimit:{team_id}"


def _token_key(team_id: str) -> str:
    return f"ratelimit:tokens:{team_id}"


async def check_rate_limit(
    team_id: str,
    requests_per_minute: int,
    redis_client,
    tokens_per_minute: int = 0,
    estimated_tokens: int = 0,
) -> tuple[bool, int, str]:
    """Admit or reject a request against the team's per-minute limits.

    Returns ``(allowed, remaining_requests, limit_kind)`` where ``limit_kind`` names the
    limit that rejected the request — ``"requests"`` or ``"tokens"`` — so the caller can
    say which one was hit rather than reporting a generic 429.

    ``tokens_per_minute`` of 0 disables token limiting, which keeps the behaviour of teams
    configured before the limit existed unchanged.

    Only the *prompt* tokens are charged here, because output length is unknown until the
    provider responds. ``record_token_usage`` adds the difference afterwards. The window
    can therefore be overshot by the output tokens of requests still in flight — unlike
    the monthly budget, where overshoot is permanent financial leakage and so is prevented
    by reserving the worst case. A per-minute window self-heals in 60 seconds, and
    reserving ``max_tokens`` instead would cap an 8,000 TPM team at roughly seven requests
    a minute regardless of how few tokens they actually used.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    member = f"{now}:{uuid4()}"

    allowed, limit_kind, remaining = await redis_client.eval(
        RATE_LIMIT_LUA_SCRIPT,
        2,
        _request_key(team_id),
        _token_key(team_id),
        now,
        window_start,
        requests_per_minute,
        tokens_per_minute,
        estimated_tokens,
        member,
        RATE_LIMIT_WINDOW_SECONDS,
    )

    if isinstance(limit_kind, bytes):
        limit_kind = limit_kind.decode()

    return bool(allowed), int(remaining), str(limit_kind)


async def record_token_usage(
    team_id: str,
    tokens: int,
    redis_client,
) -> None:
    """Charge tokens the admission check could not have known about.

    Appends rather than amending the admission entry, so the window is an append-only log
    of token deltas and no read-modify-write is needed. ``tokens`` is the difference
    between what actually got used and what was charged up front, and may be negative if
    the prompt estimate overshot.
    """
    if tokens == 0:
        return

    now = time.time()
    await redis_client.zadd(_token_key(team_id), {f"{tokens}:{uuid4()}": now})
    await redis_client.expire(_token_key(team_id), RATE_LIMIT_WINDOW_SECONDS)
