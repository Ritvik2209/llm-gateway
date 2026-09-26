"""Shared Redis test double.

fakeredis provides real Redis data structures, but executes Lua only when `lupa` is
installed — it is not installed and is not a dependency, so `EVAL` raises
`unknown command 'eval'`. This subclass transcribes the rate limiter's script into Python
**against fakeredis's own sorted sets**, so a scripted call and an ordinary `zadd` operate
on the same data.

That detail matters: an emulation holding its own dictionary would diverge the moment the
two were mixed, and token accounting does mix them — admission goes through the script
while reconciliation appends with a plain `zadd`.
"""

import re

import fakeredis.aioredis


_TOKEN_PREFIX = re.compile(r"^(-?\d+)")


class FakeRedis(fakeredis.aioredis.FakeRedis):
    """fakeredis with a Python transcription of the rate limiter's Lua script."""

    async def eval(self, script, numkeys, *args):
        request_key, token_key = args[0], args[1]
        (
            now,
            window_start,
            request_limit,
            token_limit,
            tokens_requested,
            member,
            window_seconds,
        ) = args[2:9]

        now = float(now)
        window_start = float(window_start)
        request_limit = int(request_limit)
        token_limit = int(token_limit)
        tokens_requested = int(tokens_requested)
        window_seconds = int(window_seconds)

        await self.zremrangebyscore(request_key, "-inf", window_start)
        await self.zremrangebyscore(token_key, "-inf", window_start)

        request_count = int(await self.zcard(request_key))
        if request_count >= request_limit:
            return [0, "requests", 0]

        if token_limit > 0:
            token_sum = 0
            for entry in await self.zrange(token_key, 0, -1):
                if isinstance(entry, bytes):
                    entry = entry.decode()
                match = _TOKEN_PREFIX.match(entry)
                if match:
                    token_sum += int(match.group(1))

            if token_sum + tokens_requested > token_limit:
                return [0, "tokens", 0]

        await self.zadd(request_key, {member: now})
        await self.expire(request_key, window_seconds)

        if tokens_requested != 0:
            await self.zadd(token_key, {f"{tokens_requested}:{member}": now})
            await self.expire(token_key, window_seconds)

        return [1, "ok", request_limit - request_count - 1]
