"""Simulated provider outages, for demo instances only."""

import time
from typing import Any, AsyncGenerator, Callable

from app.models.schemas import UnifiedChatRequest, UnifiedChatResponse
from app.providers.base import LLMProvider
from app.providers.errors import ProviderUnavailable


class OutageSwitch(LLMProvider):
    """Wraps a real provider so that an outage can be switched on for a fixed time.

    While an outage is running, every call fails with ``ProviderUnavailable`` without
    reaching the wrapped provider. That is the error a real outage produces, so retry,
    fallback, the circuit breaker and the health monitor react exactly as they would in
    production. Nothing downstream knows this class exists, which is the point: the demo
    exercises the real resilience path rather than a parallel one built for show.

    An outage always ends on its own. A switch that stayed on until someone turned it off
    would let one visitor leave the demo broken for everyone who came after.
    """

    def __init__(
        self,
        inner: LLMProvider,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.inner = inner
        # The wrapped provider's name, so routing, metrics and the circuit breaker still key
        # on "groq" rather than on a name of the wrapper's own.
        self.provider_name = inner.provider_name
        self._clock = clock
        self._outage_until = 0.0

    def start_outage(self, seconds: float) -> bool:
        """Begin an outage, or return False and change nothing if one is already running.

        A running outage is not extended, so no amount of pressing holds the provider down
        for longer than one outage at a time.
        """
        if self.outage_remaining_seconds() > 0:
            return False
        self._outage_until = self._clock() + seconds
        return True

    def outage_remaining_seconds(self) -> float:
        return max(0.0, self._outage_until - self._clock())

    def _fail_if_down(self) -> None:
        if self.outage_remaining_seconds() > 0:
            raise ProviderUnavailable(f"{self.provider_name} outage (simulated)")

    async def chat(self, request: UnifiedChatRequest) -> UnifiedChatResponse:
        self._fail_if_down()
        return await self.inner.chat(request)

    async def chat_stream(
        self, request: UnifiedChatRequest
    ) -> AsyncGenerator[str, None]:
        self._fail_if_down()
        async for chunk in self.inner.chat_stream(request):
            yield chunk

    def __getattr__(self, name: str) -> Any:
        # Anything else the request path reads off a provider, such as
        # last_stream_usage, comes from the real one. Guarded so that a lookup before
        # __init__ has set `inner` raises instead of recursing.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)
