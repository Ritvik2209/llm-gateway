"""Request and response enrichment: policy the gateway applies on the caller's behalf.

The gateway is the one place every team's traffic passes through, which makes it the place
to enforce rules that would otherwise have to be reimplemented — and remembered — by every
service that calls a model. That is the whole argument for putting policy here rather than
in each caller.

Four policies are supported, and they divide along one axis that matters more than the
feature list: **whether the caller may override them**.

* ``system_prompt`` is a *default*. It is injected only when the caller supplied no system
  message of their own, so a caller who knows what they want gets what they asked for.
* ``mandatory_system_prompt`` is a *policy*. It is injected always, ahead of anything the
  caller sent, and no request shape removes it.
* ``response_disclaimer`` is a policy applied to the way out rather than the way in.
* ``blocked_patterns`` rejects the request before a provider is ever called.

That distinction is the point of this module. Before it, the team system prompt was a
default that read like a policy: any system message from the caller — not even a
conflicting one — silently dropped it, so a team's configured constraint could be removed
by a caller who did not know it existed. Both behaviours are legitimate; having only one of
them, and describing it as the other, was not.

**What "mandatory" does and does not mean.** It guarantees the instruction is present in
the request the provider receives: no caller-supplied message shape can remove it, and the
gateway does not ask the caller's permission. It does not guarantee the model obeys it —
prompt-level instruction is not a security boundary, and a determined caller can often
argue a model out of one. Treat it as policy plumbing that is reliable at the transport
layer and best-effort at the behavioural one. Content that must not reach a provider at all
belongs in ``blocked_patterns``, which is enforced by the gateway rather than requested of
the model.
"""

import re
from functools import lru_cache
from typing import Any

from app.models.schemas import ChatMessage, UnifiedChatRequest


@lru_cache(maxsize=256)
def _compiled(pattern: str) -> re.Pattern:
    """Compile and cache a blocked pattern.

    Patterns are validated when the config loads, so compilation here cannot fail on a
    config that was accepted. Caching keeps a per-request regex compile off the hot path
    without putting unpickleable compiled objects into the team config dict, which is
    serialised by the admin endpoints.
    """
    return re.compile(pattern, re.IGNORECASE)


def find_blocked_pattern(
    request: UnifiedChatRequest,
    team_config: dict[str, Any],
) -> str | None:
    """Return the first configured pattern the request matches, or ``None``.

    Every message is scanned, not only the user's: a system message is caller-supplied
    input like any other, and exempting it would leave the obvious way around the filter.

    This runs against the request as it arrived, before any enrichment, so the gateway
    never blocks a request on the strength of text it injected itself.
    """
    patterns = team_config.get("blocked_patterns") or []
    if not patterns:
        return None

    for message in request.messages:
        content = message.content or ""
        for pattern in patterns:
            if _compiled(pattern).search(content):
                return pattern

    return None


def enrich_request(
    request: UnifiedChatRequest,
    team_config: dict[str, Any],
) -> UnifiedChatRequest:
    """Apply the team's prompt policy to an outbound request.

    Ordering is deliberate. The mandatory prompt goes first, so it frames everything that
    follows and is not buried behind a long caller-supplied conversation. The default
    prompt, when it applies, goes after it: a team that configures both is saying "this
    always holds, and absent other instruction, so does this."
    """
    mandatory = team_config.get("mandatory_system_prompt")
    default = team_config.get("system_prompt")

    if not mandatory and not default:
        return request

    prefix: list[ChatMessage] = []

    if mandatory:
        prefix.append(ChatMessage(role="system", content=mandatory))

    if default:
        caller_set_their_own = any(
            message.role == "system" for message in request.messages
        )
        if not caller_set_their_own:
            prefix.append(ChatMessage(role="system", content=default))

    request.messages = [*prefix, *request.messages]
    return request


def apply_response_disclaimer(content: str, team_config: dict[str, Any]) -> str:
    """Append the team's disclaimer to a completed response body.

    Appended rather than prepended so the answer the caller asked for is the first thing
    they read, and separated by a blank line so it reads as an annotation rather than as
    part of the model's output.
    """
    disclaimer = team_config.get("response_disclaimer")
    if not disclaimer:
        return content

    return f"{content}\n\n{disclaimer}" if content else disclaimer


def disclaimer_chunk(team_config: dict[str, Any]) -> str | None:
    """The disclaimer as a final streamed chunk, or ``None`` when none is configured.

    A stream cannot be rewritten once sent, so the disclaimer is emitted after the last
    model chunk instead of being spliced into the body. A stream that fails partway
    therefore carries no disclaimer — correctly, since it also carries no complete answer
    to qualify.
    """
    disclaimer = team_config.get("response_disclaimer")
    if not disclaimer:
        return None

    return f"\n\n{disclaimer}"
