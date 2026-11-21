"""Routes for a public demo instance: the page, and the outage switch it drives.

Registered only when DEMO_MODE is on (see app/main.py). None of these routes take a key,
so none of them may do more than an anonymous visitor is allowed to: read which teams the
demo publishes, read the outage and circuit state, and start a time-limited outage on the
one provider wrapped for it. Limits, config and the admin API stay behind the admin key.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from app.circuit_breaker import CircuitBreaker
from app.providers.outage import OutageSwitch


DEMO_PAGE_PATH = Path(__file__).parent / "static" / "demo.html"


def seconds_until_half_open(
    circuit_breaker: CircuitBreaker, provider_name: str
) -> float | None:
    """How long until an open circuit lets one request through, or None if not open.

    Zero means the cooldown is over and the next request will be the trial; the breaker
    only moves to half_open when a request actually asks, not on a timer.
    """
    state = circuit_breaker.provider_states.get(provider_name)
    if not state or state["state"] != "open" or state["opened_at"] is None:
        return None
    elapsed = (datetime.now(timezone.utc) - state["opened_at"]).total_seconds()
    return round(max(0.0, circuit_breaker.cooldown_seconds - elapsed), 1)


def public_demo_teams(teams_config: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The teams whose keys the page may use, with what each one is configured to do.

    Read from the live config on every call, so a hot-reloaded change to a demo team shows
    up on the page without a restart. Admin teams are excluded here as well as refused at
    load time: publishing an admin key is the one mistake this page must never make.
    """
    teams = []
    for api_key, team in teams_config.items():
        if not team.get("public_demo") or team.get("is_admin"):
            continue
        allowed_providers = set(team.get("allowed_providers", []))
        teams.append(
            {
                "team_id": team["team_id"],
                "api_key": api_key,
                "allowed_models": team.get("allowed_models", []),
                "provider_chain": [
                    name
                    for name in team.get("provider_priority", [])
                    if name in allowed_providers
                ],
                "requests_per_minute": team.get("requests_per_minute"),
                "tokens_per_minute": team.get("tokens_per_minute"),
                "monthly_budget_usd": team.get("monthly_budget_usd"),
                "system_prompt": team.get("system_prompt"),
                "mandatory_system_prompt": team.get("mandatory_system_prompt"),
                "response_disclaimer": team.get("response_disclaimer"),
                "blocked_pattern_count": len(team.get("blocked_patterns") or []),
            }
        )
    return teams


def build_demo_router(
    outage_switch: OutageSwitch,
    outage_seconds: int,
    grafana_url: str = "",
) -> APIRouter:
    router = APIRouter()

    def status_payload(request: Request) -> dict[str, Any]:
        # The breaker is read from app state on each call rather than captured here, so it
        # is always the one the request path is using.
        circuit_breaker = request.app.state.circuit_breaker
        provider_name = outage_switch.provider_name
        return {
            "provider": provider_name,
            "outage_seconds": outage_seconds,
            "outage_remaining_seconds": round(
                outage_switch.outage_remaining_seconds(), 1
            ),
            "circuit_state": circuit_breaker.get_state(provider_name),
            "circuit_retry_in_seconds": seconds_until_half_open(
                circuit_breaker, provider_name
            ),
            "grafana_url": grafana_url or None,
        }

    @router.get("/", include_in_schema=False)
    async def demo_page() -> FileResponse:
        return FileResponse(DEMO_PAGE_PATH, media_type="text/html")

    @router.get("/demo/teams")
    async def demo_teams(request: Request) -> dict[str, list[dict[str, Any]]]:
        """The public demo teams and their configured limits and policies."""
        return {"teams": public_demo_teams(request.app.state.teams_config)}

    @router.get("/demo/status")
    async def demo_status(request: Request) -> dict[str, Any]:
        """Whether a simulated outage is running, and the provider's circuit state."""
        return status_payload(request)

    @router.post("/demo/outage")
    async def demo_outage(request: Request) -> dict[str, Any]:
        """Start a time-limited simulated outage. A running outage is left as it is."""
        started = outage_switch.start_outage(outage_seconds)
        return {"started": started, **status_payload(request)}

    return router
