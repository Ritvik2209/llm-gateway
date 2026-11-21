"""Routes for a public demo instance: the page, and the outage switches it drives.

Registered only when DEMO_MODE is on (see app/main.py). None of these routes take a key,
so none of them may do more than an anonymous visitor is allowed to: read which teams the
demo publishes, read outage and circuit state, and start a time-limited outage on a
provider that a published team actually uses. Limits, config and the admin API stay behind
the admin key.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
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


def demo_provider_names(
    teams_config: dict[str, dict[str, Any]],
    outage_switches: dict[str, OutageSwitch],
) -> list[str]:
    """The providers a visitor may take down, in the order the demo chains first name them.

    Only providers that a published team can actually reach, so a demo server without
    Ollama shows no Ollama controls and an anonymous visitor cannot touch a provider no
    demo team uses. Mock is never wrapped in a switch: it is the last resort every demo
    chain ends on, and breaking it would turn fallback into a page of 503s.
    """
    names: list[str] = []
    for team in public_demo_teams(teams_config):
        for name in team["provider_chain"]:
            if name in outage_switches and name not in names:
                names.append(name)
    return names


def build_demo_router(
    outage_switches: dict[str, OutageSwitch],
    outage_seconds: int,
    grafana_url: str = "",
) -> APIRouter:
    router = APIRouter()

    def provider_status(request: Request, provider_name: str) -> dict[str, Any]:
        # The breaker is read from app state on each call rather than captured here, so it
        # is always the one the request path is using.
        circuit_breaker = request.app.state.circuit_breaker
        return {
            "name": provider_name,
            "outage_remaining_seconds": round(
                outage_switches[provider_name].outage_remaining_seconds(), 1
            ),
            "circuit_state": circuit_breaker.get_state(provider_name),
            "circuit_retry_in_seconds": seconds_until_half_open(
                circuit_breaker, provider_name
            ),
        }

    def status_payload(request: Request) -> dict[str, Any]:
        names = demo_provider_names(request.app.state.teams_config, outage_switches)
        return {
            "outage_seconds": outage_seconds,
            "grafana_url": grafana_url or None,
            "providers": [provider_status(request, name) for name in names],
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
        """Outage and circuit state for each provider the demo teams use."""
        return status_payload(request)

    @router.post("/demo/outage/{provider_name}")
    async def demo_outage(provider_name: str, request: Request) -> dict[str, Any]:
        """Start a time-limited simulated outage on one provider the demo teams use.

        A running outage is left as it is rather than extended.
        """
        if provider_name not in demo_provider_names(
            request.app.state.teams_config, outage_switches
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'"{provider_name}" is not a provider this demo can take down.',
            )
        started = outage_switches[provider_name].start_outage(outage_seconds)
        return {"started": started, "provider": provider_name, **status_payload(request)}

    return router
