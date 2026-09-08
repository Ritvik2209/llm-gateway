"""Authentication dependencies for team API keys."""

from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Request, status


def get_team_from_api_key(
    api_key: str,
    teams_config: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return the team config for a valid API key."""
    team_config = teams_config.get(api_key)
    if team_config is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return team_config


def get_teams_config(request: Request) -> dict[str, dict[str, Any]]:
    """Read the app-level teams config loaded at startup."""
    return request.app.state.teams_config


def verify_api_key(
    authorization: Annotated[str | None, Header()] = None,
    teams_config: Annotated[dict[str, dict[str, Any]], Depends(get_teams_config)] = None,
) -> dict[str, Any]:
    """Authenticate a Bearer API key and return the matching team config."""
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
        )

    scheme, separator, api_key = authorization.partition(" ")
    if scheme != "Bearer" or not separator or not api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Malformed Authorization header. Expected "Bearer <api_key>".',
        )

    return get_team_from_api_key(api_key.strip(), teams_config or {})
