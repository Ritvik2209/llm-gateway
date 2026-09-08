import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.auth import get_team_from_api_key, verify_api_key


TEAMS_CONFIG = {
    "demo-team-alpha-local-only": {
        "team_id": "team-alpha",
        "allowed_models": ["llama3.2"],
        "allowed_providers": ["ollama"],
    }
}


def create_test_client() -> TestClient:
    app = FastAPI()
    app.state.teams_config = TEAMS_CONFIG

    @app.get("/protected")
    async def protected(team_config: dict = Depends(verify_api_key)) -> dict:
        return team_config

    return TestClient(app)


def test_valid_api_key_returns_correct_team_config():
    team_config = get_team_from_api_key("demo-team-alpha-local-only", TEAMS_CONFIG)

    assert team_config == {
        "team_id": "team-alpha",
        "allowed_models": ["llama3.2"],
        "allowed_providers": ["ollama"],
    }


def test_invalid_api_key_raises_401():
    with pytest.raises(HTTPException) as exc_info:
        get_team_from_api_key("invalid-key", TEAMS_CONFIG)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid API key."


def test_missing_authorization_header_raises_401():
    client = create_test_client()

    response = client.get("/protected")

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing Authorization header."


def test_malformed_authorization_header_raises_401():
    client = create_test_client()

    response = client.get(
        "/protected",
        headers={"Authorization": "Token demo-team-alpha-local-only"},
    )

    assert response.status_code == 401
    assert (
        response.json()["detail"]
        == 'Malformed Authorization header. Expected "Bearer <api_key>".'
    )
