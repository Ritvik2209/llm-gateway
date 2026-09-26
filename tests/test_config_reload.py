"""Configuration hot reload.

The property that matters is not that a reload works — it is that a *failed* reload is
harmless. A config file is edited by hand, so a malformed edit is a routine event, and it
must not be able to take the gateway down or leave it serving half a change.
"""

import pytest
from fastapi.testclient import TestClient

from app import config as config_module
from app import main as main_module


ADMIN_TEAMS = {
    "reload-admin-key": {
        "team_id": "reload-admin",
        "allowed_models": [],
        "allowed_providers": [],
        "provider_priority": [],
        "requests_per_minute": 60,
        "monthly_budget_usd": 0.0,
        "is_admin": True,
    },
}

VALID_TEAMS_YAML = """\
teams:
  - team_id: "reloaded-team"
    api_key: "reloaded-key"
    allowed_models: ["mock-model"]
    allowed_providers: ["mock"]
    provider_priority: ["mock"]
    requests_per_minute: 7
    tokens_per_minute: 700
    monthly_budget_usd: 2.50
"""

VALID_MODELS_YAML = """\
providers:
  mock:
    - "mock-model"
"""

# The exact defect this project started with: a list item indented to the depth of the
# mapping keys above it, which makes the whole file unparseable rather than just that entry.
MALFORMED_TEAMS_YAML = """\
teams:
  - team_id: "one"
    api_key: "k1"
    requests_per_minute: 5

    - team_id: "two"
    api_key: "k2"
    requests_per_minute: 5
"""


def write_config(tmp_path, teams_yaml=VALID_TEAMS_YAML, models_yaml=VALID_MODELS_YAML):
    teams_path = tmp_path / "teams.yaml"
    models_path = tmp_path / "models.yaml"
    teams_path.write_text(teams_yaml, encoding="utf-8")
    models_path.write_text(models_yaml, encoding="utf-8")
    return teams_path, models_path


def point_gateway_at(monkeypatch, teams_path, models_path):
    monkeypatch.setattr(main_module, "TEAMS_CONFIG_PATH", str(teams_path))
    monkeypatch.setattr(main_module, "MODEL_CATALOG_PATH", str(models_path))


@pytest.mark.asyncio
async def test_reload_applies_a_valid_edit(monkeypatch, tmp_path):
    teams_path, models_path = write_config(tmp_path)
    point_gateway_at(monkeypatch, teams_path, models_path)
    monkeypatch.setattr(main_module.app.state, "teams_config", dict(ADMIN_TEAMS))
    monkeypatch.setattr(main_module.app.state, "model_catalog", {})

    result = await main_module.reload_configuration()

    assert result["teams"] == 1
    assert "reloaded-key" in main_module.app.state.teams_config
    reloaded = main_module.app.state.teams_config["reloaded-key"]
    assert reloaded["requests_per_minute"] == 7
    assert reloaded["tokens_per_minute"] == 700
    assert main_module.app.state.model_catalog == {"mock": {"mock-model"}}


@pytest.mark.asyncio
async def test_a_malformed_edit_is_rejected_and_the_previous_config_survives(
    monkeypatch, tmp_path
):
    """A YAML typo must not become an outage."""
    teams_path, models_path = write_config(tmp_path, teams_yaml=MALFORMED_TEAMS_YAML)
    point_gateway_at(monkeypatch, teams_path, models_path)
    monkeypatch.setattr(main_module.app.state, "teams_config", dict(ADMIN_TEAMS))
    monkeypatch.setattr(main_module.app.state, "model_catalog", {"mock": {"mock-model"}})

    with pytest.raises(main_module.ConfigReloadError):
        await main_module.reload_configuration()

    # Untouched: the gateway is still serving the configuration it had.
    assert main_module.app.state.teams_config == ADMIN_TEAMS
    assert main_module.app.state.model_catalog == {"mock": {"mock-model"}}


@pytest.mark.asyncio
async def test_a_missing_file_is_rejected_rather_than_emptying_the_config(
    monkeypatch, tmp_path
):
    """Treating a missing file as 'no teams' would lock every caller out."""
    teams_path, models_path = write_config(tmp_path)
    teams_path.unlink()
    point_gateway_at(monkeypatch, teams_path, models_path)
    monkeypatch.setattr(main_module.app.state, "teams_config", dict(ADMIN_TEAMS))

    with pytest.raises(main_module.ConfigReloadError):
        await main_module.reload_configuration()

    assert main_module.app.state.teams_config == ADMIN_TEAMS


@pytest.mark.asyncio
async def test_config_is_replaced_not_mutated(monkeypatch, tmp_path):
    """A request in flight must read either the whole old config or the whole new one."""
    teams_path, models_path = write_config(tmp_path)
    point_gateway_at(monkeypatch, teams_path, models_path)
    original = dict(ADMIN_TEAMS)
    monkeypatch.setattr(main_module.app.state, "teams_config", original)
    monkeypatch.setattr(main_module.app.state, "model_catalog", {})

    await main_module.reload_configuration()

    # The old dictionary object is intact; the state now points at a different one.
    assert original == ADMIN_TEAMS
    assert main_module.app.state.teams_config is not original


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------


def test_reload_endpoint_requires_an_admin_key(monkeypatch):
    monkeypatch.setattr(main_module.app.state, "teams_config", dict(ADMIN_TEAMS))
    client = TestClient(main_module.app)

    assert client.post("/admin/config/reload").status_code == 401
    assert client.get("/admin/config").status_code == 401


def test_reload_endpoint_applies_an_edit(monkeypatch, tmp_path):
    teams_path, models_path = write_config(tmp_path)
    point_gateway_at(monkeypatch, teams_path, models_path)
    monkeypatch.setattr(main_module.app.state, "teams_config", dict(ADMIN_TEAMS))
    monkeypatch.setattr(main_module.app.state, "model_catalog", {})
    client = TestClient(main_module.app)

    response = client.post(
        "/admin/config/reload",
        headers={"Authorization": "Bearer reload-admin-key"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "applied"
    assert response.json()["teams"] == 1


def test_reload_endpoint_reports_a_rejected_edit_as_409(monkeypatch, tmp_path):
    """409, not 500: the request was fine, the state on disk is not applicable."""
    teams_path, models_path = write_config(tmp_path, teams_yaml=MALFORMED_TEAMS_YAML)
    point_gateway_at(monkeypatch, teams_path, models_path)
    monkeypatch.setattr(main_module.app.state, "teams_config", dict(ADMIN_TEAMS))
    client = TestClient(main_module.app)

    response = client.post(
        "/admin/config/reload",
        headers={"Authorization": "Bearer reload-admin-key"},
    )

    assert response.status_code == 409
    assert "still in effect" in response.json()["detail"]
    assert main_module.app.state.teams_config == ADMIN_TEAMS


def test_config_endpoint_reports_what_is_in_effect(monkeypatch):
    monkeypatch.setattr(main_module.app.state, "teams_config", dict(ADMIN_TEAMS))
    monkeypatch.setattr(main_module.app.state, "model_catalog", {"mock": {"mock-model"}})
    client = TestClient(main_module.app)

    response = client.get(
        "/admin/config",
        headers={"Authorization": "Bearer reload-admin-key"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["teams"] == ["reload-admin"]
    assert body["model_catalog"] == {"mock": ["mock-model"]}
    assert body["loaded_at"]


def test_reload_interval_can_be_disabled():
    """0 leaves the explicit endpoint as the only way to apply an edit."""
    assert config_module.CONFIG_RELOAD_INTERVAL_SECONDS >= 0
