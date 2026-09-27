"""Admin API: viewing limits, live usage, runtime adjustment, and the audit trail.

Runtime adjustment writes back to the team config file rather than holding overrides in a
second store, so the file stays the single source of truth and the running policy cannot
silently diverge from the declared one. These tests pin that, and pin the boundaries of
what the endpoint is allowed to change.
"""

import pytest
from fastapi.testclient import TestClient

from app import audit as audit_module
from app import main as main_module
from app.config_writer import (
    ConfigWriteError,
    TeamNotFoundError,
    read_team_limits,
    update_team_limits,
)
from tests.fakes import FakeRedis


TEAMS_YAML = """\
# A comment that must survive a runtime edit.
teams:
  - team_id: "admin-api-tenant"
    api_key: "tenant-key"
    allowed_models: ["mock-model"]
    allowed_providers: ["mock"]
    provider_priority: ["mock"]
    system_prompt: null
    # Why this team is throttled, explained for whoever reads the config next.
    requests_per_minute: 10
    tokens_per_minute: 1000
    monthly_budget_usd: 5.00

  - team_id: "admin-api-admin"
    api_key: "admin-key"
    allowed_models: []
    allowed_providers: []
    provider_priority: []
    system_prompt: null
    requests_per_minute: 60
    tokens_per_minute: 0
    monthly_budget_usd: 0.00
    is_admin: true
"""

MODELS_YAML = """\
providers:
  mock:
    - "mock-model"
"""


def build_environment(monkeypatch, tmp_path):
    """Point the gateway at a throwaway config and an in-memory Redis."""
    teams_path = tmp_path / "teams.yaml"
    models_path = tmp_path / "models.yaml"
    teams_path.write_text(TEAMS_YAML, encoding="utf-8")
    models_path.write_text(MODELS_YAML, encoding="utf-8")

    monkeypatch.setattr(main_module, "TEAMS_CONFIG_PATH", str(teams_path))
    monkeypatch.setattr(main_module, "MODEL_CATALOG_PATH", str(models_path))

    redis_client = FakeRedis(decode_responses=True)
    monkeypatch.setattr(main_module.app.state, "redis_client", redis_client)
    monkeypatch.setattr(
        main_module.app.state,
        "teams_config",
        main_module.load_teams_config(str(teams_path)),
    )
    monkeypatch.setattr(main_module.app.state, "model_catalog", {"mock": {"mock-model"}})

    return TestClient(main_module.app), redis_client, teams_path


ADMIN = {"Authorization": "Bearer admin-key"}
TENANT = {"Authorization": "Bearer tenant-key"}


# ---------------------------------------------------------------------------
# config_writer
# ---------------------------------------------------------------------------


def test_editing_a_limit_preserves_comments(tmp_path):
    """pyyaml would reserialise the document and drop every comment."""
    path = tmp_path / "teams.yaml"
    path.write_text(TEAMS_YAML, encoding="utf-8")

    update_team_limits("admin-api-tenant", {"requests_per_minute": 25}, str(path))

    written = path.read_text(encoding="utf-8")
    assert "# A comment that must survive a runtime edit." in written
    assert "# Why this team is throttled" in written
    assert "requests_per_minute: 25" in written


def test_only_the_named_field_changes(tmp_path):
    path = tmp_path / "teams.yaml"
    path.write_text(TEAMS_YAML, encoding="utf-8")

    result = update_team_limits(
        "admin-api-tenant", {"monthly_budget_usd": 9.5}, str(path)
    )

    assert result["before"] == {"monthly_budget_usd": 5.0}
    assert result["after"] == {"monthly_budget_usd": 9.5}

    limits = read_team_limits(str(path))
    assert limits["admin-api-tenant"]["monthly_budget_usd"] == 9.5
    # Untouched.
    assert limits["admin-api-tenant"]["requests_per_minute"] == 10
    assert limits["admin-api-tenant"]["tokens_per_minute"] == 1000


def test_a_no_op_edit_does_not_rewrite_the_file(tmp_path):
    """Writing an identical value would produce a pointless audit entry and reload."""
    path = tmp_path / "teams.yaml"
    path.write_text(TEAMS_YAML, encoding="utf-8")
    before_bytes = path.read_bytes()

    result = update_team_limits(
        "admin-api-tenant", {"requests_per_minute": 10}, str(path)
    )

    assert result == {"before": {}, "after": {}}
    assert path.read_bytes() == before_bytes


def test_an_unknown_team_is_rejected_before_the_file_is_touched(tmp_path):
    path = tmp_path / "teams.yaml"
    path.write_text(TEAMS_YAML, encoding="utf-8")
    before_bytes = path.read_bytes()

    with pytest.raises(TeamNotFoundError):
        update_team_limits("no-such-team", {"requests_per_minute": 1}, str(path))

    assert path.read_bytes() == before_bytes


def test_a_non_editable_field_is_rejected_before_the_file_is_touched(tmp_path):
    """The privilege boundary: is_admin must not be reachable from here."""
    path = tmp_path / "teams.yaml"
    path.write_text(TEAMS_YAML, encoding="utf-8")
    before_bytes = path.read_bytes()

    with pytest.raises(ConfigWriteError):
        update_team_limits("admin-api-tenant", {"is_admin": True}, str(path))

    assert path.read_bytes() == before_bytes


def test_an_edit_changes_exactly_one_line(tmp_path):
    """A writable config file is only useful if its diffs are reviewable.

    ruamel will fold long scalars and rewrite ``null`` as an empty value unless told not
    to, which would touch unrelated lines on every edit and bury the one that changed.
    """
    path = tmp_path / "teams.yaml"
    path.write_text(TEAMS_YAML, encoding="utf-8")
    before_lines = path.read_text(encoding="utf-8").splitlines()

    update_team_limits("admin-api-tenant", {"requests_per_minute": 25}, str(path))

    after_lines = path.read_text(encoding="utf-8").splitlines()
    assert len(before_lines) == len(after_lines)

    differing = [
        (before, after)
        for before, after in zip(before_lines, after_lines)
        if before != after
    ]
    assert differing == [
        ("    requests_per_minute: 10", "    requests_per_minute: 25")
    ]


def test_null_values_stay_written_as_null(tmp_path):
    path = tmp_path / "teams.yaml"
    path.write_text(TEAMS_YAML, encoding="utf-8")

    update_team_limits("admin-api-tenant", {"requests_per_minute": 25}, str(path))

    assert "system_prompt: null" in path.read_text(encoding="utf-8")


def test_no_temporary_files_are_left_behind(tmp_path):
    path = tmp_path / "teams.yaml"
    path.write_text(TEAMS_YAML, encoding="utf-8")

    update_team_limits("admin-api-tenant", {"requests_per_minute": 11}, str(path))

    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["teams.yaml"]


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, path",
    [
        ("get", "/admin/teams"),
        ("get", "/admin/teams/admin-api-tenant/usage"),
        ("patch", "/admin/teams/admin-api-tenant"),
        ("get", "/admin/audit"),
    ],
)
def test_admin_endpoints_reject_a_tenant_key(monkeypatch, tmp_path, method, path):
    client, _redis, _teams_path = build_environment(monkeypatch, tmp_path)

    kwargs = {"headers": TENANT}
    if method == "patch":
        kwargs["json"] = {"requests_per_minute": 1}
    response = getattr(client, method)(path, **kwargs)

    assert response.status_code == 403


def test_patch_rejects_privilege_escalation(monkeypatch, tmp_path):
    """extra="forbid" turns an unknown field into a 422 rather than a silent ignore."""
    client, _redis, teams_path = build_environment(monkeypatch, tmp_path)
    before_bytes = teams_path.read_bytes()

    response = client.patch(
        "/admin/teams/admin-api-tenant",
        headers=ADMIN,
        json={"is_admin": True},
    )

    assert response.status_code == 422
    # The file is untouched, so the tenant gained nothing. Comparing bytes rather than
    # grepping for "is_admin" matters: the admin team legitimately carries that flag.
    assert teams_path.read_bytes() == before_bytes


def test_patch_rejects_negative_limits(monkeypatch, tmp_path):
    client, _redis, _teams_path = build_environment(monkeypatch, tmp_path)

    response = client.patch(
        "/admin/teams/admin-api-tenant",
        headers=ADMIN,
        json={"requests_per_minute": -5},
    )

    assert response.status_code == 422


def test_patch_requires_at_least_one_field(monkeypatch, tmp_path):
    client, _redis, _teams_path = build_environment(monkeypatch, tmp_path)

    response = client.patch("/admin/teams/admin-api-tenant", headers=ADMIN, json={})

    assert response.status_code == 400


def test_patch_on_an_unknown_team_is_404(monkeypatch, tmp_path):
    client, _redis, _teams_path = build_environment(monkeypatch, tmp_path)

    response = client.patch(
        "/admin/teams/nobody",
        headers=ADMIN,
        json={"requests_per_minute": 5},
    )

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_patch_applies_the_change_to_live_traffic(monkeypatch, tmp_path):
    """The point of the endpoint: a new limit takes effect without a restart."""
    client, _redis, teams_path = build_environment(monkeypatch, tmp_path)

    response = client.patch(
        "/admin/teams/admin-api-tenant",
        headers=ADMIN,
        json={"requests_per_minute": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "applied"
    assert body["before"] == {"requests_per_minute": 10}
    assert body["after"] == {"requests_per_minute": 1}

    # Written to the file...
    assert "requests_per_minute: 1\n" in teams_path.read_text(encoding="utf-8")
    # ...and in effect for the running gateway.
    assert main_module.app.state.teams_config["tenant-key"]["requests_per_minute"] == 1


def test_teams_endpoint_lists_limits_and_what_is_editable(monkeypatch, tmp_path):
    client, _redis, _teams_path = build_environment(monkeypatch, tmp_path)

    body = client.get("/admin/teams", headers=ADMIN).json()

    assert body["editable_fields"] == [
        "requests_per_minute",
        "tokens_per_minute",
        "monthly_budget_usd",
    ]
    tenant = next(t for t in body["teams"] if t["team_id"] == "admin-api-tenant")
    assert tenant["requests_per_minute"] == 10
    assert tenant["tokens_per_minute"] == 1000
    assert tenant["is_admin"] is False


def test_usage_endpoint_reports_position_against_each_limit(monkeypatch, tmp_path):
    client, redis_client, _teams_path = build_environment(monkeypatch, tmp_path)

    for _ in range(3):
        client.post(
            "/v1/chat",
            headers=TENANT,
            json={"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]},
        )

    body = client.get("/admin/teams/admin-api-tenant/usage", headers=ADMIN).json()

    assert body["requests"]["used"] == 3
    assert body["requests"]["limit"] == 10
    assert body["requests"]["remaining"] == 7
    assert body["tokens"]["used"] > 0
    assert body["tokens"]["limit"] == 1000
    assert body["budget"]["budget_usd"] == 5.0
    assert body["window_seconds"] == 60


def test_usage_endpoint_does_not_consume_limiter_capacity(monkeypatch, tmp_path):
    """An endpoint that reports limiter state must not alter it."""
    client, _redis, _teams_path = build_environment(monkeypatch, tmp_path)

    client.post(
        "/v1/chat",
        headers=TENANT,
        json={"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    first = client.get("/admin/teams/admin-api-tenant/usage", headers=ADMIN).json()
    for _ in range(5):
        client.get("/admin/teams/admin-api-tenant/usage", headers=ADMIN)
    last = client.get("/admin/teams/admin-api-tenant/usage", headers=ADMIN).json()

    assert first["requests"]["used"] == last["requests"]["used"] == 1
    assert first["tokens"]["used"] == last["tokens"]["used"]


def test_usage_on_an_unknown_team_is_404(monkeypatch, tmp_path):
    client, _redis, _teams_path = build_environment(monkeypatch, tmp_path)

    assert client.get("/admin/teams/nobody/usage", headers=ADMIN).status_code == 404


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


def test_a_change_is_recorded_with_actor_and_both_values(monkeypatch, tmp_path):
    client, _redis, _teams_path = build_environment(monkeypatch, tmp_path)

    client.patch(
        "/admin/teams/admin-api-tenant",
        headers=ADMIN,
        json={"monthly_budget_usd": 12.0},
    )

    body = client.get("/admin/audit", headers=ADMIN).json()

    assert body["count"] == 1
    entry = body["entries"][0]
    assert entry["actor"] == "admin-api-admin"
    assert entry["action"] == "update_team_limits"
    assert entry["target"] == "admin-api-tenant"
    assert entry["before"] == {"monthly_budget_usd": 5.0}
    assert entry["after"] == {"monthly_budget_usd": 12.0}
    assert entry["timestamp"]


def test_audit_entries_are_newest_first(monkeypatch, tmp_path):
    client, _redis, _teams_path = build_environment(monkeypatch, tmp_path)

    client.patch(
        "/admin/teams/admin-api-tenant", headers=ADMIN, json={"requests_per_minute": 3}
    )
    client.patch(
        "/admin/teams/admin-api-tenant", headers=ADMIN, json={"requests_per_minute": 4}
    )

    entries = client.get("/admin/audit", headers=ADMIN).json()["entries"]

    assert entries[0]["after"] == {"requests_per_minute": 4}
    assert entries[1]["after"] == {"requests_per_minute": 3}


@pytest.mark.asyncio
async def test_a_redis_failure_does_not_lose_the_change():
    """The change is already applied, so an audit write failure must not report failure."""

    class BrokenRedis:
        async def lpush(self, *args, **kwargs):
            raise ConnectionError("redis is down")

        async def ltrim(self, *args, **kwargs):
            raise ConnectionError("redis is down")

    entry = await audit_module.record_change(
        BrokenRedis(),
        actor="someone",
        action="update_team_limits",
        target="a-team",
        before={"requests_per_minute": 1},
        after={"requests_per_minute": 2},
    )

    # Returned, and logged to stdout, even though Redis rejected it.
    assert entry["after"] == {"requests_per_minute": 2}
