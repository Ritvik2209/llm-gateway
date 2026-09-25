"""Authorization tests for the operational /admin endpoints.

These endpoints expose cross-tenant state and can change gateway behaviour, so they
require an admin-scoped key rather than any valid tenant key.
"""

from fastapi.testclient import TestClient

from app import main as main_module


TEAMS_CONFIG = {
    "tenant-key": {
        "team_id": "admin-test-tenant",
        "allowed_models": ["mock-model"],
        "allowed_providers": ["mock"],
        "provider_priority": ["mock"],
        "requests_per_minute": 100,
        "monthly_budget_usd": 1.0,
        "is_admin": False,
    },
    "admin-key": {
        "team_id": "admin-test-admin",
        "allowed_models": [],
        "allowed_providers": [],
        "provider_priority": [],
        "requests_per_minute": 100,
        "monthly_budget_usd": 0.0,
        "is_admin": True,
    },
    # A team config that predates the flag: absence must deny, not grant.
    "legacy-key": {
        "team_id": "admin-test-legacy",
        "allowed_models": [],
        "allowed_providers": [],
        "provider_priority": [],
        "requests_per_minute": 100,
        "monthly_budget_usd": 0.0,
    },
}

ADMIN_ENDPOINTS = [
    ("get", "/admin/health"),
    ("post", "/admin/mock/toggle-failure"),
]


def create_test_client(monkeypatch):
    monkeypatch.setattr(main_module.app.state, "teams_config", TEAMS_CONFIG)
    return TestClient(main_module.app)


def test_admin_endpoints_reject_missing_credentials(monkeypatch):
    client = create_test_client(monkeypatch)

    for method, path in ADMIN_ENDPOINTS:
        response = getattr(client, method)(path)
        assert response.status_code == 401, path


def test_admin_endpoints_reject_a_valid_tenant_key(monkeypatch):
    """Authentication is not authorization: a real tenant key must still be refused."""
    client = create_test_client(monkeypatch)

    for method, path in ADMIN_ENDPOINTS:
        response = getattr(client, method)(
            path,
            headers={"Authorization": "Bearer tenant-key"},
        )
        assert response.status_code == 403, path
        assert "admin" in response.json()["detail"].lower()


def test_admin_flag_defaults_closed(monkeypatch):
    """A team config without is_admin must be denied, not granted."""
    client = create_test_client(monkeypatch)

    response = client.get(
        "/admin/health",
        headers={"Authorization": "Bearer legacy-key"},
    )

    assert response.status_code == 403


def test_admin_health_allows_an_admin_key(monkeypatch):
    client = create_test_client(monkeypatch)

    response = client.get(
        "/admin/health",
        headers={"Authorization": "Bearer admin-key"},
    )

    assert response.status_code == 200
    # Reports one entry per registered provider, whatever the registry currently holds.
    assert set(response.json()) == set(main_module.health_monitor.providers)


def test_toggle_failure_allows_an_admin_key_and_restores_state(monkeypatch):
    client = create_test_client(monkeypatch)
    original = main_module.mock_provider.should_fail

    try:
        first = client.post(
            "/admin/mock/toggle-failure",
            headers={"Authorization": "Bearer admin-key"},
        )
        assert first.status_code == 200
        assert first.json()["should_fail"] is not original

        second = client.post(
            "/admin/mock/toggle-failure",
            headers={"Authorization": "Bearer admin-key"},
        )
        assert second.json()["should_fail"] is original
    finally:
        main_module.mock_provider.should_fail = original
