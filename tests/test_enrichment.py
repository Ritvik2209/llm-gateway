from fastapi.testclient import TestClient

from app import main as main_module
from app.models.schemas import UnifiedChatResponse


TEAM_SYSTEM_PROMPT = "You are a helpful assistant for Team Alpha. Always be concise."

TEAMS_CONFIG = {
    "demo-team-alpha-local-only": {
        "team_id": "team-alpha",
        "allowed_models": ["llama3.2"],
        "allowed_providers": ["ollama"],
        "system_prompt": TEAM_SYSTEM_PROMPT,
        "requests_per_minute": 10,
        "monthly_budget_usd": 5.0,
    },
    "demo-team-beta-local-only": {
        "team_id": "team-beta",
        "allowed_models": ["llama3.2"],
        "allowed_providers": ["ollama"],
        "requests_per_minute": 5,
        "monthly_budget_usd": 1.0,
    },
}


class CapturingProvider:
    def __init__(self):
        self.requests = []

    async def chat(self, request):
        self.requests.append(
            [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ]
        )
        return UnifiedChatResponse(
            id="test-response",
            model=request.model,
            content="ok",
            input_tokens=0,
            output_tokens=0,
            provider="ollama",
            finish_reason="stop",
        )


def create_test_client(monkeypatch, provider):
    async def allow_request(*args, **kwargs):
        return True, 9

    async def allow_budget(*args, **kwargs):
        return True, 0.0, False

    async def record_spend(*args, **kwargs):
        return 0.0

    monkeypatch.setattr(main_module.app.state, "teams_config", TEAMS_CONFIG)
    monkeypatch.setattr(main_module, "ollama_provider", provider)
    monkeypatch.setattr(main_module, "check_rate_limit", allow_request)
    monkeypatch.setattr(main_module, "check_budget", allow_budget)
    monkeypatch.setattr(main_module, "add_spend", record_spend)
    return TestClient(main_module.app)


def test_team_system_prompt_is_prepended_when_request_has_no_system_message(
    monkeypatch,
):
    provider = CapturingProvider()
    client = create_test_client(monkeypatch, provider)

    response = client.post(
        "/v1/chat",
        headers={"Authorization": "Bearer demo-team-alpha-local-only"},
        json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    assert provider.requests[0] == [
        {"role": "system", "content": TEAM_SYSTEM_PROMPT},
        {"role": "user", "content": "Hello"},
    ]


def test_team_system_prompt_is_not_added_when_request_has_system_message(
    monkeypatch,
):
    provider = CapturingProvider()
    client = create_test_client(monkeypatch, provider)

    response = client.post(
        "/v1/chat",
        headers={"Authorization": "Bearer demo-team-alpha-local-only"},
        json={
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": "Use the caller prompt."},
                {"role": "user", "content": "Hello"},
            ],
        },
    )

    assert response.status_code == 200
    assert provider.requests[0] == [
        {"role": "system", "content": "Use the caller prompt."},
        {"role": "user", "content": "Hello"},
    ]


def test_team_without_system_prompt_leaves_messages_unchanged(monkeypatch):
    provider = CapturingProvider()
    client = create_test_client(monkeypatch, provider)

    response = client.post(
        "/v1/chat",
        headers={"Authorization": "Bearer demo-team-beta-local-only"},
        json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    assert provider.requests[0] == [
        {"role": "user", "content": "Hello"},
    ]
