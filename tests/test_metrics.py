from fastapi.testclient import TestClient

from app.main import app


def test_metrics_endpoint_exposes_gateway_metrics():
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "gateway_requests_total" in response.text
    assert "gateway_request_duration_seconds" in response.text
    assert "gateway_errors_total" in response.text
    assert "gateway_fallback_triggered_total" in response.text
    assert "gateway_circuit_breaker_state" in response.text
    assert "gateway_tokens_total" in response.text
