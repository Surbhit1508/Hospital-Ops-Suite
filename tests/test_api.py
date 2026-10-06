from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_dashboard_page_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "Hospital Operations Intelligence Suite" in response.text


def test_metrics_endpoint_returns_expected_sections():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    for key in ("classification", "regression", "nlp", "forecasting", "optimization", "run_history"):
        assert key in body


def test_dashboard_shows_optimization_departments_when_data_present():
    response = client.get("/api/metrics")
    body = response.json()
    if body["optimization"]:
        assert "Emergency" in body["optimization"]["departments"]
