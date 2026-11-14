from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import create_app


def test_health_and_forecast_unavailable_message():
    client = TestClient(create_app())
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    # No production model bundle in isolated workspace -> unavailable (409) or clear response.
    response = client.post("/api/v1/forecasts", json={"currency": "USD", "cost_basis": "billed"})
    assert response.status_code in (409, 400)
    detail = response.json().get("detail", "")
    assert "validated production model" in detail.lower() or "not yet available" in detail.lower()
