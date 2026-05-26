"""Smoke tests for the FastAPI service."""
from __future__ import annotations


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_response(client, sample_payload):
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["decision"] in {"approve", "reject"}
    assert "model_name" in body


def test_predict_batch(client, sample_payload):
    response = client.post(
        "/predict/batch", json={"items": [sample_payload, sample_payload]}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    for pred in body["predictions"]:
        assert 0.0 <= pred["default_probability"] <= 1.0


def test_predict_validates_input(client):
    # AMT_INCOME_TOTAL must be >= 0.
    response = client.post("/predict", json={"AMT_INCOME_TOTAL": -100})
    assert response.status_code == 422


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "credit-scoring" in response.json()["service"]
