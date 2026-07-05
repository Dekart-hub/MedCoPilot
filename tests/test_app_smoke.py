from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import create_app


@pytest.fixture
def client(monkeypatch) -> TestClient:
    # Dummy-ключ: контейнер поднимается без сети (запросов к LLM здесь нет).
    monkeypatch.setenv("OPENAI__API_KEY", "sk-test")
    from config.settings import get_settings

    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        yield client
    get_settings.cache_clear()


def test_health_is_ok(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_after_startup(client: TestClient):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_create_dialogue_from_text_endpoint(client: TestClient):
    response = client.post(
        "/api/v1/dialogues/from-text",
        json={"text": "person hi\nmedic hello"},
    )

    assert response.status_code == 201
    body = response.json()
    assert len(body["turns"]) == 2
    assert body["turns"][0]["role"] == "person"


def test_create_dialogue_structured_endpoint(client: TestClient):
    response = client.post(
        "/api/v1/dialogues",
        json={"turns": [{"role": "person", "content": "hi"}]},
    )

    assert response.status_code == 201
    assert response.json()["turns"][0]["content"] == "hi"


def test_report_for_unknown_dialogue_is_404(client: TestClient):
    response = client.post(
        "/api/v1/reports",
        json={"dialogue_id": "00000000-0000-0000-0000-000000000000"},
    )

    assert response.status_code == 404
