from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import create_app
from dialogue import SAMPLE_DIALOGUE_ID
from shared.value_objects import Id
from soap import ReportView


@pytest.fixture
def client(monkeypatch) -> TestClient:
    # Dummy-ключ: контейнер поднимается без сети (запросов к LLM здесь нет).
    monkeypatch.setenv("OPENAI__API_KEY", "sk-test")
    monkeypatch.setenv("EHR__ENABLED", "false")
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


def test_dialogue_can_carry_explicit_ehr_linkage(client: TestClient):
    response = client.post(
        "/api/v1/dialogues",
        json={
            "turns": [{"role": "person", "content": "hi"}],
            "patient_ref": "Patient/p1",
            "encounter_ref": "Encounter/e1",
        },
    )

    assert response.status_code == 201
    assert response.json()["patient_ref"] == "Patient/p1"
    assert response.json()["encounter_ref"] == "Encounter/e1"


def test_mock_ehr_context_is_explicitly_disabled_by_default(client: TestClient):
    response = client.get(
        "/api/v1/ehr/dialogues/11111111-1111-1111-1111-111111111111/context"
    )

    assert response.status_code == 503
    assert "EHR integration is disabled" in response.json()["detail"]


def test_report_approval_and_disabled_sync_routes(client: TestClient):
    container = client.app.state.container
    dialogue = asyncio.run(
        container.dialogue_repository.get(SAMPLE_DIALOGUE_ID)
    )
    assert dialogue is not None
    now = datetime.now(timezone.utc)
    report = ReportView(id=Id.new(), notes=[], created_at=now, updated_at=now)
    record = asyncio.run(
        container.report_workflow.store_generated_report(report, dialogue)
    )

    approval = client.post(
        f"/api/v1/reports/{record.report_id}/approve",
        json={"clinician_ref": "Practitioner/mock-gp-001"},
    )
    sync = client.post(f"/api/v1/reports/{record.report_id}/ehr-sync")
    workflow = client.get(f"/api/v1/reports/{record.report_id}/workflow")

    assert approval.status_code == 200
    assert approval.json()["approval_status"] == "approved"
    assert sync.status_code == 503
    assert workflow.json()["sync_status"] == "failed"


def test_report_for_unknown_dialogue_is_404(client: TestClient):
    response = client.post(
        "/api/v1/reports",
        json={"dialogue_id": "00000000-0000-0000-0000-000000000000"},
    )

    assert response.status_code == 404
