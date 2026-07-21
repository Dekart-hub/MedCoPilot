from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.dependencies import (
    get_correction_repository,
    get_dialogue_repository,
    get_ehr_publication_repository,
    get_publication_outbox_repository,
    get_soap_report_repository,
)
from app.main import create_app
from ehr.publication import PublicationOutbox
from infra.db import get_session
from soap.correction import CorrectionStatus
from tests.test_publication_use_cases import Environment


def _payload(**overrides: str) -> dict[str, str]:
    return {
        "patient_ref": "Patient/p1",
        "encounter_ref": "Encounter/e1",
        "author_ref": "Practitioner/d1",
        **overrides,
    }


def _client(env: Environment) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_session] = lambda: env.session
    app.dependency_overrides[get_dialogue_repository] = lambda: env.dialogues
    app.dependency_overrides[get_soap_report_repository] = lambda: env.reports
    app.dependency_overrides[get_correction_repository] = lambda: env.corrections
    app.dependency_overrides[get_ehr_publication_repository] = lambda: env.publications
    app.dependency_overrides[get_publication_outbox_repository] = lambda: env.outbox
    return TestClient(app)


def test_verified_report_is_accepted_as_durable_pending_publication() -> None:
    env = Environment()
    client = _client(env)

    response = client.post(f"/reports/{env.report.id}/publication", json=_payload())

    assert response.status_code == 202
    body = response.json()
    assert body["source_report_id"] == str(env.report.id)
    assert body["correction_id"] == str(env.correction.id)
    assert body["status"] == "pending"
    assert body["attempts"] == 0
    assert body["last_error"] is None
    assert body["remote_reference"] is None
    assert env.correction.status is CorrectionStatus.PUBLICATION_PENDING
    assert env.session.commits == 1


def test_repeat_post_and_get_return_the_same_publication() -> None:
    env = Environment()
    client = _client(env)
    first = client.post(f"/reports/{env.report.id}/publication", json=_payload())

    repeated = client.post(
        f"/reports/{env.report.id}/publication",
        json=_payload(patient_ref="Patient/ignored"),
    )
    loaded = client.get(f"/reports/{env.report.id}/publication")

    assert repeated.status_code == 202
    assert repeated.json()["id"] == first.json()["id"]
    assert repeated.json()["patient_ref"] == "Patient/p1"
    assert loaded.status_code == 200
    assert loaded.json() == repeated.json()


def test_unverified_report_is_409_and_missing_publication_is_404() -> None:
    env = Environment(verified=False)
    client = _client(env)

    response = client.post(f"/reports/{env.report.id}/publication", json=_payload())
    missing = client.get(f"/reports/{env.report.id}/publication")

    assert response.status_code == 409
    assert response.json()["code"] == "report_not_verified"
    assert missing.status_code == 404
    assert missing.json()["code"] == "publication_not_found"


def test_invalid_fhir_reference_is_stable_422() -> None:
    env = Environment()
    client = _client(env)

    response = client.post(
        f"/reports/{env.report.id}/publication",
        json=_payload(patient_ref="p1"),
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_fhir_reference"


def test_pending_publication_blocks_reopen_and_edit() -> None:
    env = Environment()
    client = _client(env)
    client.post(f"/reports/{env.report.id}/publication", json=_payload())

    reopen = client.post(f"/reports/{env.report.id}/correction/reopen")
    edit = client.post(
        f"/reports/{env.report.id}/correction/notes",
        json={
            "subjective": [
                {
                    "text": "Late change",
                    "citations": [{"turn_id": str(env.dialogue.turns[0].id)}],
                }
            ]
        },
    )

    assert reopen.status_code == 409
    assert reopen.json()["code"] == "invalid_correction_transition"
    assert edit.status_code == 409
    assert edit.json()["code"] == "correction_not_editable"


def test_repeat_post_returns_200_after_delivery() -> None:
    env = Environment()
    client = _client(env)
    client.post(f"/reports/{env.report.id}/publication", json=_payload())
    publication = next(iter(env.publications.items.values()))
    event: PublicationOutbox = next(iter(env.outbox.items.values()))
    delivered_at = datetime(2026, 7, 21, 13, 0, tzinfo=UTC)
    publication.mark_delivered(remote_reference="Bundle/b1", remote_version="1", at=delivered_at)
    event.mark_delivered(at=delivered_at)
    env.correction.mark_published(at=delivered_at)

    response = client.post(f"/reports/{env.report.id}/publication", json=_payload())

    assert response.status_code == 200
    assert response.json()["status"] == "delivered"
    assert response.json()["remote_reference"] == "Bundle/b1"
    assert response.json()["attempts"] == 1
    assert response.json()["next_attempt_at"] is None
