"""API tests for the SOAP-correction workflow, without Postgres or an LLM.

The database and the extractor are replaced through FastAPI dependency
overrides: in-memory repositories stand in for the tables, a stub session
accepts the commits, and a canned extractor stands in for the LLM. That keeps
the tests hermetic while exercising the real routes end to end — the whole
draft → edit → verify → reopen workflow, the "original report never changes"
invariant, and every error category (404 / 409 / 422).
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.dependencies import (
    get_correction_editor_session_repository,
    get_correction_repository,
    get_dialogue_repository,
    get_soap_extractor,
    get_soap_report_repository,
)
from app.main import create_app
from dialogue.dialogue import Dialogue, DialogueId
from dialogue.repository import DialogueRepository
from infra.db import get_session
from shared.value_objects import Id
from soap.correction import CorrectionId, SoapReportCorrection
from soap.correction_repository import SoapReportCorrectionRepository
from soap.extractor import SoapExtractor
from soap.proposal import CorrectionEditorSession, SessionId
from soap.proposal_repository import CorrectionEditorSessionRepository
from soap.repository import ReportSummary, SoapReportRepository
from soap.soap import (
    AssessmentClaim,
    IcdCoding,
    SoapClaim,
    SoapNote,
    SoapReport,
    SoapReportId,
    TurnCitation,
)

_ICD = IcdCoding(code="G44.2", name="Tension-type headache", classifier_url="https://icd/G44.2")
_NEW_ICD = IcdCoding(code="G43.9", name="Migraine", classifier_url="https://icd/G43.9")


class FakeSession:
    async def flush(self) -> None: ...

    async def rollback(self) -> None: ...

    async def commit(self) -> None: ...


class InMemoryDialogueRepository(DialogueRepository):
    def __init__(self) -> None:
        self._store: dict[object, Dialogue] = {}

    async def save(self, dialogue: Dialogue) -> None:
        self._store[dialogue.id.value] = dialogue

    async def get(self, dialogue_id: DialogueId) -> Dialogue | None:
        return self._store.get(dialogue_id.value)


class InMemorySoapReportRepository(SoapReportRepository):
    def __init__(self) -> None:
        self._by_id: dict[object, SoapReport] = {}
        self._by_dialogue: dict[object, SoapReport] = {}
        self._dialogue_of: dict[object, DialogueId] = {}
        self._created_at: dict[object, datetime] = {}

    async def save(
        self, report: SoapReport, *, dialogue_id: DialogueId, created_at: datetime
    ) -> None:
        self._by_id[report.id.value] = report
        self._by_dialogue[dialogue_id.value] = report
        self._dialogue_of[report.id.value] = dialogue_id
        self._created_at[report.id.value] = created_at

    async def get(self, report_id: SoapReportId) -> SoapReport | None:
        return self._by_id.get(report_id.value)

    async def get_by_dialogue_id(self, dialogue_id: DialogueId) -> SoapReport | None:
        return self._by_dialogue.get(dialogue_id.value)

    async def get_dialogue_id(self, report_id: SoapReportId) -> DialogueId | None:
        return self._dialogue_of.get(report_id.value)

    async def list_summaries(self) -> list[ReportSummary]:
        summaries = [
            ReportSummary(
                report_id=report.id,
                dialogue_id=self._dialogue_of[report.id.value],
                created_at=self._created_at[report.id.value],
            )
            for report in self._by_id.values()
        ]
        return sorted(summaries, key=lambda summary: summary.created_at, reverse=True)


class InMemoryCorrectionRepository(SoapReportCorrectionRepository):
    def __init__(self) -> None:
        self._by_id: dict[object, SoapReportCorrection] = {}
        self._by_source: dict[object, SoapReportCorrection] = {}

    async def save(self, correction: SoapReportCorrection) -> None:
        self._by_id[correction.id.value] = correction
        self._by_source[correction.source_report_id.value] = correction

    async def get(self, correction_id: CorrectionId) -> SoapReportCorrection | None:
        return self._by_id.get(correction_id.value)

    async def get_by_source_report_id(self, report_id: SoapReportId) -> SoapReportCorrection | None:
        return self._by_source.get(report_id.value)


class InMemoryEditorSessionRepository(CorrectionEditorSessionRepository):
    def __init__(self) -> None:
        self._by_id: dict[object, CorrectionEditorSession] = {}
        self._by_correction: dict[object, CorrectionEditorSession] = {}

    async def save(self, session: CorrectionEditorSession) -> None:
        self._by_id[session.id.value] = session
        self._by_correction[session.correction_id.value] = session

    async def get(self, session_id: SessionId) -> CorrectionEditorSession | None:
        return self._by_id.get(session_id.value)

    async def get_for_correction(
        self, correction_id: CorrectionId
    ) -> CorrectionEditorSession | None:
        return self._by_correction.get(correction_id.value)


class StubExtractor(SoapExtractor):
    async def extract(self, dialogue: Dialogue, patient_context: str) -> SoapReport:
        citation = TurnCitation(turn_id=dialogue.turns[0].id, quote="headache")
        note = SoapNote(
            id=Id.new(),
            subjective=[SoapClaim(id=Id.new(), text="Headache.", citations=[citation])],
            assessment=[
                AssessmentClaim(
                    id=Id.new(), text="Tension headache.", citations=[citation], icd=_ICD
                )
            ],
            confidence=0.87,
        )
        return SoapReport(id=Id.new(), notes=[note])


def _client() -> TestClient:
    session = FakeSession()
    dialogues = InMemoryDialogueRepository()
    reports = InMemorySoapReportRepository()
    corrections = InMemoryCorrectionRepository()
    app = create_app()
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_dialogue_repository] = lambda: dialogues
    app.dependency_overrides[get_soap_report_repository] = lambda: reports
    app.dependency_overrides[get_correction_repository] = lambda: corrections
    app.dependency_overrides[get_correction_editor_session_repository] = lambda: (
        InMemoryEditorSessionRepository()
    )
    app.dependency_overrides[get_soap_extractor] = lambda: StubExtractor()
    return TestClient(app)


def _seed_report(client: TestClient) -> tuple[str, str]:
    """Create a dialogue, extract its report, and return ``(report_id, turn_id)``."""
    dialogue = client.post(
        "/dialogues",
        json={
            "turns": [
                {"speaker": "patient", "text": "I've had a headache for three days."},
                {"speaker": "doctor", "text": "Blood pressure is 140 over 90."},
            ]
        },
    )
    dialogue_id = dialogue.json()["id"]
    report = client.post(f"/dialogues/{dialogue_id}/report").json()
    turn_id = report["notes"][0]["sections"]["subjective"][0]["citations"][0]["turn_id"]
    return report["id"], turn_id


def _grounded_note(turn_id: str) -> dict[str, object]:
    return {"objective": [{"text": "BP 140/90.", "citations": [{"turn_id": turn_id}]}]}


def _recoded_assessment(turn_id: str) -> dict[str, object]:
    return {
        "assessment": [
            {
                "text": "Migraine.",
                "citations": [{"turn_id": turn_id}],
                "icd": {"code": "G43.9", "name": "Migraine", "classifier_url": "https://icd/G43.9"},
            }
        ]
    }


def test_start_creates_a_draft_copy_of_the_source() -> None:
    client = _client()
    report_id, _ = _seed_report(client)

    response = client.post(f"/reports/{report_id}/correction")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "draft"
    assert body["source_report_id"] == report_id
    assert body["notes"][0]["source_note_id"] is not None


def test_start_is_idempotent() -> None:
    client = _client()
    report_id, _ = _seed_report(client)

    first = client.post(f"/reports/{report_id}/correction").json()
    second = client.post(f"/reports/{report_id}/correction").json()

    assert second["id"] == first["id"]
    assert len(second["notes"]) == len(first["notes"])


def test_get_returns_the_draft() -> None:
    client = _client()
    report_id, _ = _seed_report(client)
    client.post(f"/reports/{report_id}/correction")

    response = client.get(f"/reports/{report_id}/correction")

    assert response.status_code == 200
    assert response.json()["status"] == "draft"


def test_update_replaces_sections_and_icd() -> None:
    client = _client()
    report_id, turn_id = _seed_report(client)
    note_id = client.post(f"/reports/{report_id}/correction").json()["notes"][0]["id"]

    response = client.put(
        f"/reports/{report_id}/correction/notes/{note_id}",
        json=_recoded_assessment(turn_id),
    )

    assert response.status_code == 200
    note = response.json()["notes"][0]
    assert note["sections"]["assessment"][0]["icd"]["code"] == "G43.9"
    assert note["sections"]["subjective"] == []


def test_add_note_appends_a_doctor_authored_note() -> None:
    client = _client()
    report_id, turn_id = _seed_report(client)
    before = client.post(f"/reports/{report_id}/correction").json()["notes"]

    response = client.post(f"/reports/{report_id}/correction/notes", json=_grounded_note(turn_id))

    assert response.status_code == 200
    notes = response.json()["notes"]
    assert len(notes) == len(before) + 1
    assert notes[-1]["source_note_id"] is None


def test_delete_removes_a_note() -> None:
    client = _client()
    report_id, _ = _seed_report(client)
    note_id = client.post(f"/reports/{report_id}/correction").json()["notes"][0]["id"]

    response = client.delete(f"/reports/{report_id}/correction/notes/{note_id}")

    assert response.status_code == 200
    assert note_id not in [note["id"] for note in response.json()["notes"]]


def test_verify_stamps_the_doctor() -> None:
    client = _client()
    report_id, _ = _seed_report(client)
    client.post(f"/reports/{report_id}/correction")

    response = client.post(
        f"/reports/{report_id}/correction/verify", json={"doctor_id": "dr-house"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "verified"
    assert body["verified_by"] == "dr-house"
    assert body["verified_at"] is not None


def test_reopen_returns_to_draft_and_re_enables_editing() -> None:
    client = _client()
    report_id, turn_id = _seed_report(client)
    client.post(f"/reports/{report_id}/correction")
    client.post(f"/reports/{report_id}/correction/verify", json={"doctor_id": "dr-house"})

    reopened = client.post(f"/reports/{report_id}/correction/reopen")
    assert reopened.status_code == 200
    assert reopened.json()["status"] == "draft"

    added = client.post(f"/reports/{report_id}/correction/notes", json=_grounded_note(turn_id))
    assert added.status_code == 200


def test_get_without_correction_is_404() -> None:
    client = _client()
    report_id, _ = _seed_report(client)

    response = client.get(f"/reports/{report_id}/correction")

    assert response.status_code == 404
    assert response.json()["code"] == "correction_not_found"


def test_start_on_unknown_report_is_404() -> None:
    client = _client()

    response = client.post(f"/reports/{uuid4()}/correction")

    assert response.status_code == 404
    assert response.json()["code"] == "report_not_found"


def test_update_unknown_note_is_404() -> None:
    client = _client()
    report_id, turn_id = _seed_report(client)
    client.post(f"/reports/{report_id}/correction")

    response = client.put(
        f"/reports/{report_id}/correction/notes/{uuid4()}", json=_grounded_note(turn_id)
    )

    assert response.status_code == 404
    assert response.json()["code"] == "note_not_found"


def test_edit_after_verify_is_409() -> None:
    client = _client()
    report_id, turn_id = _seed_report(client)
    client.post(f"/reports/{report_id}/correction")
    client.post(f"/reports/{report_id}/correction/verify", json={"doctor_id": "dr-house"})

    response = client.post(f"/reports/{report_id}/correction/notes", json=_grounded_note(turn_id))

    assert response.status_code == 409
    assert response.json()["code"] == "correction_not_editable"


def test_ungrounded_citation_is_422() -> None:
    client = _client()
    report_id, _ = _seed_report(client)
    client.post(f"/reports/{report_id}/correction")

    response = client.post(
        f"/reports/{report_id}/correction/notes", json=_grounded_note(str(uuid4()))
    )

    assert response.status_code == 422
    assert response.json()["code"] == "citation_not_in_source_dialogue"


def test_empty_doctor_id_is_422() -> None:
    client = _client()
    report_id, _ = _seed_report(client)
    client.post(f"/reports/{report_id}/correction")

    response = client.post(f"/reports/{report_id}/correction/verify", json={"doctor_id": "  "})

    assert response.status_code == 422
    assert response.json()["code"] == "empty_doctor_id"


def test_source_report_is_unchanged_across_the_workflow() -> None:
    client = _client()
    report_id, turn_id = _seed_report(client)
    original = client.get(f"/reports/{report_id}").json()

    client.post(f"/reports/{report_id}/correction")
    note_id = client.get(f"/reports/{report_id}/correction").json()["notes"][0]["id"]
    client.put(
        f"/reports/{report_id}/correction/notes/{note_id}", json=_recoded_assessment(turn_id)
    )
    client.post(f"/reports/{report_id}/correction/notes", json=_grounded_note(turn_id))
    client.post(f"/reports/{report_id}/correction/verify", json={"doctor_id": "dr-house"})

    assert client.get(f"/reports/{report_id}").json() == original


def test_full_workflow_original_to_verified_to_blocked_to_reopened() -> None:
    client = _client()
    report_id, turn_id = _seed_report(client)
    original = client.get(f"/reports/{report_id}").json()

    draft = client.post(f"/reports/{report_id}/correction").json()
    assert draft["status"] == "draft"
    note_id = draft["notes"][0]["id"]

    edited = client.put(
        f"/reports/{report_id}/correction/notes/{note_id}", json=_recoded_assessment(turn_id)
    ).json()
    assert edited["notes"][0]["sections"]["assessment"][0]["icd"]["code"] == "G43.9"

    added = client.post(
        f"/reports/{report_id}/correction/notes", json=_grounded_note(turn_id)
    ).json()
    assert len(added["notes"]) == 2

    verified = client.post(
        f"/reports/{report_id}/correction/verify", json={"doctor_id": "dr-house"}
    ).json()
    assert verified["status"] == "verified"

    blocked = client.post(f"/reports/{report_id}/correction/notes", json=_grounded_note(turn_id))
    assert blocked.status_code == 409
    assert blocked.json()["code"] == "correction_not_editable"

    reopened = client.post(f"/reports/{report_id}/correction/reopen").json()
    assert reopened["status"] == "draft"

    editable_again = client.post(
        f"/reports/{report_id}/correction/notes", json=_grounded_note(turn_id)
    )
    assert editable_again.status_code == 200

    assert client.get(f"/reports/{report_id}").json() == original


def test_started_correction_ids_are_valid_uuids() -> None:
    client = _client()
    report_id, _ = _seed_report(client)

    body = client.post(f"/reports/{report_id}/correction").json()

    UUID(body["id"])
    UUID(body["notes"][0]["id"])
