"""HTTP integration tests for dialogue-level online SOAP quality (T23).

The real routes and use cases run together while repositories, session and LLM
are replaced with in-memory fakes.  This pins the complete lifecycle from
correction through verification, quality, reopen and re-verification.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

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

_SOURCE_ICD = IcdCoding("G44.2", "Tension headache", "https://icd/G44.2")
# Goes through the correction API ⇒ must be canonical for T29 catalog validation.
_CORRECTED_ICD = {
    "code": "G43.9",
    "name": "Migraine, unspecified",
}


class FakeSession:
    async def flush(self) -> None: ...

    async def rollback(self) -> None: ...

    async def commit(self) -> None: ...


class InMemoryDialogues(DialogueRepository):
    def __init__(self) -> None:
        self.store: dict[object, Dialogue] = {}

    async def save(self, dialogue: Dialogue) -> None:
        self.store[dialogue.id.value] = dialogue

    async def get(self, dialogue_id: DialogueId) -> Dialogue | None:
        return self.store.get(dialogue_id.value)


class InMemoryReports(SoapReportRepository):
    def __init__(self) -> None:
        self.by_id: dict[object, SoapReport] = {}
        self.by_dialogue: dict[object, SoapReport] = {}
        self.dialogue_of: dict[object, DialogueId] = {}
        self.created_at: dict[object, datetime] = {}

    async def save(
        self, report: SoapReport, *, dialogue_id: DialogueId, created_at: datetime
    ) -> None:
        self.by_id[report.id.value] = report
        self.by_dialogue[dialogue_id.value] = report
        self.dialogue_of[report.id.value] = dialogue_id
        self.created_at[report.id.value] = created_at

    async def get(self, report_id: SoapReportId) -> SoapReport | None:
        return self.by_id.get(report_id.value)

    async def get_by_dialogue_id(self, dialogue_id: DialogueId) -> SoapReport | None:
        return self.by_dialogue.get(dialogue_id.value)

    async def get_dialogue_id(self, report_id: SoapReportId) -> DialogueId | None:
        return self.dialogue_of.get(report_id.value)

    async def list_summaries(self) -> list[ReportSummary]:
        summaries = [
            ReportSummary(
                report_id=report.id,
                dialogue_id=self.dialogue_of[report.id.value],
                created_at=self.created_at[report.id.value],
            )
            for report in self.by_id.values()
        ]
        return sorted(summaries, key=lambda summary: summary.created_at, reverse=True)


class InMemoryCorrections(SoapReportCorrectionRepository):
    def __init__(self) -> None:
        self.by_id: dict[object, SoapReportCorrection] = {}
        self.by_source: dict[object, SoapReportCorrection] = {}

    async def save(self, correction: SoapReportCorrection) -> None:
        self.by_id[correction.id.value] = correction
        self.by_source[correction.source_report_id.value] = correction

    async def get(self, correction_id: CorrectionId) -> SoapReportCorrection | None:
        return self.by_id.get(correction_id.value)

    async def get_by_source_report_id(self, report_id: SoapReportId) -> SoapReportCorrection | None:
        return self.by_source.get(report_id.value)


class InMemoryEditorSessions(CorrectionEditorSessionRepository):
    def __init__(self) -> None:
        self.by_id: dict[object, CorrectionEditorSession] = {}
        self.by_correction: dict[object, CorrectionEditorSession] = {}

    async def save(self, session: CorrectionEditorSession) -> None:
        self.by_id[session.id.value] = session
        self.by_correction[session.correction_id.value] = session

    async def get(self, session_id: SessionId) -> CorrectionEditorSession | None:
        return self.by_id.get(session_id.value)

    async def get_for_correction(
        self, correction_id: CorrectionId
    ) -> CorrectionEditorSession | None:
        return self.by_correction.get(correction_id.value)


class TwoNoteExtractor(SoapExtractor):
    async def extract(self, dialogue: Dialogue, patient_context: str) -> SoapReport:
        citation = TurnCitation(turn_id=dialogue.turns[0].id, quote="pain")
        return SoapReport(
            id=Id.new(),
            notes=[
                SoapNote(
                    id=Id.new(),
                    subjective=[SoapClaim(id=Id.new(), text="pain", citations=[citation])],
                    assessment=[
                        AssessmentClaim(
                            id=Id.new(),
                            text="Headache diagnosis.",
                            citations=[citation],
                            icd=_SOURCE_ICD,
                        )
                    ],
                ),
                SoapNote(
                    id=Id.new(),
                    plan=[SoapClaim(id=Id.new(), text="Rest.", citations=[citation])],
                ),
            ],
        )


def _client() -> TestClient:
    dialogues = InMemoryDialogues()
    reports = InMemoryReports()
    corrections = InMemoryCorrections()
    app = create_app()
    app.dependency_overrides[get_session] = lambda: FakeSession()
    app.dependency_overrides[get_dialogue_repository] = lambda: dialogues
    app.dependency_overrides[get_soap_report_repository] = lambda: reports
    app.dependency_overrides[get_correction_repository] = lambda: corrections
    app.dependency_overrides[get_correction_editor_session_repository] = lambda: (
        InMemoryEditorSessions()
    )
    app.dependency_overrides[get_soap_extractor] = lambda: TwoNoteExtractor()
    return TestClient(app)


def _seed(client: TestClient) -> tuple[str, dict[str, object], str]:
    dialogue = client.post(
        "/dialogues",
        json={"turns": [{"speaker": "patient", "text": "I have head pain."}]},
    )
    assert dialogue.status_code == 201
    dialogue_id = dialogue.json()["id"]
    report = client.post(f"/dialogues/{dialogue_id}/report")
    assert report.status_code == 200
    report_body = report.json()
    turn_id = report_body["notes"][0]["sections"]["subjective"][0]["citations"][0]["turn_id"]
    return dialogue_id, report_body, turn_id


def _matched_note(turn_id: str, subjective: str) -> dict[str, object]:
    citation = [{"turn_id": turn_id}]
    return {
        "subjective": [{"text": subjective, "citations": citation}],
        "assessment": [
            {
                "text": "Headache diagnosis.",
                "citations": citation,
                "icd": _CORRECTED_ICD,
            }
        ],
    }


def _added_note(turn_id: str) -> dict[str, object]:
    return {
        "objective": [{"text": "Doctor-added observation.", "citations": [{"turn_id": turn_id}]}]
    }


def test_edit_add_delete_and_icd_change_return_exact_quality() -> None:
    client = _client()
    dialogue_id, report, turn_id = _seed(client)
    draft = client.post(f"/reports/{report['id']}/correction").json()
    kept = draft["notes"][0]
    removed = draft["notes"][1]

    assert (
        client.put(
            f"/reports/{report['id']}/correction/notes/{kept['id']}",
            json=_matched_note(turn_id, "gain"),
        ).status_code
        == 200
    )
    assert (
        client.delete(f"/reports/{report['id']}/correction/notes/{removed['id']}").status_code
        == 200
    )
    assert (
        client.post(
            f"/reports/{report['id']}/correction/notes", json=_added_note(turn_id)
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/reports/{report['id']}/correction/verify",
            json={"doctor_id": "doctor-1"},
        ).status_code
        == 200
    )

    response = client.get(f"/dialogues/{dialogue_id}/quality")

    assert response.status_code == 200
    assert response.json() == {
        "dialogue_id": dialogue_id,
        "report_id": report["id"],
        "correction_id": draft["id"],
        "notes_added": 1,
        "notes_removed": 1,
        "changed_characters": 1,
        "diagnosis_changes": 1,
        "note_diffs": [
            {
                "source_note_id": report["notes"][0]["id"],
                "corrected_note_id": kept["id"],
                "changed_characters": 1,
                "diagnosis_changed": True,
            }
        ],
    }


def test_draft_and_reopen_hide_quality_until_reverification() -> None:
    client = _client()
    dialogue_id, report, turn_id = _seed(client)
    draft = client.post(f"/reports/{report['id']}/correction").json()
    note_id = draft["notes"][0]["id"]

    draft_quality = client.get(f"/dialogues/{dialogue_id}/quality")
    assert draft_quality.status_code == 409
    assert draft_quality.json()["code"] == "REPORT_NOT_VERIFIED"

    client.put(
        f"/reports/{report['id']}/correction/notes/{note_id}",
        json=_matched_note(turn_id, "gain"),
    )
    client.post(f"/reports/{report['id']}/correction/verify", json={"doctor_id": "doctor-1"})
    assert client.get(f"/dialogues/{dialogue_id}/quality").json()["changed_characters"] == 1

    client.post(f"/reports/{report['id']}/correction/reopen")
    reopened_quality = client.get(f"/dialogues/{dialogue_id}/quality")
    assert reopened_quality.status_code == 409
    assert reopened_quality.json()["code"] == "REPORT_NOT_VERIFIED"

    client.put(
        f"/reports/{report['id']}/correction/notes/{note_id}",
        json=_matched_note(turn_id, "gain!!"),
    )
    client.post(f"/reports/{report['id']}/correction/verify", json={"doctor_id": "doctor-2"})
    recalculated = client.get(f"/dialogues/{dialogue_id}/quality")
    assert recalculated.status_code == 200
    assert recalculated.json()["changed_characters"] == 3


def test_missing_report_and_correction_have_stable_errors() -> None:
    client = _client()

    missing_report = client.get(f"/dialogues/{uuid4()}/quality")
    assert missing_report.status_code == 404
    assert missing_report.json()["code"] == "report_not_found"

    dialogue_id, _, _ = _seed(client)
    missing_correction = client.get(f"/dialogues/{dialogue_id}/quality")
    assert missing_correction.status_code == 404
    assert missing_correction.json()["code"] == "correction_not_found"
