"""API tests for the dialogue → report pipeline, without Postgres or an LLM.

The database and the extractor are replaced through FastAPI dependency
overrides: in-memory repositories stand in for the tables, a stub session
accepts the commits, and a canned extractor stands in for the LLM. That keeps
the tests hermetic while exercising the real routes end to end — persistence,
idempotency and the 404 paths.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app.dependencies import (
    get_dialogue_repository,
    get_soap_extractor,
    get_soap_report_repository,
)
from app.main import create_app
from dialogue.dialogue import Dialogue, DialogueId
from dialogue.repository import DialogueRepository
from infra.db import get_session
from shared.value_objects import Id
from soap.extractor import SoapExtractor
from soap.repository import SoapReportRepository
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

    async def save(self, report: SoapReport, *, dialogue_id: DialogueId) -> None:
        self._by_id[report.id.value] = report
        self._by_dialogue[dialogue_id.value] = report
        self._dialogue_of[report.id.value] = dialogue_id

    async def get(self, report_id: SoapReportId) -> SoapReport | None:
        return self._by_id.get(report_id.value)

    async def get_by_dialogue_id(self, dialogue_id: DialogueId) -> SoapReport | None:
        return self._by_dialogue.get(dialogue_id.value)

    async def get_dialogue_id(self, report_id: SoapReportId) -> DialogueId | None:
        return self._dialogue_of.get(report_id.value)


class StubExtractor(SoapExtractor):
    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, dialogue: Dialogue, patient_context: str) -> SoapReport:
        self.calls += 1
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


def _client() -> tuple[TestClient, StubExtractor]:
    session = FakeSession()
    dialogues = InMemoryDialogueRepository()
    reports = InMemorySoapReportRepository()
    extractor = StubExtractor()
    app = create_app()
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_dialogue_repository] = lambda: dialogues
    app.dependency_overrides[get_soap_report_repository] = lambda: reports
    app.dependency_overrides[get_soap_extractor] = lambda: extractor
    return TestClient(app), extractor


def _create_dialogue(client: TestClient) -> str:
    response = client.post(
        "/dialogues",
        json={
            "turns": [
                {"speaker": "patient", "text": "I've had a headache for three days."},
                {"speaker": "doctor", "text": "Blood pressure is 140 over 90."},
            ]
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_dialogue_extracts_and_reads_back_a_report_with_icd_and_confidence() -> None:
    client, _ = _client()
    dialogue_id = _create_dialogue(client)

    extracted = client.post(f"/dialogues/{dialogue_id}/report")
    assert extracted.status_code == 200
    body = extracted.json()
    note = body["notes"][0]
    assert note["confidence"] == 0.87
    assert note["sections"]["assessment"][0]["icd"]["code"] == "G44.2"

    fetched = client.get(f"/reports/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == body["id"]


def test_repeat_report_returns_the_same_id_without_re_extracting() -> None:
    client, extractor = _client()
    dialogue_id = _create_dialogue(client)

    first = client.post(f"/dialogues/{dialogue_id}/report").json()["id"]
    second = client.post(f"/dialogues/{dialogue_id}/report").json()["id"]

    assert second == first
    assert extractor.calls == 1


def test_report_for_unknown_dialogue_is_404() -> None:
    client, extractor = _client()

    response = client.post(f"/dialogues/{uuid4()}/report")

    assert response.status_code == 404
    assert extractor.calls == 0


def test_unknown_report_is_404() -> None:
    client, _ = _client()

    response = client.get(f"/reports/{uuid4()}")

    assert response.status_code == 404
