"""API tests for the LLM SOAP editor workflow (#81), without Postgres or an LLM.

The routes and use cases run for real; the database, the SOAP extractor and the
:class:`~soap.llm_editor.SoapEditAgent` are replaced through FastAPI dependency
overrides. A ``FakeEditAgent`` turns the live :class:`EditContext` into a canned
:class:`ProposalDraft`, so the tests pin the whole workflow — propose, per-op
decisions, ICD preservation, verify-blocking, idempotency, the acceptance metric
and the doctor-edit auto-reject — with no network and no Postgres.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

from fastapi.testclient import TestClient

from app.dependencies import (
    get_correction_editor_session_repository,
    get_correction_repository,
    get_dialogue_repository,
    get_soap_edit_agent,
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
from soap.correction_use_cases import (
    AddCorrectedNote,
    DeleteCorrectedNote,
    UpdateCorrectedNote,
)
from soap.editor_use_cases import AcceptProposalOperation, DecideOperationCommand
from soap.extractor import SoapExtractor
from soap.llm_editor import EditContext, InvalidProposalError, ProposalDraft
from soap.proposal import (
    AddNoteOperation,
    CorrectionEditorSession,
    DeleteNoteOperation,
    OperationSpec,
    ProposedClaim,
    ProposedNote,
    SessionId,
    StaleOperationTarget,
    UpdateNoteOperation,
)
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

Builder = Callable[[EditContext], list[OperationSpec]]


class FakeSession:
    async def flush(self) -> None: ...

    async def rollback(self) -> None: ...

    async def commit(self) -> None: ...


class InMemoryDialogues(DialogueRepository):
    def __init__(self) -> None:
        self._store: dict[object, Dialogue] = {}

    async def save(self, dialogue: Dialogue) -> None:
        self._store[dialogue.id.value] = dialogue

    async def get(self, dialogue_id: DialogueId) -> Dialogue | None:
        return self._store.get(dialogue_id.value)


class InMemoryReports(SoapReportRepository):
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
        return [
            ReportSummary(
                report_id=report.id,
                dialogue_id=self._dialogue_of[report.id.value],
                created_at=self._created_at[report.id.value],
            )
            for report in self._by_id.values()
        ]


class InMemoryCorrections(SoapReportCorrectionRepository):
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
    """Seeds a two-note report: an ICD-coded note and a plain plan note."""

    async def extract(self, dialogue: Dialogue, patient_context: str) -> SoapReport:
        cite = TurnCitation(turn_id=dialogue.turns[0].id, quote="headache")
        coded = SoapNote(
            id=Id.new(),
            subjective=[SoapClaim(id=Id.new(), text="Headache.", citations=[cite])],
            assessment=[
                AssessmentClaim(id=Id.new(), text="Tension headache.", citations=[cite], icd=_ICD)
            ],
        )
        plain = SoapNote(id=Id.new(), plan=[SoapClaim(id=Id.new(), text="Rest.", citations=[cite])])
        return SoapReport(id=Id.new(), notes=[coded, plain])


class FakeEditAgent:
    """Turns the live EditContext into a canned draft, or raises like the real agent."""

    def __init__(self, builder: Builder) -> None:
        self._builder = builder

    async def propose(self, context: EditContext, *, user_request: str) -> ProposalDraft:
        return ProposalDraft(
            user_request=user_request,
            model_id="fake-model",
            prompt_version="fake/v1",
            operations=self._builder(context),
        )


def _client(builder: Builder) -> TestClient:
    dialogues = InMemoryDialogues()
    reports = InMemoryReports()
    corrections = InMemoryCorrections()
    sessions = InMemoryEditorSessions()
    app = create_app()
    app.dependency_overrides[get_session] = lambda: FakeSession()
    app.dependency_overrides[get_dialogue_repository] = lambda: dialogues
    app.dependency_overrides[get_soap_report_repository] = lambda: reports
    app.dependency_overrides[get_correction_repository] = lambda: corrections
    app.dependency_overrides[get_correction_editor_session_repository] = lambda: sessions
    app.dependency_overrides[get_soap_extractor] = lambda: TwoNoteExtractor()
    app.dependency_overrides[get_soap_edit_agent] = lambda: FakeEditAgent(builder)
    return TestClient(app)


def _seed(client: TestClient) -> str:
    """Create a dialogue and its two-note report; return the report id."""
    dialogue = client.post(
        "/dialogues",
        json={"turns": [{"speaker": "patient", "text": "I've had a headache for three days."}]},
    )
    dialogue_id = dialogue.json()["id"]
    report = client.post(f"/dialogues/{dialogue_id}/report").json()
    return str(report["id"])


def _start(client: TestClient, report_id: str) -> None:
    client.post(f"/reports/{report_id}/correction")


def _propose(client: TestClient, report_id: str):
    return client.post(
        f"/reports/{report_id}/correction/editor/proposals",
        json={"user_request": "Tidy the notes.", "patient_id": "P001"},
    )


def _decide(client: TestClient, report_id: str, proposal_id: str, op_id: str, verb: str):
    return client.post(
        f"/reports/{report_id}/correction/editor/proposals/{proposal_id}/operations/{op_id}/{verb}"
    )


def _turn_of(context: EditContext) -> TurnCitation:
    return TurnCitation(turn_id=context.dialogue.turns[0].id)


def _add_plan(context: EditContext) -> list[OperationSpec]:
    claim = ProposedClaim(text="Return if symptoms worsen.", citations=[_turn_of(context)])
    return [AddNoteOperation(content=ProposedNote(plan=[claim]))]


def _update_coded_note(context: EditContext) -> list[OperationSpec]:
    claim = ProposedClaim(text="Tension-type headache, chronic.", citations=[_turn_of(context)])
    return [
        UpdateNoteOperation(
            target_note_id=context.correction.notes[0].id,
            content=ProposedNote(assessment=[claim]),
        )
    ]


def _update0_delete1(context: EditContext) -> list[OperationSpec]:
    return [*_update_coded_note(context), DeleteNoteOperation(context.correction.notes[1].id)]


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def test_propose_persists_pending_operations_without_applying() -> None:
    client = _client(_add_plan)
    report_id = _seed(client)
    _start(client, report_id)

    proposal = _propose(client, report_id)

    assert proposal.status_code == 201
    body = proposal.json()
    assert body["status"] == "pending"
    assert [op["decision"] for op in body["operations"]] == ["pending"]
    # nothing applied: the correction still has exactly its two seeded notes
    assert len(client.get(f"/reports/{report_id}/correction").json()["notes"]) == 2


def test_get_proposal_shows_before_and_proposed_diff() -> None:
    client = _client(_update_coded_note)
    report_id = _seed(client)
    _start(client, report_id)
    _propose(client, report_id)

    operation = client.get(f"/reports/{report_id}/correction/editor/proposals").json()[
        "operations"
    ][0]

    assert operation["type"] == "update_note"
    assert operation["before"]["sections"]["assessment"][0]["text"] == "Tension headache."
    assert operation["proposed"]["sections"]["assessment"][0]["text"] == (
        "Tension-type headache, chronic."
    )


def test_accepted_operation_is_applied_to_the_correction() -> None:
    client = _client(_add_plan)
    report_id = _seed(client)
    _start(client, report_id)
    proposal = _propose(client, report_id).json()

    _decide(client, report_id, proposal["id"], proposal["operations"][0]["id"], "accept")

    notes = client.get(f"/reports/{report_id}/correction").json()["notes"]
    assert len(notes) == 3
    assert notes[-1]["source_note_id"] is None
    assert notes[-1]["sections"]["plan"][0]["text"] == "Return if symptoms worsen."


def test_rejected_operation_leaves_the_correction_unchanged() -> None:
    client = _client(_add_plan)
    report_id = _seed(client)
    _start(client, report_id)
    proposal = _propose(client, report_id).json()

    _decide(client, report_id, proposal["id"], proposal["operations"][0]["id"], "reject")

    assert len(client.get(f"/reports/{report_id}/correction").json()["notes"]) == 2


def test_mixed_decisions_apply_only_the_accepted_operation() -> None:
    client = _client(_update0_delete1)
    report_id = _seed(client)
    _start(client, report_id)
    proposal = _propose(client, report_id).json()
    update_op, delete_op = proposal["operations"]

    _decide(client, report_id, proposal["id"], update_op["id"], "accept")
    reject = _decide(client, report_id, proposal["id"], delete_op["id"], "reject")

    assert reject.json()["status"] == "mixed"
    notes = client.get(f"/reports/{report_id}/correction").json()["notes"]
    assert len(notes) == 2  # the rejected delete kept the second note
    assert notes[0]["sections"]["assessment"][0]["text"] == "Tension-type headache, chronic."


def test_accepted_update_preserves_the_note_icd() -> None:
    client = _client(_update_coded_note)
    report_id = _seed(client)
    _start(client, report_id)
    proposal = _propose(client, report_id).json()

    _decide(client, report_id, proposal["id"], proposal["operations"][0]["id"], "accept")

    assessment = client.get(f"/reports/{report_id}/correction").json()["notes"][0]["sections"][
        "assessment"
    ][0]
    assert assessment["text"] == "Tension-type headache, chronic."
    assert assessment["icd"]["code"] == _ICD.code


def test_icd_unchanged_end_to_end_propose_mixed_verify() -> None:
    client = _client(_update0_delete1)
    report_id = _seed(client)
    _start(client, report_id)
    original_report = client.get(f"/reports/{report_id}").json()
    proposal = _propose(client, report_id).json()
    update_op, delete_op = proposal["operations"]

    _decide(client, report_id, proposal["id"], update_op["id"], "accept")
    _decide(client, report_id, proposal["id"], delete_op["id"], "reject")
    verified = client.post(
        f"/reports/{report_id}/correction/verify", json={"doctor_id": "dr-house"}
    )

    assert verified.status_code == 200
    assert client.get(f"/reports/{report_id}").json() == original_report  # source never changes
    note = client.get(f"/reports/{report_id}/correction").json()["notes"][0]
    assert note["sections"]["assessment"][0]["icd"]["code"] == _ICD.code


def test_verify_is_blocked_while_an_operation_is_pending() -> None:
    client = _client(_add_plan)
    report_id = _seed(client)
    _start(client, report_id)
    _propose(client, report_id)

    response = client.post(
        f"/reports/{report_id}/correction/verify", json={"doctor_id": "dr-house"}
    )

    assert response.status_code == 409
    assert response.json()["code"] == "pending_operations_block_verify"


def test_verify_succeeds_once_every_operation_is_decided() -> None:
    client = _client(_add_plan)
    report_id = _seed(client)
    _start(client, report_id)
    proposal = _propose(client, report_id).json()
    _decide(client, report_id, proposal["id"], proposal["operations"][0]["id"], "accept")

    response = client.post(
        f"/reports/{report_id}/correction/verify", json={"doctor_id": "dr-house"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_second_proposal_is_blocked_while_one_is_active() -> None:
    client = _client(_add_plan)
    report_id = _seed(client)
    _start(client, report_id)
    _propose(client, report_id)

    second = _propose(client, report_id)

    assert second.status_code == 409
    assert second.json()["code"] == "active_proposal_exists"


def test_new_proposal_is_allowed_after_all_operations_are_decided() -> None:
    client = _client(_add_plan)
    report_id = _seed(client)
    _start(client, report_id)
    first = _propose(client, report_id).json()
    _decide(client, report_id, first["id"], first["operations"][0]["id"], "reject")

    second = _propose(client, report_id)

    assert second.status_code == 201


def test_repeated_accept_is_idempotent() -> None:
    client = _client(_add_plan)
    report_id = _seed(client)
    _start(client, report_id)
    proposal = _propose(client, report_id).json()
    op_id = proposal["operations"][0]["id"]

    first = _decide(client, report_id, proposal["id"], op_id, "accept")
    second = _decide(client, report_id, proposal["id"], op_id, "accept")

    assert first.status_code == 200 and second.status_code == 200
    # the note was applied exactly once, not twice
    assert len(client.get(f"/reports/{report_id}/correction").json()["notes"]) == 3


def test_repeated_reject_is_idempotent() -> None:
    client = _client(_add_plan)
    report_id = _seed(client)
    _start(client, report_id)
    proposal = _propose(client, report_id).json()
    op_id = proposal["operations"][0]["id"]

    first = _decide(client, report_id, proposal["id"], op_id, "reject")
    second = _decide(client, report_id, proposal["id"], op_id, "reject")

    assert first.status_code == 200 and second.json()["operations"][0]["decision"] == "rejected"


def test_accept_then_reject_is_a_conflict() -> None:
    client = _client(_add_plan)
    report_id = _seed(client)
    _start(client, report_id)
    proposal = _propose(client, report_id).json()
    op_id = proposal["operations"][0]["id"]
    _decide(client, report_id, proposal["id"], op_id, "accept")

    conflict = _decide(client, report_id, proposal["id"], op_id, "reject")

    assert conflict.status_code == 409
    assert conflict.json()["code"] == "conflicting_decision"


def test_acceptance_rate_uses_the_decided_denominator() -> None:
    client = _client(_update0_delete1)
    report_id = _seed(client)
    _start(client, report_id)
    proposal = _propose(client, report_id).json()
    update_op, delete_op = proposal["operations"]
    _decide(client, report_id, proposal["id"], update_op["id"], "accept")
    _decide(client, report_id, proposal["id"], delete_op["id"], "reject")

    metric = client.get(f"/reports/{report_id}/correction/editor/metric").json()

    assert metric == {
        "correction_id": metric["correction_id"],
        "since": None,
        "until": None,
        "proposed": 2,
        "accepted": 1,
        "rejected": 1,
        "pending": 0,
        "acceptance_rate": 0.5,
        "breakdown": [
            {
                "model_id": "fake-model",
                "prompt_version": "fake/v1",
                "proposed": 2,
                "accepted": 1,
                "rejected": 1,
                "pending": 0,
                "acceptance_rate": 0.5,
            }
        ],
    }


def test_acceptance_rate_is_null_without_any_decision() -> None:
    client = _client(_add_plan)
    report_id = _seed(client)
    _start(client, report_id)
    _propose(client, report_id)

    metric = client.get(f"/reports/{report_id}/correction/editor/metric").json()

    assert metric["proposed"] == 1
    assert metric["pending"] == 1
    assert metric["acceptance_rate"] is None


def test_metric_time_window_excludes_proposals_outside_the_range() -> None:
    client = _client(_add_plan)
    report_id = _seed(client)
    _start(client, report_id)
    _propose(client, report_id)

    metric = client.get(
        f"/reports/{report_id}/correction/editor/metric",
        params={"until": "2000-01-01T00:00:00"},
    ).json()

    assert metric["proposed"] == 0
    assert metric["acceptance_rate"] is None


def test_doctor_manual_edit_auto_rejects_pending_operations() -> None:
    client = _client(_update_coded_note)
    report_id = _seed(client)
    _start(client, report_id)
    _propose(client, report_id)
    note_id = client.get(f"/reports/{report_id}/correction").json()["notes"][0]["id"]

    client.put(
        f"/reports/{report_id}/correction/notes/{note_id}",
        json={
            "subjective": [
                {
                    "text": "Doctor rewrote this.",
                    "citations": [{"turn_id": _turn(client, report_id)}],
                }
            ]
        },
    )

    operation = client.get(f"/reports/{report_id}/correction/editor/proposals").json()[
        "operations"
    ][0]
    assert operation["decision"] == "rejected"
    assert operation["decision_reason"] == "doctor_edit"
    # the auto-reject is recorded and counts toward the metric
    metric = client.get(f"/reports/{report_id}/correction/editor/metric").json()
    assert metric["rejected"] == 1 and metric["acceptance_rate"] == 0.0
    # and verify is no longer blocked
    assert (
        client.post(
            f"/reports/{report_id}/correction/verify", json={"doctor_id": "dr-house"}
        ).status_code
        == 200
    )


def test_propose_on_a_report_without_a_correction_is_404() -> None:
    client = _client(_add_plan)

    response = _propose(client, str(uuid4()))

    assert response.status_code == 404
    assert response.json()["code"] == "correction_not_found"


def test_get_proposal_without_any_proposal_is_404() -> None:
    client = _client(_add_plan)
    report_id = _seed(client)
    _start(client, report_id)

    response = client.get(f"/reports/{report_id}/correction/editor/proposals")

    assert response.status_code == 404
    assert response.json()["code"] == "proposal_not_found"


def test_deciding_an_unknown_operation_is_404() -> None:
    client = _client(_add_plan)
    report_id = _seed(client)
    _start(client, report_id)
    proposal = _propose(client, report_id).json()

    response = _decide(client, report_id, proposal["id"], str(uuid4()), "accept")

    assert response.status_code == 404
    assert response.json()["code"] == "operation_not_found"


def test_invalid_generated_content_is_422() -> None:
    def _reject_everything(_: EditContext) -> list[OperationSpec]:
        raise InvalidProposalError("unknown_target_note")

    client = _client(_reject_everything)
    report_id = _seed(client)
    _start(client, report_id)

    response = _propose(client, report_id)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_generated_content"


def test_stale_operation_is_rejected_by_the_accept_use_case() -> None:
    dialogue = Dialogue.start()
    turn = dialogue.add_turn("patient", "I've had a headache for three days.")
    cite = TurnCitation(turn_id=turn.id)
    report = SoapReport(
        id=Id.new(),
        notes=[
            SoapNote(id=Id.new(), subjective=[SoapClaim(id=Id.new(), text="H.", citations=[cite])])
        ],
    )
    at = datetime(2026, 7, 21, 9, 0, tzinfo=UTC)
    correction = SoapReportCorrection.start(report, created_at=at)
    editor = CorrectionEditorSession.start(correction.id, created_at=at)
    proposal = editor.propose(
        correction,
        user_request="Rewrite note one.",
        model_id="fake-model",
        prompt_version="fake/v1",
        operations=[
            UpdateNoteOperation(
                correction.notes[0].id,
                ProposedNote(subjective=[ProposedClaim("Rewritten.", [cite])]),
            )
        ],
        at=at,
    )
    correction.update_note(  # the doctor changed the target after the proposal was formed
        correction.notes[0].id,
        at=at,
        subjective=[SoapClaim(id=Id.new(), text="Moved on.", citations=[cite])],
    )

    sessions = InMemoryEditorSessions()
    dialogues = InMemoryDialogues()
    reports = InMemoryReports()
    corrections = InMemoryCorrections()
    _run(dialogues.save(dialogue))
    _run(reports.save(report, dialogue_id=dialogue.id, created_at=at))
    _run(corrections.save(correction))
    _run(sessions.save(editor))
    fake = FakeSession()
    accept = AcceptProposalOperation(
        fake,
        sessions,
        AddCorrectedNote(fake, corrections, reports, dialogues),
        UpdateCorrectedNote(fake, corrections, reports, dialogues),
        DeleteCorrectedNote(fake, corrections),
    )

    try:
        _run(
            accept.execute(
                correction, DecideOperationCommand(proposal.id, proposal.operations[0].id)
            )
        )
        raise AssertionError("expected a stale-target rejection")
    except StaleOperationTarget:
        pass


def _turn(client: TestClient, report_id: str) -> str:
    note = client.get(f"/reports/{report_id}/correction").json()["notes"][0]
    return note["sections"]["subjective"][0]["citations"][0]["turn_id"]
