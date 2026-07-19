"""Unit tests for the SOAP-correction workflow use cases, with in-memory fakes.

No database and no LLM: the dialogue, report and correction repositories are
dictionaries and the session is a stub. The tests pin the behaviours that matter
— starting is idempotent (one draft, no duplicate), notes can be edited, added
and deleted, edits persist without auto-verifying, verify stamps the doctor and
blocks edits while reopen re-enables them — and every error path: a missing
source report, a missing correction, a missing note, an ungrounded citation and
an edit of a verified correction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy.exc import IntegrityError

from dialogue.dialogue import Dialogue, DialogueId
from dialogue.repository import DialogueRepository
from shared.value_objects import Id
from soap.correction import (
    CorrectionNotEditable,
    CorrectionStatus,
    NoteNotInCorrection,
    SoapReportCorrection,
)
from soap.correction_repository import SoapReportCorrectionRepository
from soap.correction_use_cases import (
    AddCorrectedNote,
    AddCorrectedNoteCommand,
    CitationNotInSourceDialogue,
    CorrectionNotFoundError,
    DeleteCorrectedNote,
    DeleteCorrectedNoteCommand,
    ReopenSoapCorrection,
    ReopenSoapCorrectionCommand,
    SourceReportNotFoundError,
    StartSoapCorrection,
    StartSoapCorrectionCommand,
    UpdateCorrectedNote,
    UpdateCorrectedNoteCommand,
    VerifySoapCorrection,
    VerifySoapCorrectionCommand,
)
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

_ICD = IcdCoding(code="I10", name="Essential hypertension", classifier_url="https://icd/I10")
_NEW_ICD = IcdCoding(code="G44.2", name="Tension-type headache", classifier_url="https://icd/G44.2")


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


class FakeSession:
    """Stands in for an AsyncSession: counts flush/rollback/commit, no-ops otherwise."""

    def __init__(self) -> None:
        self.flushes = 0
        self.rollbacks = 0
        self.commits = 0

    async def flush(self) -> None:
        self.flushes += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def commit(self) -> None:
        self.commits += 1


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


class InMemoryCorrectionRepository(SoapReportCorrectionRepository):
    def __init__(self) -> None:
        self._by_id: dict[UUID, SoapReportCorrection] = {}
        self._by_source: dict[UUID, SoapReportCorrection] = {}
        self.saves = 0

    async def save(self, correction: SoapReportCorrection) -> None:
        self.saves += 1
        self._by_id[correction.id.value] = correction
        self._by_source[correction.source_report_id.value] = correction

    async def get(self, correction_id: Id[SoapReportCorrection]) -> SoapReportCorrection | None:
        return self._by_id.get(correction_id.value)

    async def get_by_source_report_id(self, report_id: SoapReportId) -> SoapReportCorrection | None:
        return self._by_source.get(report_id.value)

    def rows(self) -> int:
        return len(self._by_id)


def _dialogue() -> Dialogue:
    dialogue = Dialogue.start()
    dialogue.add_turn("patient", "I've had a headache for three days.")
    dialogue.add_turn("doctor", "Blood pressure is 140 over 90.")
    return dialogue


def _report(dialogue: Dialogue) -> SoapReport:
    cite = TurnCitation(turn_id=dialogue.turns[0].id, quote="headache")
    note = SoapNote(
        id=Id.new(),
        subjective=[SoapClaim(id=Id.new(), text="Headache for three days.", citations=[cite])],
        assessment=[
            AssessmentClaim(id=Id.new(), text="Tension headache.", citations=[cite], icd=_ICD)
        ],
    )
    return SoapReport(id=Id.new(), notes=[note])


@dataclass(slots=True)
class Env:
    """A seeded world: one dialogue and its source report, all repositories wired."""

    session: FakeSession
    dialogues: InMemoryDialogueRepository
    reports: InMemorySoapReportRepository
    corrections: InMemoryCorrectionRepository
    dialogue: Dialogue
    report: SoapReport

    def start(self) -> StartSoapCorrection:
        return StartSoapCorrection(self.session, self.corrections, self.reports)

    def add(self) -> AddCorrectedNote:
        return AddCorrectedNote(self.session, self.corrections, self.reports, self.dialogues)

    def update(self) -> UpdateCorrectedNote:
        return UpdateCorrectedNote(self.session, self.corrections, self.reports, self.dialogues)

    def delete(self) -> DeleteCorrectedNote:
        return DeleteCorrectedNote(self.session, self.corrections)

    def verify(self) -> VerifySoapCorrection:
        return VerifySoapCorrection(self.session, self.corrections)

    def reopen(self) -> ReopenSoapCorrection:
        return ReopenSoapCorrection(self.session, self.corrections)


def _env() -> Env:
    dialogue = _dialogue()
    report = _report(dialogue)
    dialogues = InMemoryDialogueRepository()
    reports = InMemorySoapReportRepository()
    _run(dialogues.save(dialogue))
    _run(reports.save(report, dialogue_id=dialogue.id))
    return Env(FakeSession(), dialogues, reports, InMemoryCorrectionRepository(), dialogue, report)


def _started(env: Env) -> SoapReportCorrection:
    return _run(env.start().execute(StartSoapCorrectionCommand(report_id=env.report.id)))


def _grounded_claim(env: Env, text: str) -> SoapClaim:
    return SoapClaim(
        id=Id.new(), text=text, citations=[TurnCitation(turn_id=env.dialogue.turns[1].id)]
    )


def test_first_start_creates_a_draft_copy_of_the_source() -> None:
    env = _env()

    correction = _started(env)

    assert correction.status is CorrectionStatus.DRAFT
    assert correction.source_report_id == env.report.id
    assert [note.source_note_id for note in correction.notes] == [env.report.notes[0].id]


def test_repeat_start_returns_the_same_correction_without_duplicating() -> None:
    env = _env()

    first = _started(env)
    second = _started(env)

    assert second.id == first.id
    assert env.corrections.rows() == 1
    assert env.corrections.saves == 1  # the second call did not persist a new row


def test_start_on_a_missing_report_is_rejected() -> None:
    env = _env()

    with pytest.raises(SourceReportNotFoundError):
        _run(env.start().execute(StartSoapCorrectionCommand(report_id=Id.new())))


class RacingCorrections(SoapReportCorrectionRepository):
    """A concurrent writer won: invisible on the pre-check, visible on the re-fetch."""

    def __init__(self, winner: SoapReportCorrection) -> None:
        self._winner = winner
        self._probes = 0

    async def save(self, correction: SoapReportCorrection) -> None: ...

    async def get(self, correction_id: Id[SoapReportCorrection]) -> SoapReportCorrection | None:
        return self._winner

    async def get_by_source_report_id(self, report_id: SoapReportId) -> SoapReportCorrection | None:
        self._probes += 1
        return None if self._probes == 1 else self._winner


class ConflictingSession(FakeSession):
    async def flush(self) -> None:
        raise IntegrityError("insert", None, Exception("duplicate source_report_id"))


def test_lost_race_on_start_resolves_to_the_winning_correction() -> None:
    env = _env()
    winner = SoapReportCorrection.start(env.report, created_at=datetime(2026, 7, 19, tzinfo=UTC))
    use_case = StartSoapCorrection(ConflictingSession(), RacingCorrections(winner), env.reports)

    result = _run(use_case.execute(StartSoapCorrectionCommand(report_id=env.report.id)))

    assert result.id == winner.id


def test_update_replaces_a_section_text_and_citations() -> None:
    env = _env()
    correction = _started(env)
    note = correction.notes[0]

    _run(
        env.update().execute(
            UpdateCorrectedNoteCommand(
                correction_id=correction.id,
                note_id=note.id,
                plan=[_grounded_claim(env, "Start amlodipine.")],
            )
        )
    )

    assert [claim.text for claim in note.plan] == ["Start amlodipine."]
    assert note.plan[0].citations[0].turn_id == env.dialogue.turns[1].id
    assert note.subjective == []  # the section replace cleared the untouched lists


def test_update_can_change_the_icd_coding() -> None:
    env = _env()
    correction = _started(env)
    note = correction.notes[0]
    recoded = AssessmentClaim(
        id=Id.new(),
        text="Migraine.",
        citations=[TurnCitation(turn_id=env.dialogue.turns[0].id)],
        icd=_NEW_ICD,
    )

    _run(
        env.update().execute(
            UpdateCorrectedNoteCommand(
                correction_id=correction.id, note_id=note.id, assessment=[recoded]
            )
        )
    )

    assert note.assessment[0].icd == _NEW_ICD


def test_add_note_has_no_source_lineage_and_joins_the_correction() -> None:
    env = _env()
    correction = _started(env)

    added = _run(
        env.add().execute(
            AddCorrectedNoteCommand(
                correction_id=correction.id, objective=[_grounded_claim(env, "BP 140/90.")]
            )
        )
    )

    assert added.source_note_id is None
    assert added in correction.notes


def test_delete_removes_a_note_from_the_correction() -> None:
    env = _env()
    correction = _started(env)
    note_id = correction.notes[0].id

    _run(
        env.delete().execute(
            DeleteCorrectedNoteCommand(correction_id=correction.id, note_id=note_id)
        )
    )

    assert note_id not in [note.id for note in correction.notes]


def test_edits_persist_without_auto_verifying() -> None:
    env = _env()
    correction = _started(env)
    saves_before = env.corrections.saves

    _run(
        env.update().execute(
            UpdateCorrectedNoteCommand(
                correction_id=correction.id,
                note_id=correction.notes[0].id,
                plan=[_grounded_claim(env, "Order ECG.")],
            )
        )
    )

    reloaded = _run(env.corrections.get(correction.id))
    assert reloaded is not None
    assert reloaded.status is CorrectionStatus.DRAFT
    assert env.corrections.saves == saves_before + 1
    assert env.session.commits == 0  # the caller, not the use case, owns the commit


def test_verify_stamps_the_doctor_and_time() -> None:
    env = _env()
    correction = _started(env)

    verified = _run(
        env.verify().execute(
            VerifySoapCorrectionCommand(correction_id=correction.id, doctor_id="dr-house")
        )
    )

    assert verified.status is CorrectionStatus.VERIFIED
    assert verified.verified_by == "dr-house"
    assert verified.verified_at is not None


def test_verified_correction_rejects_further_edits() -> None:
    env = _env()
    correction = _started(env)
    _run(env.verify().execute(VerifySoapCorrectionCommand(correction.id, doctor_id="dr-house")))

    with pytest.raises(CorrectionNotEditable):
        _run(
            env.add().execute(
                AddCorrectedNoteCommand(
                    correction_id=correction.id, plan=[_grounded_claim(env, "Late thought.")]
                )
            )
        )


def test_reopen_returns_to_draft_and_re_enables_edits() -> None:
    env = _env()
    correction = _started(env)
    _run(env.verify().execute(VerifySoapCorrectionCommand(correction.id, doctor_id="dr-house")))
    _run(env.reopen().execute(ReopenSoapCorrectionCommand(correction_id=correction.id)))

    added = _run(
        env.add().execute(
            AddCorrectedNoteCommand(
                correction_id=correction.id, objective=[_grounded_claim(env, "BP 120/80.")]
            )
        )
    )

    assert correction.status is CorrectionStatus.DRAFT
    assert added in correction.notes


def test_mutating_a_missing_correction_is_rejected() -> None:
    env = _env()

    with pytest.raises(CorrectionNotFoundError):
        _run(env.verify().execute(VerifySoapCorrectionCommand(Id.new(), doctor_id="dr-house")))


def test_updating_a_note_absent_from_the_correction_is_rejected() -> None:
    env = _env()
    correction = _started(env)

    with pytest.raises(NoteNotInCorrection):
        _run(
            env.update().execute(
                UpdateCorrectedNoteCommand(
                    correction_id=correction.id,
                    note_id=Id.new(),
                    subjective=[_grounded_claim(env, "Orphan edit.")],
                )
            )
        )


def test_adding_a_note_with_an_ungrounded_citation_is_rejected() -> None:
    env = _env()
    correction = _started(env)
    stray = SoapClaim(id=Id.new(), text="Made up.", citations=[TurnCitation(turn_id=Id.new())])

    with pytest.raises(CitationNotInSourceDialogue):
        _run(
            env.add().execute(
                AddCorrectedNoteCommand(correction_id=correction.id, subjective=[stray])
            )
        )


def test_updating_a_note_with_an_ungrounded_citation_is_rejected() -> None:
    env = _env()
    correction = _started(env)
    stray = SoapClaim(id=Id.new(), text="Made up.", citations=[TurnCitation(turn_id=Id.new())])

    with pytest.raises(CitationNotInSourceDialogue):
        _run(
            env.update().execute(
                UpdateCorrectedNoteCommand(
                    correction_id=correction.id, note_id=correction.notes[0].id, plan=[stray]
                )
            )
        )
