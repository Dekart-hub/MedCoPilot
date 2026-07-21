"""Integration tests: an LLM editor session round-trips through Postgres.

Requires a reachable database via ``DATABASE_URL`` (skipped otherwise, as in
``tests/test_soap_correction_repository.py``). Migrations run first; the
dialogue, its source report and the correction are saved so the session's
``correction_id`` FK holds, then a session carrying one proposal — an add, an
update and a delete, each with its own decision — is saved in one session and
loaded in a fresh one to prove operation ordering, proposed content,
before-snapshots (ICD included), fingerprints and per-operation decisions all
survive the database.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError

from dialogue.dialogue import Dialogue
from dialogue.sqlalchemy_repository import SqlAlchemyDialogueRepository
from infra.db import dispose_engine, get_sessionmaker
from infra.migrations import run_migrations
from shared.value_objects import Id
from soap.correction import SoapReportCorrection
from soap.correction_sqlalchemy_repository import SqlAlchemySoapReportCorrectionRepository
from soap.proposal import (
    AddNoteOperation,
    CorrectionEditorSession,
    DeleteNoteOperation,
    OperationDecision,
    OperationType,
    ProposalStatus,
    ProposedClaim,
    ProposedNote,
    UpdateNoteOperation,
)
from soap.proposal_sqlalchemy_repository import SqlAlchemyCorrectionEditorSessionRepository
from soap.soap import (
    AssessmentClaim,
    IcdCoding,
    SoapClaim,
    SoapNote,
    SoapReport,
    TurnCitation,
)
from soap.sqlalchemy_repository import SqlAlchemySoapReportRepository

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not configured; skipping DB integration test",
)

_ICD = IcdCoding(
    code="I10",
    name="Essential (primary) hypertension",
    classifier_url="https://icd.who.int/browse10/2019/en#/I10",
)
_CREATED = datetime(2026, 7, 21, 9, 0, tzinfo=UTC)


def _dialogue_and_source() -> tuple[Dialogue, SoapReport]:
    dialogue = Dialogue.start()
    turn = dialogue.add_turn("patient", "I've had a headache for three days.")
    cite = TurnCitation(turn_id=turn.id, quote="headache")
    coded = SoapNote(
        id=Id.new(),
        subjective=[SoapClaim(id=Id.new(), text="Headache for three days.", citations=[cite])],
        assessment=[AssessmentClaim(id=Id.new(), text="Hypertension.", citations=[cite], icd=_ICD)],
    )
    other = SoapNote(
        id=Id.new(),
        subjective=[SoapClaim(id=Id.new(), text="Also reports fatigue.", citations=[cite])],
    )
    return dialogue, SoapReport(id=Id.new(), notes=[coded, other])


def _proposed(text: str) -> ProposedNote:
    return ProposedNote(
        plan=[ProposedClaim(text=text, citations=[TurnCitation(turn_id=Id.new(), quote="ECG")])]
    )


async def _save_source(dialogue: Dialogue, report: SoapReport) -> None:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        await SqlAlchemyDialogueRepository(session).save(dialogue)
        await SqlAlchemySoapReportRepository(session).save(
            report, dialogue_id=dialogue.id, created_at=_CREATED
        )
        await session.commit()


async def _save_correction(correction: SoapReportCorrection) -> None:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        await SqlAlchemySoapReportCorrectionRepository(session).save(correction)
        await session.commit()


async def _save_editor_session(editor: CorrectionEditorSession) -> None:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        await SqlAlchemyCorrectionEditorSessionRepository(session).save(editor)
        await session.commit()


async def _get_editor_session(editor: CorrectionEditorSession) -> CorrectionEditorSession | None:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        return await SqlAlchemyCorrectionEditorSessionRepository(session).get(editor.id)


def _decided_session() -> tuple[
    Dialogue, SoapReport, SoapReportCorrection, CorrectionEditorSession
]:
    dialogue, report = _dialogue_and_source()
    correction = SoapReportCorrection.start(report, created_at=_CREATED)
    editor = CorrectionEditorSession.start(correction.id, created_at=_CREATED)
    proposal = editor.propose(
        correction,
        user_request="Order an ECG, tighten note one, drop the fatigue note.",
        model_id="medgemma-27b",
        prompt_version="v1",
        operations=[
            AddNoteOperation(_proposed("Order ECG.")),
            UpdateNoteOperation(correction.notes[0].id, _proposed("Tighter plan.")),
            DeleteNoteOperation(correction.notes[1].id),
        ],
        at=_CREATED,
    )
    add_op, update_op, delete_op = proposal.operations
    proposal.accept_operation(add_op.id, correction, at=_CREATED)
    proposal.reject_operation(update_op.id, at=_CREATED, reason="wording is fine")
    proposal.accept_operation(delete_op.id, correction, at=_CREATED)
    return dialogue, report, correction, editor


def test_editor_session_round_trips_operations_decisions_and_snapshots() -> None:
    run_migrations()
    dialogue, report, correction, editor = _decided_session()

    async def _round_trip() -> CorrectionEditorSession | None:
        await _save_source(dialogue, report)
        await _save_correction(correction)
        await _save_editor_session(editor)
        loaded = await _get_editor_session(editor)
        await dispose_engine()
        return loaded

    loaded = asyncio.run(_round_trip())

    assert loaded is not None
    assert loaded.correction_id == correction.id
    assert len(loaded.proposals) == 1
    proposal = loaded.proposals[0]
    assert proposal.base_correction_revision == 1
    assert proposal.model_id == "medgemma-27b"
    assert [op.type for op in proposal.operations] == [
        OperationType.ADD_NOTE,
        OperationType.UPDATE_NOTE,
        OperationType.DELETE_NOTE,
    ]
    add_op, update_op, delete_op = proposal.operations
    assert add_op.decision is OperationDecision.ACCEPTED
    assert add_op.before is None and add_op.target_fingerprint is None
    assert add_op.proposed is not None
    assert [c.text for c in add_op.proposed.plan] == ["Order ECG."]
    assert update_op.decision is OperationDecision.REJECTED
    assert update_op.decision_reason == "wording is fine"
    assert update_op.before is not None
    assert update_op.before.assessment[0].icd == _ICD
    assert update_op.target_fingerprint is not None
    assert delete_op.decision is OperationDecision.ACCEPTED
    assert delete_op.before is not None
    assert [c.text for c in delete_op.before.subjective] == ["Also reports fatigue."]
    assert proposal.status() is ProposalStatus.MIXED


def test_a_second_session_for_the_same_correction_is_rejected() -> None:
    run_migrations()
    dialogue, report = _dialogue_and_source()
    correction = SoapReportCorrection.start(report, created_at=_CREATED)
    first = CorrectionEditorSession.start(correction.id, created_at=_CREATED)
    second = CorrectionEditorSession.start(correction.id, created_at=_CREATED)

    async def _save_both() -> None:
        await _save_source(dialogue, report)
        await _save_correction(correction)
        await _save_editor_session(first)
        await _save_editor_session(second)

    try:
        with pytest.raises(IntegrityError):
            asyncio.run(_save_both())
    finally:
        asyncio.run(dispose_engine())


def test_correction_revision_survives_the_round_trip() -> None:
    run_migrations()
    dialogue, report = _dialogue_and_source()
    correction = SoapReportCorrection.start(report, created_at=_CREATED)
    correction.delete_note(correction.notes[0].id, at=_CREATED)

    async def _round_trip() -> SoapReportCorrection | None:
        await _save_source(dialogue, report)
        await _save_correction(correction)
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
            loaded = await SqlAlchemySoapReportCorrectionRepository(session).get(correction.id)
        await dispose_engine()
        return loaded

    loaded = asyncio.run(_round_trip())

    assert loaded is not None
    assert loaded.revision == correction.revision == 2
