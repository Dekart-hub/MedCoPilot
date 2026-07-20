"""Integration tests: a SoapReportCorrection round-trips through Postgres.

Requires a reachable database via ``DATABASE_URL`` (skipped otherwise, as in
``tests/test_soap_repository.py``). Migrations run first; the dialogue and its
source report are saved so the correction's ``source_report_id`` FK holds, then
the correction is saved in one session and loaded in a fresh one to prove the
doctor's draft — original, edited and added notes, their lineage, citations, ICD
coding, verification state and timestamps — truly round-trips the database, and
that the original ``soap_report`` rows are left untouched.
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
_CREATED = datetime(2026, 7, 19, 9, 0, tzinfo=UTC)
_VERIFIED_AT = datetime(2026, 7, 19, 10, 30, tzinfo=UTC)


def _dialogue_and_source() -> tuple[Dialogue, SoapReport]:
    dialogue = Dialogue.start()
    first = dialogue.add_turn("patient", "I've had a headache for three days.")
    second = dialogue.add_turn("doctor", "Blood pressure is 140 over 90.")
    quoted = TurnCitation(turn_id=first.id, quote="headache")
    bare = TurnCitation(turn_id=second.id, quote=None)
    coded = SoapNote(
        id=Id.new(),
        subjective=[SoapClaim(id=Id.new(), text="Headache for three days.", citations=[quoted])],
        objective=[SoapClaim(id=Id.new(), text="BP 140/90.", citations=[bare])],
        assessment=[
            AssessmentClaim(id=Id.new(), text="Hypertension.", citations=[quoted, bare], icd=_ICD)
        ],
        plan=[SoapClaim(id=Id.new(), text="Start lisinopril.", citations=[bare])],
        confidence=0.91,
    )
    other = SoapNote(
        id=Id.new(),
        subjective=[SoapClaim(id=Id.new(), text="Also reports fatigue.", citations=[quoted])],
    )
    return dialogue, SoapReport(id=Id.new(), notes=[coded, other])


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


async def _get_correction(correction: SoapReportCorrection) -> SoapReportCorrection | None:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        return await SqlAlchemySoapReportCorrectionRepository(session).get(correction.id)


def test_draft_with_original_edited_and_added_notes_round_trips() -> None:
    run_migrations()
    dialogue, report = _dialogue_and_source()
    turn = dialogue.turns[0]
    correction = SoapReportCorrection.start(report, created_at=_CREATED)
    correction.update_note(
        correction.notes[1].id,
        at=_CREATED,
        subjective=[
            SoapClaim(
                id=Id.new(),
                text="Now denies fatigue.",
                citations=[TurnCitation(turn_id=turn.id, quote=None)],
            )
        ],
    )
    added = correction.add_note(
        at=_CREATED,
        plan=[
            SoapClaim(
                id=Id.new(),
                text="Order ECG.",
                citations=[TurnCitation(turn_id=turn.id, quote="ECG")],
            )
        ],
    )

    async def _round_trip() -> SoapReportCorrection | None:
        await _save_source(dialogue, report)
        await _save_correction(correction)
        loaded = await _get_correction(correction)
        await dispose_engine()
        return loaded

    loaded = asyncio.run(_round_trip())

    assert loaded is not None
    assert [note.source_note_id for note in loaded.notes] == [
        report.notes[0].id,
        report.notes[1].id,
        None,
    ]
    original = loaded.notes[0]
    assert original.assessment[0].icd == _ICD
    assert [(c.turn_id, c.quote) for c in original.assessment[0].citations] == [
        (dialogue.turns[0].id, "headache"),
        (dialogue.turns[1].id, None),
    ]
    assert [c.text for c in loaded.notes[1].subjective] == ["Now denies fatigue."]
    assert loaded.notes[2].id == added.id
    assert [c.text for c in loaded.notes[2].plan] == ["Order ECG."]


def test_verified_correction_round_trips_status_doctor_and_timestamps() -> None:
    run_migrations()
    dialogue, report = _dialogue_and_source()
    correction = SoapReportCorrection.start(report, created_at=_CREATED)
    correction.verify("dr-house", at=_VERIFIED_AT)

    async def _round_trip() -> SoapReportCorrection | None:
        await _save_source(dialogue, report)
        await _save_correction(correction)
        loaded = await _get_correction(correction)
        await dispose_engine()
        return loaded

    loaded = asyncio.run(_round_trip())

    assert loaded is not None
    assert loaded.status == correction.status
    assert loaded.verified_by == "dr-house"
    assert loaded.verified_at == _VERIFIED_AT
    assert loaded.created_at == _CREATED
    assert loaded.updated_at == _VERIFIED_AT


def test_deleting_a_note_is_reflected_after_round_trip() -> None:
    run_migrations()
    dialogue, report = _dialogue_and_source()
    correction = SoapReportCorrection.start(report, created_at=_CREATED)
    correction.delete_note(correction.notes[0].id, at=_CREATED)

    async def _round_trip() -> SoapReportCorrection | None:
        await _save_source(dialogue, report)
        await _save_correction(correction)
        loaded = await _get_correction(correction)
        await dispose_engine()
        return loaded

    loaded = asyncio.run(_round_trip())

    assert loaded is not None
    assert [note.source_note_id for note in loaded.notes] == [report.notes[1].id]


def test_second_correction_for_the_same_source_report_is_rejected() -> None:
    run_migrations()
    dialogue, report = _dialogue_and_source()
    first = SoapReportCorrection.start(report, created_at=_CREATED)
    second = SoapReportCorrection.start(report, created_at=_CREATED)

    async def _save_both() -> None:
        await _save_source(dialogue, report)
        await _save_correction(first)
        await _save_correction(second)

    try:
        with pytest.raises(IntegrityError):
            asyncio.run(_save_both())
    finally:
        asyncio.run(dispose_engine())


def test_saving_a_correction_leaves_the_source_report_untouched() -> None:
    run_migrations()
    dialogue, report = _dialogue_and_source()
    correction = SoapReportCorrection.start(report, created_at=_CREATED)
    correction.delete_note(correction.notes[1].id, at=_CREATED)
    correction.update_note(
        correction.notes[0].id,
        at=_CREATED,
        subjective=[
            SoapClaim(
                id=Id.new(),
                text="Overwritten.",
                citations=[TurnCitation(turn_id=dialogue.turns[0].id, quote=None)],
            )
        ],
    )

    async def _save_then_reload_source() -> SoapReport | None:
        await _save_source(dialogue, report)
        await _save_correction(correction)
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
            loaded = await SqlAlchemySoapReportRepository(session).get(report.id)
        await dispose_engine()
        return loaded

    loaded = asyncio.run(_save_then_reload_source())

    assert loaded is not None
    assert [note.id for note in loaded.notes] == [report.notes[0].id, report.notes[1].id]
    assert [c.text for c in loaded.notes[0].subjective] == ["Headache for three days."]
    assert loaded.notes[0].assessment[0].icd == _ICD
