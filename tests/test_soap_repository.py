"""Integration test: a SoapReport survives a save/get round-trip through Postgres.

Requires a reachable database via ``DATABASE_URL`` (skipped otherwise, as in
``tests/test_dialogue_repository.py``). Migrations run first; the dialogue and
its report are saved in one session and the report is loaded in a fresh one to
prove the full S/O/A/P shape — citations, ICD coding and note confidence — truly
round-trips the database.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

import pytest

from dialogue.dialogue import Dialogue
from dialogue.sqlalchemy_repository import SqlAlchemyDialogueRepository
from infra.db import dispose_engine, get_sessionmaker
from infra.migrations import run_migrations
from shared.value_objects import Id
from soap.orm import SoapReportRow
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

_CREATED_AT = datetime(2026, 7, 20, 9, 30, tzinfo=UTC)

_ICD = IcdCoding(
    code="I10",
    name="Essential (primary) hypertension",
    classifier_url="https://icd.who.int/browse10/2019/en#/I10",
)


def _dialogue_and_report() -> tuple[Dialogue, SoapReport]:
    dialogue = Dialogue.start()
    first = dialogue.add_turn("patient", "I've had a headache for three days.")
    second = dialogue.add_turn("doctor", "Blood pressure is 140 over 90.")
    quoted = TurnCitation(turn_id=first.id, quote="headache")
    bare = TurnCitation(turn_id=second.id, quote=None)
    note = SoapNote(
        id=Id.new(),
        subjective=[SoapClaim(id=Id.new(), text="Headache for three days.", citations=[quoted])],
        objective=[SoapClaim(id=Id.new(), text="BP 140/90.", citations=[bare])],
        assessment=[
            AssessmentClaim(id=Id.new(), text="Hypertension.", citations=[quoted, bare], icd=_ICD)
        ],
        plan=[SoapClaim(id=Id.new(), text="Start lisinopril.", citations=[bare])],
        confidence=0.91,
    )
    return dialogue, SoapReport(id=Id.new(), notes=[note])


async def _save_then_get(dialogue: Dialogue, report: SoapReport) -> SoapReport | None:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        await SqlAlchemyDialogueRepository(session).save(dialogue)
        await SqlAlchemySoapReportRepository(session).save(
            report, dialogue_id=dialogue.id, created_at=_CREATED_AT
        )
        await session.commit()
    async with sessionmaker() as session:
        loaded = await SqlAlchemySoapReportRepository(session).get(report.id)
    await dispose_engine()
    return loaded


def test_report_round_trips_through_the_database() -> None:
    run_migrations()
    dialogue, report = _dialogue_and_report()

    loaded = asyncio.run(_save_then_get(dialogue, report))

    assert loaded is not None
    assert loaded.id == report.id
    note = loaded.notes[0]
    assert note.confidence == 0.91
    assert [claim.text for claim in note.subjective] == ["Headache for three days."]
    assert [claim.text for claim in note.plan] == ["Start lisinopril."]

    assessment = note.assessment[0]
    assert assessment.icd == _ICD
    assert [(c.turn_id, c.quote) for c in assessment.citations] == [
        (dialogue.turns[0].id, "headache"),
        (dialogue.turns[1].id, None),
    ]


def test_get_by_dialogue_id_reads_the_persisted_report() -> None:
    run_migrations()
    dialogue, report = _dialogue_and_report()

    async def _save_then_get_by_dialogue() -> SoapReport | None:
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
            await SqlAlchemyDialogueRepository(session).save(dialogue)
            await SqlAlchemySoapReportRepository(session).save(
                report, dialogue_id=dialogue.id, created_at=_CREATED_AT
            )
            await session.commit()
        async with sessionmaker() as session:
            loaded = await SqlAlchemySoapReportRepository(session).get_by_dialogue_id(dialogue.id)
        await dispose_engine()
        return loaded

    loaded = asyncio.run(_save_then_get_by_dialogue())

    assert loaded is not None
    assert loaded.id == report.id


def test_created_at_is_persisted_and_read_back() -> None:
    run_migrations()
    dialogue, report = _dialogue_and_report()

    async def _save_then_read_created_at() -> datetime | None:
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
            await SqlAlchemyDialogueRepository(session).save(dialogue)
            await SqlAlchemySoapReportRepository(session).save(
                report, dialogue_id=dialogue.id, created_at=_CREATED_AT
            )
            await session.commit()
        async with sessionmaker() as session:
            stored = await session.get(SoapReportRow, report.id.value)
            value = stored.created_at if stored is not None else None
        await dispose_engine()
        return value

    loaded = asyncio.run(_save_then_read_created_at())

    assert loaded == _CREATED_AT
