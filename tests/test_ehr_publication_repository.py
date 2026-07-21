from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError

from dialogue.dialogue import Dialogue
from dialogue.sqlalchemy_repository import SqlAlchemyDialogueRepository
from ehr.publication import EhrPublication, PublicationOutbox
from ehr.publication_sqlalchemy_repository import (
    SqlAlchemyEhrPublicationRepository,
    SqlAlchemyPublicationOutboxRepository,
)
from infra.db import dispose_engine, get_sessionmaker
from infra.migrations import run_migrations
from shared.value_objects import Id
from soap.correction import CorrectionStatus, SoapReportCorrection
from soap.correction_sqlalchemy_repository import (
    SqlAlchemySoapReportCorrectionRepository,
)
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

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _aggregate() -> tuple[Dialogue, SoapReport, SoapReportCorrection]:
    dialogue = Dialogue.start()
    patient = dialogue.add_turn("patient", "Headache for three days")
    doctor = dialogue.add_turn(
        "doctor",
        "Neurological exam is normal; likely tension headache; maintain hydration",
    )
    report = SoapReport(
        id=Id.new(),
        notes=[
            SoapNote(
                id=Id.new(),
                subjective=[
                    SoapClaim(
                        id=Id.new(),
                        text="Headache for three days",
                        citations=[TurnCitation(patient.id, "Headache")],
                    )
                ],
                objective=[
                    SoapClaim(
                        id=Id.new(),
                        text="Neurological exam is normal",
                        citations=[TurnCitation(doctor.id, "exam is normal")],
                    )
                ],
                assessment=[
                    AssessmentClaim(
                        id=Id.new(),
                        text="Tension-type headache",
                        citations=[TurnCitation(doctor.id, "tension headache")],
                        icd=IcdCoding(
                            code="G44.2",
                            name="Tension-type headache",
                            classifier_url=("https://icd.who.int/browse10/2019/en#/G44.2"),
                        ),
                    )
                ],
                plan=[
                    SoapClaim(
                        id=Id.new(),
                        text="Maintain hydration",
                        citations=[TurnCitation(doctor.id, "maintain hydration")],
                    )
                ],
            )
        ],
    )
    correction = SoapReportCorrection.start(report, created_at=_NOW)
    correction.verify("doctor-1", at=_NOW)
    return dialogue, report, correction


async def _save_source(
    dialogue: Dialogue, report: SoapReport, correction: SoapReportCorrection
) -> None:
    async with get_sessionmaker()() as session:
        await SqlAlchemyDialogueRepository(session).save(dialogue)
        await SqlAlchemySoapReportRepository(session).save(
            report, dialogue_id=dialogue.id, created_at=_NOW
        )
        await SqlAlchemySoapReportCorrectionRepository(session).save(correction)
        await session.commit()


def test_publication_snapshot_and_outbox_round_trip() -> None:
    run_migrations()
    dialogue, report, correction = _aggregate()
    publication = EhrPublication.request(
        correction,
        dialogue,
        patient_ref="Patient/p1",
        encounter_ref="Encounter/e1",
        author_ref="Practitioner/d1",
        at=_NOW,
    )
    event = PublicationOutbox.for_publication(publication, at=_NOW)
    correction.begin_publication(at=_NOW)

    async def exercise() -> tuple[EhrPublication | None, PublicationOutbox | None]:
        async with get_sessionmaker()() as session:
            await SqlAlchemyDialogueRepository(session).save(dialogue)
            await SqlAlchemySoapReportRepository(session).save(
                report, dialogue_id=dialogue.id, created_at=_NOW
            )
            await SqlAlchemySoapReportCorrectionRepository(session).save(correction)
            await SqlAlchemyEhrPublicationRepository(session).save(publication)
            await SqlAlchemyPublicationOutboxRepository(session).save(event)
            await session.commit()
        async with get_sessionmaker()() as session:
            loaded_publication = await SqlAlchemyEhrPublicationRepository(session).get(
                publication.id
            )
            loaded_event = await SqlAlchemyPublicationOutboxRepository(
                session
            ).get_by_publication_id(publication.id)
        await dispose_engine()
        return loaded_publication, loaded_event

    loaded, loaded_event = asyncio.run(exercise())

    assert loaded is not None
    assert loaded.snapshot == publication.snapshot
    assert loaded.snapshot_hash == publication.snapshot_hash
    assert loaded.patient_ref == "Patient/p1"
    assert [turn.speaker for turn in loaded.snapshot.dialogue_turns] == [
        "patient",
        "doctor",
    ]
    assert loaded.snapshot.notes[0].subjective[0].citations[0].quote == "Headache"
    assert loaded.snapshot.notes[0].objective[0].text == "Neurological exam is normal"
    assert loaded.snapshot.notes[0].assessment[0].text == "Tension-type headache"
    assert loaded.snapshot.notes[0].assessment[0].icd is not None
    assert loaded.snapshot.notes[0].assessment[0].icd.code == "G44.2"
    assert loaded.snapshot.notes[0].plan[0].text == "Maintain hydration"
    assert loaded.snapshot.verified_by == "doctor-1"
    assert loaded.snapshot.verified_at == _NOW
    assert loaded_event is not None
    assert loaded_event.payload == event.payload
    assert loaded_event.payload_hash == event.payload_hash


def test_publication_transaction_rolls_back_correction_lock_and_outbox() -> None:
    run_migrations()
    dialogue, report, correction = _aggregate()

    async def exercise() -> tuple[CorrectionStatus, EhrPublication | None]:
        await _save_source(dialogue, report, correction)
        async with get_sessionmaker()() as session:
            corrections = SqlAlchemySoapReportCorrectionRepository(session)
            stored = await corrections.get_by_source_report_id_for_update(report.id)
            assert stored is not None
            publication = EhrPublication.request(
                stored,
                dialogue,
                patient_ref="Patient/p1",
                encounter_ref="Encounter/e1",
                author_ref="Practitioner/d1",
                at=_NOW,
            )
            event = PublicationOutbox.for_publication(publication, at=_NOW)
            stored.begin_publication(at=_NOW)
            await corrections.save(stored)
            await SqlAlchemyEhrPublicationRepository(session).save(publication)
            await SqlAlchemyPublicationOutboxRepository(session).save(event)
            await session.flush()
            await session.rollback()
        async with get_sessionmaker()() as session:
            stored = await SqlAlchemySoapReportCorrectionRepository(
                session
            ).get_by_source_report_id(report.id)
            assert stored is not None
            missing = await SqlAlchemyEhrPublicationRepository(session).get_by_source_report_id(
                report.id
            )
        await dispose_engine()
        return stored.status, missing

    status, missing = asyncio.run(exercise())

    assert status is CorrectionStatus.VERIFIED
    assert missing is None


def test_outbox_claim_skips_a_row_locked_by_another_dispatcher() -> None:
    run_migrations()
    dialogue, report, correction = _aggregate()
    publication = EhrPublication.request(
        correction,
        dialogue,
        patient_ref="Patient/p1",
        encounter_ref="Encounter/e1",
        author_ref="Practitioner/d1",
        at=_NOW,
    )
    event = PublicationOutbox.for_publication(publication, at=_NOW)
    correction.begin_publication(at=_NOW)

    async def exercise() -> tuple[int, int]:
        async with get_sessionmaker()() as setup:
            await SqlAlchemyDialogueRepository(setup).save(dialogue)
            await SqlAlchemySoapReportRepository(setup).save(
                report, dialogue_id=dialogue.id, created_at=_NOW
            )
            await SqlAlchemySoapReportCorrectionRepository(setup).save(correction)
            await SqlAlchemyEhrPublicationRepository(setup).save(publication)
            await SqlAlchemyPublicationOutboxRepository(setup).save(event)
            await setup.commit()

        first_session = get_sessionmaker()()
        second_session = get_sessionmaker()()
        try:
            first = await SqlAlchemyPublicationOutboxRepository(first_session).claim_due(
                now=_NOW, limit=1_000
            )
            second = await SqlAlchemyPublicationOutboxRepository(second_session).claim_due(
                now=_NOW, limit=1_000
            )
            await second_session.rollback()
            await first_session.rollback()
            return len(first), len(second)
        finally:
            await first_session.close()
            await second_session.close()
            await dispose_engine()

    first_count, second_count = asyncio.run(exercise())

    assert first_count >= 1
    assert second_count == 0


def test_database_uniqueness_rejects_a_second_publication() -> None:
    run_migrations()
    dialogue, report, correction = _aggregate()
    first = EhrPublication.request(
        correction,
        dialogue,
        patient_ref="Patient/p1",
        encounter_ref="Encounter/e1",
        author_ref="Practitioner/d1",
        at=_NOW,
    )
    second = EhrPublication.request(
        correction,
        dialogue,
        patient_ref="Patient/p1",
        encounter_ref="Encounter/e1",
        author_ref="Practitioner/d1",
        at=_NOW,
    )
    correction.begin_publication(at=_NOW)

    async def exercise() -> None:
        async with get_sessionmaker()() as session:
            await SqlAlchemyDialogueRepository(session).save(dialogue)
            await SqlAlchemySoapReportRepository(session).save(
                report, dialogue_id=dialogue.id, created_at=_NOW
            )
            await SqlAlchemySoapReportCorrectionRepository(session).save(correction)
            await SqlAlchemyEhrPublicationRepository(session).save(first)
            await session.commit()
        async with get_sessionmaker()() as session:
            await SqlAlchemyEhrPublicationRepository(session).save(second)
            with pytest.raises(IntegrityError):
                await session.commit()
        await dispose_engine()

    asyncio.run(exercise())
