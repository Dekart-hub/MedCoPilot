from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

import pytest

import ehr.dispatcher as dispatcher_module
import ehr.publication_use_cases as publication_use_cases_module
from dialogue.dialogue import Dialogue
from dialogue.sqlalchemy_repository import SqlAlchemyDialogueRepository
from ehr.dispatcher import PublicationDispatcher
from ehr.fhir import FhirDeliveryResult, FhirGatewayError, FhirPublicationGateway
from ehr.publication import EhrPublication, PublicationStatus
from ehr.publication_sqlalchemy_repository import (
    SqlAlchemyEhrPublicationRepository,
    SqlAlchemyPublicationOutboxRepository,
)
from ehr.publication_use_cases import (
    RequestEhrPublication,
    RequestEhrPublicationCommand,
)
from infra.db import dispose_engine, get_sessionmaker
from infra.migrations import run_migrations
from shared.value_objects import Id
from soap.correction import CorrectionStatus, SoapReportCorrection
from soap.correction_sqlalchemy_repository import (
    SqlAlchemySoapReportCorrectionRepository,
)
from soap.soap import SoapReport
from soap.sqlalchemy_repository import SqlAlchemySoapReportRepository

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not configured; skipping DB integration test",
)

_NOW = datetime(2001, 2, 3, 4, 5, tzinfo=UTC)


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz: object = None) -> datetime:
        return _NOW if tz is not None else _NOW.replace(tzinfo=None)


class _FailingGateway(FhirPublicationGateway):
    async def deliver(self, publication: EhrPublication) -> FhirDeliveryResult:
        raise FhirGatewayError("mock EHR unavailable")


class _SuccessfulGateway(FhirPublicationGateway):
    async def deliver(self, publication: EhrPublication) -> FhirDeliveryResult:
        return FhirDeliveryResult("Bundle/e2e-publication", "3")


def test_verify_publish_retry_restart_and_deliver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_migrations()
    monkeypatch.setattr(publication_use_cases_module, "_now", lambda: _NOW)
    monkeypatch.setattr(dispatcher_module, "datetime", _FixedDateTime)

    dialogue = Dialogue.start()
    dialogue.add_turn("patient", "Headache")
    report = SoapReport(id=Id.new())
    correction = SoapReportCorrection.start(report, created_at=_NOW)
    correction.verify("doctor-1", at=_NOW)

    async def exercise() -> None:
        try:
            async with get_sessionmaker()() as session:
                await SqlAlchemyDialogueRepository(session).save(dialogue)
                await SqlAlchemySoapReportRepository(session).save(
                    report,
                    dialogue_id=dialogue.id,
                    created_at=_NOW,
                )
                await SqlAlchemySoapReportCorrectionRepository(session).save(correction)
                await session.commit()

            async with get_sessionmaker()() as session:
                request = RequestEhrPublication(
                    session,
                    SqlAlchemySoapReportCorrectionRepository(session),
                    SqlAlchemySoapReportRepository(session),
                    SqlAlchemyDialogueRepository(session),
                    SqlAlchemyEhrPublicationRepository(session),
                    SqlAlchemyPublicationOutboxRepository(session),
                )
                accepted = await request.execute(
                    RequestEhrPublicationCommand(
                        report_id=report.id,
                        patient_ref="Patient/p1",
                        encounter_ref="Encounter/e1",
                        author_ref="Practitioner/d1",
                    )
                )
                await session.commit()

            assert accepted.publication.status is PublicationStatus.PENDING
            async with get_sessionmaker()() as session:
                accepted_correction = await SqlAlchemySoapReportCorrectionRepository(session).get(
                    correction.id
                )
            assert accepted_correction is not None
            assert accepted_correction.status is CorrectionStatus.PUBLICATION_PENDING

            first_dispatcher = PublicationDispatcher(
                get_sessionmaker(),
                _FailingGateway(),
                batch_size=1,
                poll_seconds=0,
                retry_initial_seconds=0,
                retry_max_seconds=0,
            )
            assert await first_dispatcher.run_once() == 1

            async with get_sessionmaker()() as session:
                pending = await SqlAlchemyEhrPublicationRepository(session).get_by_source_report_id(
                    report.id
                )
                pending_event = await SqlAlchemyPublicationOutboxRepository(
                    session
                ).get_by_publication_id(accepted.publication.id)
                pending_correction = await SqlAlchemySoapReportCorrectionRepository(session).get(
                    correction.id
                )

            assert pending is not None
            assert pending_event is not None
            assert pending_correction is not None
            assert pending.status is PublicationStatus.PENDING
            assert pending_event.attempt_count == 1
            assert pending_event.last_error == "mock EHR unavailable"
            assert pending_correction.status is CorrectionStatus.PUBLICATION_PENDING

            restarted_dispatcher = PublicationDispatcher(
                get_sessionmaker(),
                _SuccessfulGateway(),
                batch_size=1,
                poll_seconds=0,
                retry_initial_seconds=0,
                retry_max_seconds=0,
            )
            assert await restarted_dispatcher.run_once() == 1

            async with get_sessionmaker()() as session:
                delivered = await SqlAlchemyEhrPublicationRepository(
                    session
                ).get_by_source_report_id(report.id)
                delivered_event = await SqlAlchemyPublicationOutboxRepository(
                    session
                ).get_by_publication_id(accepted.publication.id)
                published_correction = await SqlAlchemySoapReportCorrectionRepository(session).get(
                    correction.id
                )

            assert delivered is not None
            assert delivered_event is not None
            assert published_correction is not None
            assert delivered.status is PublicationStatus.DELIVERED
            assert delivered.remote_reference == "Bundle/e2e-publication"
            assert delivered.remote_version == "3"
            assert delivered_event.attempt_count == 2
            assert delivered_event.delivered_at is not None
            assert delivered_event.last_error is None
            assert published_correction.status is CorrectionStatus.PUBLISHED
        finally:
            await dispose_engine()

    asyncio.run(exercise())
