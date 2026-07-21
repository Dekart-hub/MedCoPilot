"""Application use cases for accepting and inspecting EHR publications."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from dialogue.repository import DialogueRepository
from soap.correction import SoapReportCorrection
from soap.correction_repository import SoapReportCorrectionRepository
from soap.repository import SoapReportRepository
from soap.soap import SoapReportId

from .publication import EhrPublication, PublicationOutbox
from .publication_repository import (
    EhrPublicationRepository,
    PublicationOutboxRepository,
)


class PublicationNotFoundError(Exception):
    def __init__(self, report_id: SoapReportId) -> None:
        super().__init__(f"soap report {report_id} has no EHR publication")
        self.report_id = report_id


class PublicationCorrectionNotFoundError(Exception):
    def __init__(self, report_id: SoapReportId) -> None:
        super().__init__(f"soap report {report_id} has no correction")
        self.report_id = report_id


class PublicationSourceReportNotFoundError(Exception):
    def __init__(self, report_id: SoapReportId) -> None:
        super().__init__(f"soap report {report_id} not found")
        self.report_id = report_id


class PublicationSourceDialogueNotFoundError(Exception):
    def __init__(self, report_id: SoapReportId) -> None:
        super().__init__(f"source dialogue for soap report {report_id} not found")
        self.report_id = report_id


class PublicationOutboxNotFoundError(Exception):
    def __init__(self, publication: EhrPublication) -> None:
        super().__init__(f"publication {publication.id} has no outbox event")
        self.publication = publication


@dataclass(frozen=True, slots=True)
class RequestEhrPublicationCommand:
    report_id: SoapReportId
    patient_ref: str
    encounter_ref: str
    author_ref: str


@dataclass(frozen=True, slots=True)
class PublicationDelivery:
    publication: EhrPublication
    outbox: PublicationOutbox


def _now() -> datetime:
    return datetime.now(UTC)


class RequestEhrPublication:
    def __init__(
        self,
        session: AsyncSession,
        corrections: SoapReportCorrectionRepository,
        reports: SoapReportRepository,
        dialogues: DialogueRepository,
        publications: EhrPublicationRepository,
        outbox: PublicationOutboxRepository,
    ) -> None:
        self._session = session
        self._corrections = corrections
        self._reports = reports
        self._dialogues = dialogues
        self._publications = publications
        self._outbox = outbox

    async def execute(self, command: RequestEhrPublicationCommand) -> PublicationDelivery:
        existing = await self._publications.get_by_source_report_id(command.report_id)
        if existing is not None:
            return await self._delivery(existing)

        correction = await self._corrections.get_by_source_report_id_for_update(command.report_id)
        if correction is None:
            raise PublicationCorrectionNotFoundError(command.report_id)

        existing = await self._publications.get_by_source_report_id(command.report_id)
        if existing is not None:
            return await self._delivery(existing)

        dialogue_id = await self._reports.get_dialogue_id(command.report_id)
        if dialogue_id is None:
            raise PublicationSourceReportNotFoundError(command.report_id)
        dialogue = await self._dialogues.get(dialogue_id)
        if dialogue is None:
            raise PublicationSourceDialogueNotFoundError(command.report_id)

        at = _now()
        publication = EhrPublication.request(
            correction,
            dialogue,
            patient_ref=command.patient_ref,
            encounter_ref=command.encounter_ref,
            author_ref=command.author_ref,
            at=at,
        )
        event = PublicationOutbox.for_publication(publication, at=at)
        correction.begin_publication(at=at)
        return await self._create(correction, publication, event)

    async def _create(
        self,
        correction: SoapReportCorrection,
        publication: EhrPublication,
        event: PublicationOutbox,
    ) -> PublicationDelivery:
        try:
            await self._corrections.save(correction)
            await self._publications.save(publication)
            await self._outbox.save(event)
            await self._session.flush()
        except IntegrityError:
            await self._session.rollback()
            winner = await self._publications.get_by_source_report_id(publication.source_report_id)
            if winner is None:
                raise
            return await self._delivery(winner)
        return PublicationDelivery(publication=publication, outbox=event)

    async def _delivery(self, publication: EhrPublication) -> PublicationDelivery:
        event = await self._outbox.get_by_publication_id(publication.id)
        if event is None:
            raise PublicationOutboxNotFoundError(publication)
        return PublicationDelivery(publication=publication, outbox=event)


class GetEhrPublication:
    def __init__(
        self,
        publications: EhrPublicationRepository,
        outbox: PublicationOutboxRepository,
    ) -> None:
        self._publications = publications
        self._outbox = outbox

    async def execute(self, report_id: SoapReportId) -> PublicationDelivery:
        publication = await self._publications.get_by_source_report_id(report_id)
        if publication is None:
            raise PublicationNotFoundError(report_id)
        event = await self._outbox.get_by_publication_id(publication.id)
        if event is None:
            raise PublicationOutboxNotFoundError(publication)
        return PublicationDelivery(publication=publication, outbox=event)
