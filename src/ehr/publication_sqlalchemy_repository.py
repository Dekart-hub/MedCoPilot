"""SQLAlchemy adapters for EHR publications and delivery outbox events."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.value_objects import Id
from soap.correction import CorrectionId
from soap.soap import SoapReportId

from .publication import (
    EhrPublication,
    PublicationId,
    PublicationOutbox,
    PublicationOutboxId,
    PublicationStatus,
    snapshot_from_dict,
    snapshot_to_dict,
)
from .publication_orm import EhrPublicationRow, PublicationOutboxRow
from .publication_repository import (
    EhrPublicationRepository,
    PublicationOutboxRepository,
)


class SqlAlchemyEhrPublicationRepository(EhrPublicationRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, publication: EhrPublication) -> None:
        await self._session.merge(_publication_to_row(publication))

    async def get(self, publication_id: PublicationId) -> EhrPublication | None:
        return await self._fetch(EhrPublicationRow.id == publication_id.value)

    async def get_by_correction_id(self, correction_id: CorrectionId) -> EhrPublication | None:
        return await self._fetch(EhrPublicationRow.correction_id == correction_id.value)

    async def get_by_source_report_id(self, report_id: SoapReportId) -> EhrPublication | None:
        return await self._fetch(EhrPublicationRow.source_report_id == report_id.value)

    async def _fetch(self, condition: object) -> EhrPublication | None:
        statement = select(EhrPublicationRow).where(condition)  # type: ignore[arg-type]
        row = (await self._session.execute(statement)).scalar_one_or_none()
        if row is None:
            return None
        publication = _publication_to_domain(row)
        publication.verify_snapshot()
        return publication


class SqlAlchemyPublicationOutboxRepository(PublicationOutboxRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, event: PublicationOutbox) -> None:
        await self._session.merge(_outbox_to_row(event))

    async def get(self, event_id: PublicationOutboxId) -> PublicationOutbox | None:
        return await self._fetch(PublicationOutboxRow.id == event_id.value)

    async def get_by_publication_id(
        self, publication_id: PublicationId
    ) -> PublicationOutbox | None:
        return await self._fetch(PublicationOutboxRow.publication_id == publication_id.value)

    async def claim_due(self, *, now: datetime, limit: int) -> list[PublicationOutbox]:
        statement = (
            select(PublicationOutboxRow)
            .where(
                PublicationOutboxRow.delivered_at.is_(None),
                PublicationOutboxRow.next_attempt_at <= now,
            )
            .order_by(
                PublicationOutboxRow.next_attempt_at,
                PublicationOutboxRow.created_at,
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        rows = (await self._session.execute(statement)).scalars().all()
        events = [_outbox_to_domain(row) for row in rows]
        for event in events:
            event.verify_payload()
        return events

    async def _fetch(self, condition: object) -> PublicationOutbox | None:
        statement = select(PublicationOutboxRow).where(condition)  # type: ignore[arg-type]
        row = (await self._session.execute(statement)).scalar_one_or_none()
        if row is None:
            return None
        event = _outbox_to_domain(row)
        event.verify_payload()
        return event


def _publication_to_row(publication: EhrPublication) -> EhrPublicationRow:
    return EhrPublicationRow(
        id=publication.id.value,
        source_report_id=publication.source_report_id.value,
        correction_id=publication.correction_id.value,
        patient_ref=publication.patient_ref,
        encounter_ref=publication.encounter_ref,
        author_ref=publication.author_ref,
        snapshot=snapshot_to_dict(publication.snapshot),
        snapshot_hash=publication.snapshot_hash,
        snapshot_schema_version=publication.snapshot_schema_version,
        status=publication.status.value,
        remote_reference=publication.remote_reference,
        remote_version=publication.remote_version,
        created_at=publication.created_at,
        updated_at=publication.updated_at,
        delivered_at=publication.delivered_at,
    )


def _publication_to_domain(row: EhrPublicationRow) -> EhrPublication:
    return EhrPublication(
        id=Id(row.id),
        source_report_id=Id(row.source_report_id),
        correction_id=Id(row.correction_id),
        patient_ref=row.patient_ref,
        encounter_ref=row.encounter_ref,
        author_ref=row.author_ref,
        snapshot=snapshot_from_dict(row.snapshot),
        snapshot_hash=row.snapshot_hash,
        snapshot_schema_version=row.snapshot_schema_version,
        status=PublicationStatus(row.status),
        remote_reference=row.remote_reference,
        remote_version=row.remote_version,
        created_at=row.created_at,
        updated_at=row.updated_at,
        delivered_at=row.delivered_at,
    )


def _outbox_to_row(event: PublicationOutbox) -> PublicationOutboxRow:
    return PublicationOutboxRow(
        id=event.id.value,
        publication_id=event.publication_id.value,
        payload=event.payload,
        payload_hash=event.payload_hash,
        attempt_count=event.attempt_count,
        next_attempt_at=event.next_attempt_at,
        last_error=event.last_error,
        created_at=event.created_at,
        delivered_at=event.delivered_at,
    )


def _outbox_to_domain(row: PublicationOutboxRow) -> PublicationOutbox:
    return PublicationOutbox(
        id=Id(row.id),
        publication_id=Id(row.publication_id),
        payload=row.payload,
        payload_hash=row.payload_hash,
        attempt_count=row.attempt_count,
        next_attempt_at=row.next_attempt_at,
        last_error=row.last_error,
        created_at=row.created_at,
        delivered_at=row.delivered_at,
    )
