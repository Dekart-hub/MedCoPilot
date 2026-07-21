"""Durable outbox dispatcher for reliable FHIR publication delivery."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from soap.correction import CorrectionStatus
from soap.correction_sqlalchemy_repository import (
    SqlAlchemySoapReportCorrectionRepository,
)

from .fhir import FhirPublicationGateway
from .publication import PublicationStatus
from .publication_sqlalchemy_repository import (
    SqlAlchemyEhrPublicationRepository,
    SqlAlchemyPublicationOutboxRepository,
)


class PublicationDispatcher:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        gateway: FhirPublicationGateway,
        *,
        batch_size: int,
        poll_seconds: float,
        retry_initial_seconds: float,
        retry_max_seconds: float,
    ) -> None:
        self._session_factory = session_factory
        self._gateway = gateway
        self._batch_size = batch_size
        self._poll_seconds = poll_seconds
        self._retry_initial_seconds = retry_initial_seconds
        self._retry_max_seconds = retry_max_seconds
        self._log = structlog.get_logger(__name__)

    async def run_once(self) -> int:
        now = datetime.now(UTC)
        async with self._session_factory() as session:
            outbox = SqlAlchemyPublicationOutboxRepository(session)
            events = await outbox.claim_due(now=now, limit=self._batch_size)
            if not events:
                return 0
            publications = SqlAlchemyEhrPublicationRepository(session)
            corrections = SqlAlchemySoapReportCorrectionRepository(session)
            for event in events:
                publication = await publications.get(event.publication_id)
                if publication is None:
                    event.record_failure(
                        "publication row is missing",
                        next_attempt_at=self._next_attempt(event.attempt_count, now),
                    )
                    await outbox.save(event)
                    continue
                correction = await corrections.get(publication.correction_id)
                if correction is None:
                    event.record_failure(
                        "publication correction is missing",
                        next_attempt_at=self._next_attempt(event.attempt_count, now),
                    )
                    await outbox.save(event)
                    continue
                if publication.status is PublicationStatus.DELIVERED:
                    if correction.status is CorrectionStatus.PUBLICATION_PENDING:
                        correction.mark_published(at=now)
                        await corrections.save(correction)
                    event.mark_delivered(at=publication.delivered_at or now)
                    await outbox.save(event)
                    continue
                try:
                    result = await self._gateway.deliver(publication)
                    completed_at = datetime.now(UTC)
                    publication.mark_delivered(
                        remote_reference=result.remote_reference,
                        remote_version=result.remote_version,
                        at=completed_at,
                    )
                    correction.mark_published(at=completed_at)
                    event.mark_delivered(at=completed_at)
                except Exception as exc:
                    failed_at = datetime.now(UTC)
                    event.record_failure(
                        str(exc),
                        next_attempt_at=self._next_attempt(
                            event.attempt_count,
                            failed_at,
                        ),
                    )
                    await outbox.save(event)
                    self._log.warning(
                        "fhir_publication_delivery_failed",
                        publication_id=str(publication.id),
                        attempt=event.attempt_count,
                        error=str(exc),
                    )
                    continue
                await publications.save(publication)
                await corrections.save(correction)
                await outbox.save(event)
            await session.commit()
            return len(events)

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                processed = await self.run_once()
            except Exception as exc:
                processed = 0
                self._log.exception("fhir_publication_dispatcher_failed", error=str(exc))
            if processed:
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=self._poll_seconds)

    def _next_attempt(self, attempt_count: int, now: datetime) -> datetime:
        delay = min(
            self._retry_max_seconds,
            self._retry_initial_seconds * (2**attempt_count),
        )
        return now + timedelta(seconds=delay)
