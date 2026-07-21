"""Persistence ports for EHR publications and durable outbox events."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from soap.correction import CorrectionId
from soap.soap import SoapReportId

from .publication import (
    EhrPublication,
    PublicationId,
    PublicationOutbox,
    PublicationOutboxId,
)


class EhrPublicationRepository(ABC):
    @abstractmethod
    async def save(self, publication: EhrPublication) -> None:
        """Persist a publication without committing the caller's transaction."""

    @abstractmethod
    async def get(self, publication_id: PublicationId) -> EhrPublication | None:
        """Return a publication by id."""

    @abstractmethod
    async def get_by_correction_id(self, correction_id: CorrectionId) -> EhrPublication | None:
        """Return the single publication for a correction."""

    @abstractmethod
    async def get_by_source_report_id(self, report_id: SoapReportId) -> EhrPublication | None:
        """Return the single publication for a source report."""


class PublicationOutboxRepository(ABC):
    @abstractmethod
    async def save(self, event: PublicationOutbox) -> None:
        """Persist an outbox event without committing the caller's transaction."""

    @abstractmethod
    async def get(self, event_id: PublicationOutboxId) -> PublicationOutbox | None:
        """Return an outbox event by id."""

    @abstractmethod
    async def get_by_publication_id(
        self, publication_id: PublicationId
    ) -> PublicationOutbox | None:
        """Return the durable event for a publication."""

    @abstractmethod
    async def claim_due(self, *, now: datetime, limit: int) -> list[PublicationOutbox]:
        """Lock and return due undelivered events for this transaction."""
