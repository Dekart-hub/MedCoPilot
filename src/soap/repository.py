"""Persistence port for the SoapReport aggregate.

The domain depends on this abstraction; concrete adapters (SQLAlchemy, fakes)
implement it. A report is keyed to the dialogue it was extracted from — at most
one report per dialogue — so lookups by ``dialogue_id`` back the idempotent
extract use case.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from dialogue.dialogue import DialogueId

from .soap import SoapReport, SoapReportId


@dataclass(frozen=True, slots=True)
class ReportSummary:
    """A report reduced to what a list view needs: no notes or claims loaded.

    Backs ``GET /reports`` so the picker can page over encounters cheaply
    without materializing every aggregate.
    """

    report_id: SoapReportId
    dialogue_id: DialogueId
    created_at: datetime


class SoapReportRepository(ABC):
    """Stores and retrieves :class:`~soap.soap.SoapReport` aggregates."""

    @abstractmethod
    async def save(
        self, report: SoapReport, *, dialogue_id: DialogueId, created_at: datetime
    ) -> None:
        """Persist the report, linked to its dialogue and stamped with ``created_at``."""

    @abstractmethod
    async def list_summaries(self) -> list[ReportSummary]:
        """Return every report as a summary, newest first (by ``created_at``)."""

    @abstractmethod
    async def get(self, report_id: SoapReportId) -> SoapReport | None:
        """Return the report by id, or ``None`` if it does not exist."""

    @abstractmethod
    async def get_by_dialogue_id(self, dialogue_id: DialogueId) -> SoapReport | None:
        """Return the report extracted from ``dialogue_id``, or ``None``."""

    @abstractmethod
    async def get_dialogue_id(self, report_id: SoapReportId) -> DialogueId | None:
        """Return the id of the dialogue ``report_id`` was extracted from, or ``None``."""
