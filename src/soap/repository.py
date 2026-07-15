"""Persistence port for the SoapReport aggregate.

The domain depends on this abstraction; concrete adapters (SQLAlchemy, fakes)
implement it. A report is keyed to the dialogue it was extracted from — at most
one report per dialogue — so lookups by ``dialogue_id`` back the idempotent
extract use case.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dialogue.dialogue import DialogueId

from .soap import SoapReport, SoapReportId


class SoapReportRepository(ABC):
    """Stores and retrieves :class:`~soap.soap.SoapReport` aggregates."""

    @abstractmethod
    async def save(self, report: SoapReport, *, dialogue_id: DialogueId) -> None:
        """Persist the report, linked to the dialogue it was extracted from."""

    @abstractmethod
    async def get(self, report_id: SoapReportId) -> SoapReport | None:
        """Return the report by id, or ``None`` if it does not exist."""

    @abstractmethod
    async def get_by_dialogue_id(self, dialogue_id: DialogueId) -> SoapReport | None:
        """Return the report extracted from ``dialogue_id``, or ``None``."""
