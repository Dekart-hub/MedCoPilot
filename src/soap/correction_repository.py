"""Persistence port for the SoapReportCorrection aggregate.

The domain depends on this abstraction; concrete adapters (SQLAlchemy, fakes)
implement it. A correction is keyed to the source report it edits — at most one
per report — so lookups by ``source_report_id`` back the "resume the doctor's
draft" use case.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .correction import CorrectionId, SoapReportCorrection
from .soap import SoapReportId


class SoapReportCorrectionRepository(ABC):
    """Stores and retrieves :class:`~soap.correction.SoapReportCorrection` aggregates."""

    @abstractmethod
    async def save(self, correction: SoapReportCorrection) -> None:
        """Persist the correction and its notes, source report link included."""

    @abstractmethod
    async def get(self, correction_id: CorrectionId) -> SoapReportCorrection | None:
        """Return the correction by id, or ``None`` if it does not exist."""

    @abstractmethod
    async def get_by_source_report_id(self, report_id: SoapReportId) -> SoapReportCorrection | None:
        """Return the correction editing ``report_id``, or ``None``."""

    async def get_by_source_report_id_for_update(
        self, report_id: SoapReportId
    ) -> SoapReportCorrection | None:
        """Return and lock the correction when the adapter supports row locks."""
        return await self.get_by_source_report_id(report_id)
