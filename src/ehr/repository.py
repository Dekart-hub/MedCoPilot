from __future__ import annotations

from abc import ABC, abstractmethod

from .models import ReportRecord


class ReportRepository(ABC):
    @abstractmethod
    async def save(self, record: ReportRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get(self, report_id: str) -> ReportRecord | None:
        raise NotImplementedError


class InMemoryReportRepository(ReportRepository):
    def __init__(self) -> None:
        self._records: dict[str, ReportRecord] = {}

    async def save(self, record: ReportRecord) -> None:
        self._records[record.report_id] = record

    async def get(self, report_id: str) -> ReportRecord | None:
        return self._records.get(report_id)
