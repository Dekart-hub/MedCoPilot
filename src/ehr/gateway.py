from __future__ import annotations

from abc import ABC, abstractmethod

from .errors import EhrIntegrationDisabledError
from .models import EhrSyncResult, PatientContext, ReportRecord


class EhrGateway(ABC):
    @abstractmethod
    async def get_patient_context(
        self, *, patient_ref: str, encounter_ref: str
    ) -> PatientContext:
        raise NotImplementedError

    @abstractmethod
    async def sync_report(self, record: ReportRecord) -> EhrSyncResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release transport resources, if any."""


class DisabledEhrGateway(EhrGateway):
    _MESSAGE = (
        "Mock EHR integration is disabled; set EHR__ENABLED=true to enable it"
    )

    async def get_patient_context(
        self, *, patient_ref: str, encounter_ref: str
    ) -> PatientContext:
        raise EhrIntegrationDisabledError(self._MESSAGE)

    async def sync_report(self, record: ReportRecord) -> EhrSyncResult:
        raise EhrIntegrationDisabledError(self._MESSAGE)
