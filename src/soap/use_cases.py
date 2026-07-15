"""Application use cases for SOAP reports.

Extraction is idempotent: a dialogue has at most one report. The report's
``dialogue_id`` carries a unique constraint, so a repeat request short-circuits
to the stored report (no second LLM call), and a concurrent request that races
past the pre-check is caught on flush, rolled back, and resolved to the report
the winner persisted.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from dialogue.dialogue import DialogueId
from dialogue.repository import DialogueRepository
from ehr.client import EhrClient

from .extractor import SoapExtractor
from .repository import SoapReportRepository
from .soap import SoapReport


class DialogueNotFoundError(Exception):
    """Raised when the dialogue to extract a report from does not exist."""

    def __init__(self, dialogue_id: DialogueId) -> None:
        super().__init__(f"dialogue {dialogue_id} not found")
        self.dialogue_id = dialogue_id


@dataclass(frozen=True, slots=True)
class ExtractSoapReportCommand:
    """Request to extract, or return the existing, report for a dialogue."""

    dialogue_id: DialogueId
    patient_id: str | None = None


class ExtractSoapReport:
    """Extracts a dialogue's SOAP report, persisting exactly one per dialogue."""

    def __init__(
        self,
        session: AsyncSession,
        dialogues: DialogueRepository,
        reports: SoapReportRepository,
        extractor: SoapExtractor,
        ehr: EhrClient,
    ) -> None:
        self._session = session
        self._dialogues = dialogues
        self._reports = reports
        self._extractor = extractor
        self._ehr = ehr

    async def execute(self, command: ExtractSoapReportCommand) -> SoapReport:
        existing = await self._reports.get_by_dialogue_id(command.dialogue_id)
        if existing is not None:
            return existing
        dialogue = await self._dialogues.get(command.dialogue_id)
        if dialogue is None:
            raise DialogueNotFoundError(command.dialogue_id)
        context = await self._patient_context(command.patient_id)
        report = await self._extractor.extract(dialogue, context)
        return await self._persist(report, command.dialogue_id)

    async def _patient_context(self, patient_id: str | None) -> str:
        if not patient_id:
            return ""
        return await self._ehr.get_patient_context(patient_id)

    async def _persist(self, report: SoapReport, dialogue_id: DialogueId) -> SoapReport:
        # Flush now so a lost race trips the unique constraint here, inside the
        # guard, rather than later in the caller's commit; the caller still owns
        # the commit.
        try:
            await self._reports.save(report, dialogue_id=dialogue_id)
            await self._session.flush()
        except IntegrityError:
            await self._session.rollback()
            winner = await self._reports.get_by_dialogue_id(dialogue_id)
            if winner is None:
                raise
            return winner
        return report
