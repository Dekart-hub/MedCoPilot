"""Application use case for dialogue-level online SOAP quality (story #10).

Quality is a read model calculated on demand.  The original report and the
doctor's correction remain the sources of truth; no metric row is persisted.
The deterministic engine in :mod:`soap.quality` owns comparison semantics,
while this use case only resolves the aggregates by ``dialogue_id``.
"""

from __future__ import annotations

from dataclasses import dataclass

from dialogue.dialogue import DialogueId

from .correction import CorrectionId
from .correction_repository import SoapReportCorrectionRepository
from .quality import SoapNoteQualityDiff, calculate_soap_report_diff
from .repository import SoapReportRepository
from .soap import SoapReportId


class DialogueReportNotFoundError(Exception):
    """Raised when no source SOAP report exists for the requested dialogue."""

    def __init__(self, dialogue_id: DialogueId) -> None:
        super().__init__(f"soap report for dialogue {dialogue_id} not found")
        self.dialogue_id = dialogue_id


class QualityCorrectionNotFoundError(Exception):
    """Raised when the dialogue's source report has no correction."""

    def __init__(self, report_id: SoapReportId) -> None:
        super().__init__(f"soap correction for report {report_id} not found")
        self.report_id = report_id


@dataclass(frozen=True, slots=True)
class DialogueSoapQuality:
    """Dialogue identity plus report-level and matched-note quality metrics."""

    dialogue_id: DialogueId
    report_id: SoapReportId
    correction_id: CorrectionId
    notes_added: int
    notes_removed: int
    changed_characters: int
    diagnosis_changes: int
    note_diffs: list[SoapNoteQualityDiff]


class GetDialogueSoapQuality:
    """Calculate online quality from a report and its current verified correction."""

    def __init__(
        self,
        reports: SoapReportRepository,
        corrections: SoapReportCorrectionRepository,
    ) -> None:
        self._reports = reports
        self._corrections = corrections

    async def execute(self, dialogue_id: DialogueId) -> DialogueSoapQuality:
        report = await self._reports.get_by_dialogue_id(dialogue_id)
        if report is None:
            raise DialogueReportNotFoundError(dialogue_id)

        correction = await self._corrections.get_by_source_report_id(report.id)
        if correction is None:
            raise QualityCorrectionNotFoundError(report.id)

        diff = calculate_soap_report_diff(report, correction)
        return DialogueSoapQuality(
            dialogue_id=dialogue_id,
            report_id=report.id,
            correction_id=correction.id,
            notes_added=diff.notes_added,
            notes_removed=diff.notes_removed,
            changed_characters=diff.changed_characters,
            diagnosis_changes=diff.diagnosis_changes,
            note_diffs=diff.note_diffs,
        )
