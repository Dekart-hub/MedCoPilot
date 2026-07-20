"""Unit tests for dialogue-level online SOAP quality orchestration (T23)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime

import pytest

from dialogue.dialogue import DialogueId
from shared.value_objects import Id
from soap.correction import CorrectionId, SoapReportCorrection
from soap.correction_repository import SoapReportCorrectionRepository
from soap.quality import CorrectionNotVerifiedError
from soap.quality_use_cases import (
    DialogueReportNotFoundError,
    GetDialogueSoapQuality,
    QualityCorrectionNotFoundError,
)
from soap.repository import SoapReportRepository
from soap.soap import SoapClaim, SoapNote, SoapReport, SoapReportId, TurnCitation

_NOW = datetime(2026, 7, 20, tzinfo=UTC)


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


class InMemoryReports(SoapReportRepository):
    def __init__(self) -> None:
        self.by_id: dict[object, SoapReport] = {}
        self.by_dialogue: dict[object, SoapReport] = {}
        self.dialogue_of: dict[object, DialogueId] = {}

    async def save(
        self, report: SoapReport, *, dialogue_id: DialogueId, created_at: datetime
    ) -> None:
        self.by_id[report.id.value] = report
        self.by_dialogue[dialogue_id.value] = report
        self.dialogue_of[report.id.value] = dialogue_id

    async def get(self, report_id: SoapReportId) -> SoapReport | None:
        return self.by_id.get(report_id.value)

    async def get_by_dialogue_id(self, dialogue_id: DialogueId) -> SoapReport | None:
        return self.by_dialogue.get(dialogue_id.value)

    async def get_dialogue_id(self, report_id: SoapReportId) -> DialogueId | None:
        return self.dialogue_of.get(report_id.value)


class InMemoryCorrections(SoapReportCorrectionRepository):
    def __init__(self) -> None:
        self.by_id: dict[object, SoapReportCorrection] = {}
        self.by_source: dict[object, SoapReportCorrection] = {}

    async def save(self, correction: SoapReportCorrection) -> None:
        self.by_id[correction.id.value] = correction
        self.by_source[correction.source_report_id.value] = correction

    async def get(self, correction_id: CorrectionId) -> SoapReportCorrection | None:
        return self.by_id.get(correction_id.value)

    async def get_by_source_report_id(self, report_id: SoapReportId) -> SoapReportCorrection | None:
        return self.by_source.get(report_id.value)


def _source() -> tuple[DialogueId, SoapReport]:
    citation = TurnCitation(turn_id=Id.new())
    return Id.new(), SoapReport(
        id=Id.new(),
        notes=[
            SoapNote(
                id=Id.new(),
                subjective=[SoapClaim(id=Id.new(), text="pain", citations=[citation])],
            )
        ],
    )


def _use_case(
    dialogue_id: DialogueId,
    report: SoapReport,
    *,
    with_correction: bool = True,
    verified: bool = True,
) -> GetDialogueSoapQuality:
    reports = InMemoryReports()
    corrections = InMemoryCorrections()
    _run(reports.save(report, dialogue_id=dialogue_id, created_at=datetime.now(UTC)))
    if with_correction:
        correction = SoapReportCorrection.start(report, created_at=_NOW)
        correction.notes[0].subjective[0].text = "gain"
        if verified:
            correction.verify("doctor-1", at=_NOW)
        _run(corrections.save(correction))
    return GetDialogueSoapQuality(reports, corrections)


def test_quality_resolves_report_and_verified_correction_by_dialogue() -> None:
    dialogue_id, report = _source()
    result = _run(_use_case(dialogue_id, report).execute(dialogue_id))

    assert result.dialogue_id == dialogue_id
    assert result.report_id == report.id
    assert result.changed_characters == 1
    assert result.notes_added == 0
    assert result.notes_removed == 0
    assert len(result.note_diffs) == 1


def test_quality_rejects_dialogue_without_a_report() -> None:
    dialogue_id, report = _source()
    use_case = _use_case(dialogue_id, report)

    with pytest.raises(DialogueReportNotFoundError):
        _run(use_case.execute(Id.new()))


def test_quality_rejects_report_without_a_correction() -> None:
    dialogue_id, report = _source()
    use_case = _use_case(dialogue_id, report, with_correction=False)

    with pytest.raises(QualityCorrectionNotFoundError):
        _run(use_case.execute(dialogue_id))


def test_quality_rejects_a_draft_correction() -> None:
    dialogue_id, report = _source()
    use_case = _use_case(dialogue_id, report, verified=False)

    with pytest.raises(CorrectionNotVerifiedError):
        _run(use_case.execute(dialogue_id))
