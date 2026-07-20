"""Unit tests for the ExtractSoapReport use case, with in-memory fakes.

No database and no LLM: the dialogue and report repositories are dictionaries,
the session is a stub and the extractor is canned. The tests pin the behaviours
that matter — the report carries ICD + confidence, extraction is idempotent
(one report, one LLM call per dialogue), a lost race resolves to the winner, and
a missing dialogue is reported rather than extracted.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import datetime

import pytest
from sqlalchemy.exc import IntegrityError

from dialogue.dialogue import Dialogue, DialogueId
from dialogue.repository import DialogueRepository
from infra.ehr import MockEhrClient
from shared.value_objects import Id
from soap.extractor import SoapExtractor
from soap.repository import ReportSummary, SoapReportRepository
from soap.soap import (
    AssessmentClaim,
    IcdCoding,
    SoapClaim,
    SoapNote,
    SoapReport,
    SoapReportId,
    TurnCitation,
)
from soap.use_cases import (
    DialogueNotFoundError,
    ExtractSoapReport,
    ExtractSoapReportCommand,
)

_ICD = IcdCoding(code="G44.2", name="Tension-type headache", classifier_url="https://icd/G44.2")


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _dialogue() -> Dialogue:
    dialogue = Dialogue.start()
    dialogue.add_turn("patient", "I've had a headache for three days.")
    dialogue.add_turn("doctor", "Blood pressure is 140 over 90.")
    return dialogue


def _report_for(dialogue: Dialogue) -> SoapReport:
    citation = TurnCitation(turn_id=dialogue.turns[0].id, quote="headache")
    note = SoapNote(
        id=Id.new(),
        subjective=[SoapClaim(id=Id.new(), text="Headache for three days.", citations=[citation])],
        assessment=[
            AssessmentClaim(id=Id.new(), text="Tension headache.", citations=[citation], icd=_ICD)
        ],
        confidence=0.87,
    )
    return SoapReport(id=Id.new(), notes=[note])


class FakeSession:
    """Stands in for an AsyncSession: records commits, no-ops the rest."""

    def __init__(self) -> None:
        self.commits = 0

    async def flush(self) -> None: ...

    async def rollback(self) -> None: ...

    async def commit(self) -> None:
        self.commits += 1


class InMemoryDialogueRepository(DialogueRepository):
    def __init__(self) -> None:
        self._store: dict[object, Dialogue] = {}

    async def save(self, dialogue: Dialogue) -> None:
        self._store[dialogue.id.value] = dialogue

    async def get(self, dialogue_id: DialogueId) -> Dialogue | None:
        return self._store.get(dialogue_id.value)


class InMemorySoapReportRepository(SoapReportRepository):
    def __init__(self) -> None:
        self._by_id: dict[object, SoapReport] = {}
        self._by_dialogue: dict[object, SoapReport] = {}
        self._dialogue_of: dict[object, DialogueId] = {}
        self.created_at: dict[object, datetime] = {}

    async def save(
        self, report: SoapReport, *, dialogue_id: DialogueId, created_at: datetime
    ) -> None:
        self._by_id[report.id.value] = report
        self._by_dialogue[dialogue_id.value] = report
        self._dialogue_of[report.id.value] = dialogue_id
        self.created_at[report.id.value] = created_at

    async def get(self, report_id: SoapReportId) -> SoapReport | None:
        return self._by_id.get(report_id.value)

    async def get_by_dialogue_id(self, dialogue_id: DialogueId) -> SoapReport | None:
        return self._by_dialogue.get(dialogue_id.value)

    async def get_dialogue_id(self, report_id: SoapReportId) -> DialogueId | None:
        return self._dialogue_of.get(report_id.value)

    async def list_summaries(self) -> list[ReportSummary]:
        summaries = [
            ReportSummary(
                report_id=report.id,
                dialogue_id=self._dialogue_of[report.id.value],
                created_at=self.created_at[report.id.value],
            )
            for report in self._by_id.values()
        ]
        return sorted(summaries, key=lambda summary: summary.created_at, reverse=True)


class StubExtractor(SoapExtractor):
    """Canned extractor: counts calls and records the context it was handed."""

    def __init__(self) -> None:
        self.calls = 0
        self.contexts: list[str] = []

    async def extract(self, dialogue: Dialogue, patient_context: str) -> SoapReport:
        self.calls += 1
        self.contexts.append(patient_context)
        return _report_for(dialogue)


def _use_case(
    dialogue: Dialogue | None,
    *,
    reports: SoapReportRepository | None = None,
    extractor: StubExtractor | None = None,
    ehr: MockEhrClient | None = None,
    session: FakeSession | None = None,
) -> tuple[ExtractSoapReport, StubExtractor, SoapReportRepository]:
    dialogues = InMemoryDialogueRepository()
    if dialogue is not None:
        _run(dialogues.save(dialogue))
    reports = reports or InMemorySoapReportRepository()
    extractor = extractor or StubExtractor()
    use_case = ExtractSoapReport(
        session or FakeSession(),  # type: ignore[arg-type]
        dialogues,
        reports,
        extractor,
        ehr or MockEhrClient({}),
    )
    return use_case, extractor, reports


def test_report_carries_icd_and_confidence() -> None:
    dialogue = _dialogue()
    use_case, extractor, reports = _use_case(dialogue)

    report = _run(use_case.execute(ExtractSoapReportCommand(dialogue_id=dialogue.id)))

    assert extractor.calls == 1
    note = report.notes[0]
    assert note.confidence == 0.87
    assert note.assessment[0].icd == _ICD
    assert _run(reports.get(report.id)) is report  # it was persisted


def test_second_extraction_reuses_the_stored_report() -> None:
    dialogue = _dialogue()
    use_case, extractor, _ = _use_case(dialogue)
    command = ExtractSoapReportCommand(dialogue_id=dialogue.id)

    first = _run(use_case.execute(command))
    second = _run(use_case.execute(command))

    assert second.id == first.id
    assert extractor.calls == 1  # the extractor is not called a second time


def test_second_extraction_keeps_the_original_created_at() -> None:
    dialogue = _dialogue()
    reports = InMemorySoapReportRepository()
    use_case, _, _ = _use_case(dialogue, reports=reports)
    command = ExtractSoapReportCommand(dialogue_id=dialogue.id)

    first = _run(use_case.execute(command))
    stamped = reports.created_at[first.id.value]
    _run(use_case.execute(command))

    assert reports.created_at[first.id.value] == stamped  # [#88/FR-1] never re-stamped


def test_patient_id_resolves_ehr_context_into_the_extractor() -> None:
    dialogue = _dialogue()
    ehr = MockEhrClient({"p1": "45yo, hypertension on lisinopril"})
    use_case, extractor, _ = _use_case(dialogue, ehr=ehr)

    _run(use_case.execute(ExtractSoapReportCommand(dialogue_id=dialogue.id, patient_id="p1")))

    assert extractor.contexts == ["45yo, hypertension on lisinopril"]


def test_missing_dialogue_is_reported_not_extracted() -> None:
    use_case, extractor, _ = _use_case(None)

    with pytest.raises(DialogueNotFoundError):
        _run(use_case.execute(ExtractSoapReportCommand(dialogue_id=Id.new())))

    assert extractor.calls == 0


class RacingReports(SoapReportRepository):
    """A concurrent writer won: the report is invisible on the pre-check, then visible."""

    def __init__(self, winner: SoapReport) -> None:
        self._winner = winner
        self._probes = 0

    async def save(
        self, report: SoapReport, *, dialogue_id: DialogueId, created_at: datetime
    ) -> None: ...

    async def get(self, report_id: SoapReportId) -> SoapReport | None:
        return self._winner

    async def get_by_dialogue_id(self, dialogue_id: DialogueId) -> SoapReport | None:
        self._probes += 1
        return None if self._probes == 1 else self._winner

    async def get_dialogue_id(self, report_id: SoapReportId) -> DialogueId | None:
        return None

    async def list_summaries(self) -> list[ReportSummary]:
        return []


class ConflictingSession(FakeSession):
    async def flush(self) -> None:
        raise IntegrityError("insert soap_report", None, Exception("duplicate dialogue_id"))


def test_lost_race_resolves_to_the_winning_report() -> None:
    dialogue = _dialogue()
    winner = _report_for(dialogue)
    use_case, extractor, _ = _use_case(
        dialogue,
        reports=RacingReports(winner),
        session=ConflictingSession(),
    )

    result = _run(use_case.execute(ExtractSoapReportCommand(dialogue_id=dialogue.id)))

    assert result.id == winner.id  # the unique-constraint loser returns the winner
    assert extractor.calls == 1  # extraction ran, then the write lost the race
