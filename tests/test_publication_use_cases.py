from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from dialogue.dialogue import Dialogue, DialogueId
from dialogue.repository import DialogueRepository
from ehr.publication import (
    EhrPublication,
    PublicationId,
    PublicationOutbox,
    PublicationOutboxId,
    PublicationRequiresVerifiedCorrection,
)
from ehr.publication_repository import (
    EhrPublicationRepository,
    PublicationOutboxRepository,
)
from ehr.publication_use_cases import (
    GetEhrPublication,
    PublicationCorrectionNotFoundError,
    PublicationNotFoundError,
    RequestEhrPublication,
    RequestEhrPublicationCommand,
)
from shared.value_objects import Id
from soap.correction import CorrectionStatus, SoapReportCorrection
from soap.correction_repository import SoapReportCorrectionRepository
from soap.repository import ReportSummary, SoapReportRepository
from soap.soap import SoapReport, SoapReportId

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


class FakeSession:
    def __init__(self) -> None:
        self.flushes = 0
        self.commits = 0
        self.rollbacks = 0

    async def flush(self) -> None:
        self.flushes += 1

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


class InMemoryDialogues(DialogueRepository):
    def __init__(self) -> None:
        self.items: dict[object, Dialogue] = {}

    async def save(self, dialogue: Dialogue) -> None:
        self.items[dialogue.id.value] = dialogue

    async def get(self, dialogue_id: DialogueId) -> Dialogue | None:
        return self.items.get(dialogue_id.value)


class InMemoryReports(SoapReportRepository):
    def __init__(self) -> None:
        self.items: dict[object, SoapReport] = {}
        self.dialogue_ids: dict[object, DialogueId] = {}

    async def save(
        self, report: SoapReport, *, dialogue_id: DialogueId, created_at: datetime
    ) -> None:
        self.items[report.id.value] = report
        self.dialogue_ids[report.id.value] = dialogue_id

    async def list_summaries(self) -> list[ReportSummary]:
        return []

    async def get(self, report_id: SoapReportId) -> SoapReport | None:
        return self.items.get(report_id.value)

    async def get_by_dialogue_id(self, dialogue_id: DialogueId) -> SoapReport | None:
        return next(
            (
                report
                for report_id, report in self.items.items()
                if self.dialogue_ids[report_id] == dialogue_id
            ),
            None,
        )

    async def get_dialogue_id(self, report_id: SoapReportId) -> DialogueId | None:
        return self.dialogue_ids.get(report_id.value)


class InMemoryCorrections(SoapReportCorrectionRepository):
    def __init__(self) -> None:
        self.items: dict[object, SoapReportCorrection] = {}
        self.saves = 0

    async def save(self, correction: SoapReportCorrection) -> None:
        self.saves += 1
        self.items[correction.source_report_id.value] = correction

    async def get(self, correction_id: object) -> SoapReportCorrection | None:
        return next(
            (item for item in self.items.values() if item.id == correction_id),
            None,
        )

    async def get_by_source_report_id(self, report_id: SoapReportId) -> SoapReportCorrection | None:
        return self.items.get(report_id.value)


class InMemoryPublications(EhrPublicationRepository):
    def __init__(self) -> None:
        self.items: dict[object, EhrPublication] = {}
        self.saves = 0

    async def save(self, publication: EhrPublication) -> None:
        self.saves += 1
        self.items[publication.id.value] = publication

    async def get(self, publication_id: PublicationId) -> EhrPublication | None:
        return self.items.get(publication_id.value)

    async def get_by_correction_id(self, correction_id: object) -> EhrPublication | None:
        return next(
            (item for item in self.items.values() if item.correction_id == correction_id),
            None,
        )

    async def get_by_source_report_id(self, report_id: SoapReportId) -> EhrPublication | None:
        return next(
            (item for item in self.items.values() if item.source_report_id == report_id),
            None,
        )


class InMemoryOutbox(PublicationOutboxRepository):
    def __init__(self) -> None:
        self.items: dict[object, PublicationOutbox] = {}
        self.saves = 0

    async def save(self, event: PublicationOutbox) -> None:
        self.saves += 1
        self.items[event.id.value] = event

    async def get(self, event_id: PublicationOutboxId) -> PublicationOutbox | None:
        return self.items.get(event_id.value)

    async def get_by_publication_id(
        self, publication_id: PublicationId
    ) -> PublicationOutbox | None:
        return next(
            (item for item in self.items.values() if item.publication_id == publication_id),
            None,
        )

    async def claim_due(self, *, now: datetime, limit: int) -> list[PublicationOutbox]:
        return [
            item
            for item in self.items.values()
            if item.delivered_at is None and item.next_attempt_at <= now
        ][:limit]


class Environment:
    def __init__(self, *, verified: bool = True) -> None:
        self.session = FakeSession()
        self.dialogues = InMemoryDialogues()
        self.reports = InMemoryReports()
        self.corrections = InMemoryCorrections()
        self.publications = InMemoryPublications()
        self.outbox = InMemoryOutbox()
        self.dialogue = Dialogue.start()
        self.dialogue.add_turn("patient", "Headache")
        self.report = SoapReport(id=Id.new())
        self.correction = SoapReportCorrection.start(self.report, created_at=_NOW)
        if verified:
            self.correction.verify("doctor-1", at=_NOW)
        self.dialogues.items[self.dialogue.id.value] = self.dialogue
        self.reports.items[self.report.id.value] = self.report
        self.reports.dialogue_ids[self.report.id.value] = self.dialogue.id
        self.corrections.items[self.report.id.value] = self.correction

    def request(self) -> RequestEhrPublication:
        return RequestEhrPublication(
            self.session,  # type: ignore[arg-type]
            self.corrections,
            self.reports,
            self.dialogues,
            self.publications,
            self.outbox,
        )

    def command(self, **overrides: str) -> RequestEhrPublicationCommand:
        values = {
            "patient_ref": "Patient/p1",
            "encounter_ref": "Encounter/e1",
            "author_ref": "Practitioner/d1",
            **overrides,
        }
        return RequestEhrPublicationCommand(report_id=self.report.id, **values)


def _run[T](awaitable: object) -> T:
    return asyncio.run(awaitable)  # type: ignore[arg-type,return-value]


def test_verified_correction_is_atomically_locked_with_snapshot_and_outbox() -> None:
    env = Environment()

    delivery = _run(env.request().execute(env.command()))

    assert delivery.publication.source_report_id == env.report.id
    assert delivery.publication.snapshot.dialogue_turns[0].text == "Headache"
    assert delivery.outbox.publication_id == delivery.publication.id
    assert env.correction.status is CorrectionStatus.PUBLICATION_PENDING
    assert env.publications.saves == 1
    assert env.outbox.saves == 1
    assert env.session.flushes == 1
    assert env.session.commits == 0


def test_repeat_request_returns_the_same_immutable_publication() -> None:
    env = Environment()
    first = _run(env.request().execute(env.command()))

    second = _run(
        env.request().execute(
            env.command(
                patient_ref="Patient/other",
                encounter_ref="Encounter/other",
                author_ref="Practitioner/other",
            )
        )
    )

    assert second.publication.id == first.publication.id
    assert second.publication.patient_ref == "Patient/p1"
    assert env.publications.saves == 1
    assert env.outbox.saves == 1


def test_draft_correction_is_rejected_without_partial_persistence() -> None:
    env = Environment(verified=False)

    with pytest.raises(PublicationRequiresVerifiedCorrection):
        _run(env.request().execute(env.command()))

    assert env.correction.status is CorrectionStatus.DRAFT
    assert not env.publications.items
    assert not env.outbox.items
    assert env.session.flushes == 0


def test_missing_correction_is_rejected() -> None:
    env = Environment()
    env.corrections.items.clear()

    with pytest.raises(PublicationCorrectionNotFoundError):
        _run(env.request().execute(env.command()))


def test_get_returns_delivery_state_and_missing_is_rejected() -> None:
    env = Environment()
    created = _run(env.request().execute(env.command()))
    get = GetEhrPublication(env.publications, env.outbox)

    loaded = _run(get.execute(env.report.id))
    assert loaded == created

    with pytest.raises(PublicationNotFoundError):
        _run(get.execute(Id.new()))
