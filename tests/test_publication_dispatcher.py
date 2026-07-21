from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

import ehr.dispatcher as dispatcher_module
from dialogue.dialogue import Dialogue
from ehr.dispatcher import PublicationDispatcher
from ehr.fhir import FhirDeliveryResult, FhirGatewayError, FhirPublicationGateway
from ehr.publication import EhrPublication, PublicationOutbox, PublicationStatus
from shared.value_objects import Id
from soap.correction import CorrectionStatus, SoapReportCorrection
from soap.soap import SoapReport

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _state() -> tuple[EhrPublication, PublicationOutbox, SoapReportCorrection]:
    dialogue = Dialogue.start()
    dialogue.add_turn("patient", "Headache")
    correction = SoapReportCorrection.start(SoapReport(id=Id.new()), created_at=_NOW)
    correction.verify("doctor-1", at=_NOW)
    publication = EhrPublication.request(
        correction,
        dialogue,
        patient_ref="Patient/p1",
        encounter_ref="Encounter/e1",
        author_ref="Practitioner/d1",
        at=_NOW,
    )
    event = PublicationOutbox.for_publication(publication, at=_NOW)
    correction.begin_publication(at=_NOW)
    return publication, event, correction


class FakePublications:
    def __init__(self, publication: EhrPublication) -> None:
        self.publication = publication
        self.saves = 0

    async def get(self, publication_id: object) -> EhrPublication | None:
        return self.publication if self.publication.id == publication_id else None

    async def save(self, publication: EhrPublication) -> None:
        self.saves += 1
        self.publication = publication


class FakeCorrections:
    def __init__(self, correction: SoapReportCorrection) -> None:
        self.correction = correction
        self.saves = 0

    async def get(self, correction_id: object) -> SoapReportCorrection | None:
        return self.correction if self.correction.id == correction_id else None

    async def save(self, correction: SoapReportCorrection) -> None:
        self.saves += 1
        self.correction = correction


class FakeOutbox:
    def __init__(self, event: PublicationOutbox) -> None:
        self.event = event
        self.in_flight = False
        self.saves = 0
        self._lock = asyncio.Lock()

    async def claim_due(self, *, now: datetime, limit: int) -> list[PublicationOutbox]:
        async with self._lock:
            if (
                self.in_flight
                or self.event.delivered_at is not None
                or self.event.next_attempt_at > now
            ):
                return []
            self.in_flight = True
            return [self.event][:limit]

    async def save(self, event: PublicationOutbox) -> None:
        self.saves += 1
        self.event = event

    def release(self) -> None:
        self.in_flight = False


class FakeSession:
    def __init__(self, outbox: FakeOutbox) -> None:
        self.outbox = outbox
        self.commits = 0

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.outbox.release()

    async def commit(self) -> None:
        self.commits += 1
        self.outbox.release()


class FakeSessionFactory:
    def __init__(self, outbox: FakeOutbox) -> None:
        self.outbox = outbox
        self.sessions: list[FakeSession] = []

    def __call__(self) -> FakeSession:
        session = FakeSession(self.outbox)
        self.sessions.append(session)
        return session


class SequenceGateway(FhirPublicationGateway):
    def __init__(self, outcomes: list[Exception | FhirDeliveryResult]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def deliver(self, publication: EhrPublication) -> FhirDeliveryResult:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class SlowGateway(FhirPublicationGateway):
    def __init__(self) -> None:
        self.calls = 0

    async def deliver(self, publication: EhrPublication) -> FhirDeliveryResult:
        self.calls += 1
        await asyncio.sleep(0.01)
        return FhirDeliveryResult("Bundle/b1", "1")


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    publication: EhrPublication,
    event: PublicationOutbox,
    correction: SoapReportCorrection,
) -> tuple[FakePublications, FakeOutbox, FakeCorrections, FakeSessionFactory]:
    publications = FakePublications(publication)
    outbox = FakeOutbox(event)
    corrections = FakeCorrections(correction)
    factory = FakeSessionFactory(outbox)
    monkeypatch.setattr(
        dispatcher_module,
        "SqlAlchemyEhrPublicationRepository",
        lambda session: publications,
    )
    monkeypatch.setattr(
        dispatcher_module,
        "SqlAlchemyPublicationOutboxRepository",
        lambda session: outbox,
    )
    monkeypatch.setattr(
        dispatcher_module,
        "SqlAlchemySoapReportCorrectionRepository",
        lambda session: corrections,
    )
    return publications, outbox, corrections, factory


def _dispatcher(
    factory: FakeSessionFactory, gateway: FhirPublicationGateway
) -> PublicationDispatcher:
    return PublicationDispatcher(
        factory,  # type: ignore[arg-type]
        gateway,
        batch_size=10,
        poll_seconds=0.01,
        retry_initial_seconds=0,
        retry_max_seconds=0,
    )


def test_failure_is_persisted_and_a_new_dispatcher_resumes_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, event, correction = _state()
    publications, outbox, corrections, factory = _wire(monkeypatch, publication, event, correction)
    gateway = SequenceGateway([FhirGatewayError("offline"), FhirDeliveryResult("Bundle/b1", "2")])

    first = asyncio.run(_dispatcher(factory, gateway).run_once())

    assert first == 1
    assert outbox.event.attempt_count == 1
    assert outbox.event.last_error == "offline"
    assert outbox.event.delivered_at is None
    assert publications.publication.status is PublicationStatus.PENDING
    assert corrections.correction.status is CorrectionStatus.PUBLICATION_PENDING

    second = asyncio.run(_dispatcher(factory, gateway).run_once())

    assert second == 1
    assert gateway.calls == 2
    assert outbox.event.attempt_count == 2
    assert outbox.event.delivered_at is not None
    assert publications.publication.remote_reference == "Bundle/b1"
    assert publications.publication.status is PublicationStatus.DELIVERED
    assert corrections.correction.status is CorrectionStatus.PUBLISHED


def test_two_dispatchers_claim_one_event_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, event, correction = _state()
    _, outbox, _, factory = _wire(monkeypatch, publication, event, correction)
    gateway = SlowGateway()

    async def exercise() -> tuple[int, int]:
        first = _dispatcher(factory, gateway)
        second = _dispatcher(factory, gateway)
        results = await asyncio.gather(first.run_once(), second.run_once())
        return results[0], results[1]

    processed = asyncio.run(exercise())

    assert sorted(processed) == [0, 1]
    assert gateway.calls == 1
    assert outbox.event.delivered_at is not None
