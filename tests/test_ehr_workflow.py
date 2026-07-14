from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from dialogue import Dialogue, InMemoryDialogueRepository
from ehr import (
    ApprovalStatus,
    EhrCoding,
    EhrGateway,
    EhrGatewayError,
    EhrResourceSummary,
    EhrSyncResult,
    GenerateReport,
    InMemoryReportRepository,
    PatientContext,
    ReportNotApprovedError,
    ReportWorkflow,
    SyncStatus,
    UnlinkedDialogueError,
)
from shared.value_objects import Id
from soap import ContextStatus, ReportView


class FakeEhrGateway(EhrGateway):
    def __init__(self) -> None:
        self.sync_calls = 0
        self.context_calls = 0

    async def get_patient_context(
        self, *, patient_ref: str, encounter_ref: str
    ) -> PatientContext:
        self.context_calls += 1
        return PatientContext(
            patient_ref=patient_ref,
            encounter_ref=encounter_ref,
            encounter_start="2026-01-01T10:00:00Z",
            patient_name="Alexey Petrov",
            birth_date="1990-04-12",
            gender="male",
        )

    async def sync_report(self, record) -> EhrSyncResult:
        self.sync_calls += 1
        return EhrSyncResult(
            remote_reference="DocumentReference/mock-document-001",
            version_id="1",
        )


def _dialogue(*, linked: bool = True) -> Dialogue:
    return Dialogue(
        id=Id.new(),
        turns=[],
        created_at=datetime.now(timezone.utc),
        patient_ref="Patient/p1" if linked else None,
        encounter_ref="Encounter/e1" if linked else None,
    )


def _report() -> ReportView:
    now = datetime.now(timezone.utc)
    return ReportView(id=Id.new(), notes=[], created_at=now, updated_at=now)


def test_context_uses_explicit_dialogue_linkage():
    dialogue = _dialogue()
    gateway = FakeEhrGateway()
    workflow = ReportWorkflow(
        InMemoryDialogueRepository(initial=[dialogue]),
        InMemoryReportRepository(),
        gateway,
    )

    context = asyncio.run(workflow.get_patient_context(str(dialogue.id)))

    assert context.patient_ref == "Patient/p1"
    assert context.encounter_ref == "Encounter/e1"


def test_unlinked_dialogue_cannot_read_patient_context():
    dialogue = _dialogue(linked=False)
    workflow = ReportWorkflow(
        InMemoryDialogueRepository(initial=[dialogue]),
        InMemoryReportRepository(),
        FakeEhrGateway(),
    )

    with pytest.raises(UnlinkedDialogueError):
        asyncio.run(workflow.get_patient_context(str(dialogue.id)))


def test_generation_context_is_bounded_and_preserves_fhir_references():
    dialogue = _dialogue()
    gateway = FakeEhrGateway()

    async def get_context(*, patient_ref: str, encounter_ref: str):
        gateway.context_calls += 1
        return PatientContext(
            patient_ref=patient_ref,
            encounter_ref=encounter_ref,
            encounter_start="2026-01-01T10:00:00Z",
            patient_name="Synthetic Patient",
            birth_date="1990-04-12",
            gender="male",
            conditions=[
                EhrResourceSummary(
                    reference=f"Condition/c{i}",
                    resource_type="Condition",
                    code=EhrCoding(
                        system="http://hl7.org/fhir/sid/icd-10",
                        code=f"C{i}",
                        display=("X" * 300 if i == 0 else f"Condition {i}"),
                    ),
                )
                for i in range(12)
            ],
        )

    gateway.get_patient_context = get_context
    workflow = ReportWorkflow(
        InMemoryDialogueRepository(initial=[dialogue]),
        InMemoryReportRepository(),
        gateway,
    )

    prepared = asyncio.run(workflow.prepare_clinical_context(dialogue))

    assert prepared.status is ContextStatus.AVAILABLE
    assert prepared.context is not None
    assert len(prepared.context.resources) == 10
    assert prepared.context.resources[0].reference == "Condition/c0"
    assert prepared.context.resources[0].display == "X" * 256


def test_generation_falls_back_when_linked_context_is_unavailable():
    dialogue = _dialogue()
    gateway = FakeEhrGateway()

    async def fail(*, patient_ref: str, encounter_ref: str):
        raise EhrGatewayError("Mock EHR timed out")

    gateway.get_patient_context = fail
    workflow = ReportWorkflow(
        InMemoryDialogueRepository(initial=[dialogue]),
        InMemoryReportRepository(),
        gateway,
    )

    prepared = asyncio.run(workflow.prepare_clinical_context(dialogue))

    assert prepared.status is ContextStatus.UNAVAILABLE
    assert prepared.context is None
    assert prepared.error == "Mock EHR timed out"


def test_generation_skips_context_lookup_for_unlinked_dialogue():
    dialogue = _dialogue(linked=False)
    gateway = FakeEhrGateway()
    workflow = ReportWorkflow(
        InMemoryDialogueRepository(initial=[dialogue]),
        InMemoryReportRepository(),
        gateway,
    )

    prepared = asyncio.run(workflow.prepare_clinical_context(dialogue))

    assert prepared.status is ContextStatus.NOT_LINKED
    assert gateway.context_calls == 0


def test_generate_report_prepares_context_and_stores_the_view():
    dialogue = _dialogue()
    gateway = FakeEhrGateway()
    reports = InMemoryReportRepository()
    workflow = ReportWorkflow(
        InMemoryDialogueRepository(initial=[dialogue]), reports, gateway
    )

    class CapturingExtractor:
        def __init__(self) -> None:
            self.prepared = None

        async def execute(self, received_dialogue, prepared):
            self.prepared = prepared
            now = datetime.now(timezone.utc)
            return ReportView(
                id=Id.new(),
                notes=[],
                created_at=now,
                updated_at=now,
                context_status=prepared.status,
            )

    extractor = CapturingExtractor()
    use_case = GenerateReport(extractor, workflow)

    view = asyncio.run(use_case.execute(dialogue))
    stored = asyncio.run(reports.get(str(view.id)))

    assert extractor.prepared.status is ContextStatus.AVAILABLE
    assert stored is not None
    assert stored.report is view


def test_approval_is_required_and_sync_is_locally_idempotent():
    dialogue = _dialogue()
    gateway = FakeEhrGateway()
    workflow = ReportWorkflow(
        InMemoryDialogueRepository(initial=[dialogue]),
        InMemoryReportRepository(),
        gateway,
    )
    report = _report()
    record = asyncio.run(workflow.store_generated_report(report, dialogue))

    assert record.approval_status is ApprovalStatus.DRAFT
    with pytest.raises(ReportNotApprovedError):
        asyncio.run(workflow.sync(record.report_id))

    approved = asyncio.run(
        workflow.approve(record.report_id, "Practitioner/mock-gp-001")
    )
    first = asyncio.run(workflow.sync(record.report_id))
    second = asyncio.run(workflow.sync(record.report_id))

    assert approved.approval_status is ApprovalStatus.APPROVED
    assert first.sync_status is SyncStatus.SYNCED
    assert second.remote_reference == "DocumentReference/mock-document-001"
    assert gateway.sync_calls == 1
