from __future__ import annotations

import re
from datetime import datetime, timezone

from dialogue import Dialogue, DialogueRepository
from shared.value_objects import Id
from soap import ReportView

from .errors import (
    ApprovalConflictError,
    DialogueNotFoundError,
    EhrGatewayError,
    InvalidEhrReferenceError,
    ReportNotApprovedError,
    ReportNotFoundError,
    UnlinkedDialogueError,
)
from .gateway import EhrGateway
from .models import (
    ApprovalStatus,
    EhrSyncResult,
    PatientContext,
    ReportRecord,
    SyncStatus,
)
from .repository import ReportRepository

_CLINICIAN_REF = re.compile(r"^Practitioner/[A-Za-z0-9.-]{1,64}$")


class ReportWorkflow:
    """Coordinates patient context, approval, and idempotent EHR sync."""

    def __init__(
        self,
        dialogue_repository: DialogueRepository,
        report_repository: ReportRepository,
        ehr_gateway: EhrGateway,
    ) -> None:
        self._dialogues = dialogue_repository
        self._reports = report_repository
        self._ehr = ehr_gateway

    async def store_generated_report(
        self, report: ReportView, dialogue: Dialogue
    ) -> ReportRecord:
        record = ReportRecord(
            report_id=str(report.id),
            dialogue_id=str(dialogue.id),
            report=report,
            patient_ref=dialogue.patient_ref,
            encounter_ref=dialogue.encounter_ref,
        )
        await self._reports.save(record)
        return record

    async def get_record(self, report_id: str) -> ReportRecord:
        record = await self._reports.get(report_id)
        if record is None:
            raise ReportNotFoundError("Report not found")
        return record

    async def get_patient_context(self, dialogue_id: str) -> PatientContext:
        try:
            parsed_id = Id.from_str(dialogue_id)
        except ValueError as exc:
            raise DialogueNotFoundError("Dialogue not found") from exc
        dialogue = await self._dialogues.get(parsed_id)
        if dialogue is None:
            raise DialogueNotFoundError("Dialogue not found")
        if not dialogue.patient_ref or not dialogue.encounter_ref:
            raise UnlinkedDialogueError(
                "Dialogue has no patient_ref and encounter_ref linkage"
            )
        return await self._ehr.get_patient_context(
            patient_ref=dialogue.patient_ref,
            encounter_ref=dialogue.encounter_ref,
        )

    async def approve(self, report_id: str, clinician_ref: str) -> ReportRecord:
        if not _CLINICIAN_REF.fullmatch(clinician_ref):
            raise InvalidEhrReferenceError(
                "clinician_ref must have the form Practitioner/{id}"
            )
        record = await self.get_record(report_id)
        if record.approval_status is ApprovalStatus.APPROVED:
            if record.approved_by != clinician_ref:
                raise ApprovalConflictError(
                    "Report was already approved by another clinician"
                )
            return record

        record.approval_status = ApprovalStatus.APPROVED
        record.approved_by = clinician_ref
        record.approved_at = datetime.now(timezone.utc)
        record.last_error = None
        await self._reports.save(record)
        return record

    async def sync(self, report_id: str) -> ReportRecord:
        record = await self.get_record(report_id)
        if record.approval_status is not ApprovalStatus.APPROVED:
            raise ReportNotApprovedError(
                "Clinician approval is required before EHR sync"
            )
        if not record.patient_ref or not record.encounter_ref:
            raise UnlinkedDialogueError(
                "The report's dialogue has no patient/encounter linkage"
            )
        if record.sync_status is SyncStatus.SYNCED:
            return record

        record.sync_status = SyncStatus.SYNCING
        record.last_error = None
        await self._reports.save(record)
        try:
            result: EhrSyncResult = await self._ehr.sync_report(record)
        except (EhrGatewayError, InvalidEhrReferenceError) as exc:
            record.sync_status = SyncStatus.FAILED
            record.last_error = str(exc)
            await self._reports.save(record)
            raise

        record.sync_status = SyncStatus.SYNCED
        record.remote_reference = result.remote_reference
        record.remote_version_id = result.version_id
        await self._reports.save(record)
        return record
