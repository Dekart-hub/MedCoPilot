from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from soap import ReportView

WHO_ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10"
FIXTURE_PHASE_SYSTEM = "urn:medcopilot:fixture-phase"
PRE_VISIT_PHASE = "pre-visit"
POST_VISIT_PHASE = "post-visit"


class ApprovalStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"


class SyncStatus(StrEnum):
    NOT_SYNCED = "not-synced"
    SYNCING = "syncing"
    SYNCED = "synced"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EhrCoding:
    system: str | None
    code: str | None
    display: str | None


@dataclass(frozen=True, slots=True)
class EhrResourceSummary:
    reference: str
    resource_type: str
    code: EhrCoding | None = None
    status: str | None = None
    effective_at: str | None = None
    value: str | None = None


@dataclass(frozen=True, slots=True)
class PatientContext:
    patient_ref: str
    encounter_ref: str
    encounter_start: str | None
    patient_name: str | None
    birth_date: str | None
    gender: str | None
    conditions: list[EhrResourceSummary] = field(default_factory=list)
    allergies: list[EhrResourceSummary] = field(default_factory=list)
    medications: list[EhrResourceSummary] = field(default_factory=list)
    observations: list[EhrResourceSummary] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EhrSyncResult:
    remote_reference: str
    version_id: str | None = None


@dataclass(slots=True)
class ReportRecord:
    report_id: str
    dialogue_id: str
    report: ReportView
    patient_ref: str | None
    encounter_ref: str | None
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    approved_by: str | None = None
    approved_at: datetime | None = None
    sync_status: SyncStatus = SyncStatus.NOT_SYNCED
    remote_reference: str | None = None
    remote_version_id: str | None = None
    last_error: str | None = None
