from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from shared.entity import Entity
from shared.value_objects import Id

from .soap import SoapNoteId, SoapReport, SoapReportId

type ContextSection = Literal["assessment", "plan"]
type EhrContextSupportReportId = Id[EhrContextSupportReport]


class ContextStatus(StrEnum):
    NOT_LINKED = "not-linked"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ClinicalContextResource:
    reference: str
    resource_type: str
    category: str
    display: str | None = None
    code: str | None = None
    status: str | None = None
    effective_at: str | None = None
    value: str | None = None


@dataclass(frozen=True, slots=True)
class ClinicalContextInput:
    patient_ref: str
    encounter_ref: str
    resources: tuple[ClinicalContextResource, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedClinicalContext:
    status: ContextStatus
    context: ClinicalContextInput | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RequestedContextSupport:
    soap_note_id: SoapNoteId
    section: ContextSection
    references: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ContextSupportResult:
    soap_note_id: SoapNoteId
    section: ContextSection
    references: list[ClinicalContextResource] = field(default_factory=list)
    invalid_references: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.invalid_references


@dataclass(eq=False, slots=True)
class EhrContextSupportReport(Entity[EhrContextSupportReportId]):
    id: EhrContextSupportReportId
    soap_report_id: SoapReportId
    status: ContextStatus
    results: list[ContextSupportResult] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SoapExtraction:
    report: SoapReport
    requested_context: list[RequestedContextSupport] = field(default_factory=list)


def validate_context_support(
    extraction: SoapExtraction,
    prepared: PreparedClinicalContext,
) -> EhrContextSupportReport:
    available = {
        resource.reference: resource
        for resource in (prepared.context.resources if prepared.context else ())
    }
    requested = {
        (item.soap_note_id, item.section): _unique_references(item.references)
        for item in extraction.requested_context
    }
    results: list[ContextSupportResult] = []
    for note in extraction.report.soap_notes:
        for section in ("assessment", "plan"):
            references = requested.get((note.id, section), [])
            results.append(
                ContextSupportResult(
                    soap_note_id=note.id,
                    section=section,
                    references=[available[ref] for ref in references if ref in available],
                    invalid_references=[ref for ref in references if ref not in available],
                )
            )
    return EhrContextSupportReport(
        id=Id.new(),
        soap_report_id=extraction.report.id,
        status=prepared.status,
        results=results,
        error=prepared.error,
    )


def _unique_references(references: list[str]) -> list[str]:
    return list(dict.fromkeys(ref.strip() for ref in references if ref.strip()))
