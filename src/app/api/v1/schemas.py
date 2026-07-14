from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from dialogue import Dialogue
from ehr import EhrResourceSummary, PatientContext, ReportRecord
from soap import ReportView
from soap.view import (
    AssessmentView,
    ClaimView,
    ContextReferenceView,
    NoteView,
    Tier0View,
)

# --------------------------------------------------------------------------- #
# Запросы.
# --------------------------------------------------------------------------- #


class TurnRequest(BaseModel):
    role: str
    content: str


class CreateDialogueRequest(BaseModel):
    turns: list[TurnRequest]
    patient_ref: str | None = None
    encounter_ref: str | None = None


class CreateDialogueFromTextRequest(BaseModel):
    text: str
    patient_ref: str | None = None
    encounter_ref: str | None = None


class CreateReportRequest(BaseModel):
    dialogue_id: str


class ApproveReportRequest(BaseModel):
    clinician_ref: str


# --------------------------------------------------------------------------- #
# Ответы (DTO; маппинг из доменных сущностей).
# --------------------------------------------------------------------------- #


class TurnResponse(BaseModel):
    id: str
    role: str
    content: str
    timestamp: datetime


class DialogueResponse(BaseModel):
    id: str
    turns: list[TurnResponse]
    created_at: datetime
    patient_ref: str | None
    encounter_ref: str | None

    @classmethod
    def from_domain(cls, dialogue: Dialogue) -> DialogueResponse:
        return cls(
            id=str(dialogue.id),
            turns=[
                TurnResponse(
                    id=str(turn.id),
                    role=turn.role,
                    content=turn.content,
                    timestamp=turn.timestamp,
                )
                for turn in dialogue.turns
            ],
            created_at=dialogue.created_at,
            patient_ref=dialogue.patient_ref,
            encounter_ref=dialogue.encounter_ref,
        )


class EhrCodingResponse(BaseModel):
    system: str | None
    code: str | None
    display: str | None


class EhrResourceResponse(BaseModel):
    reference: str
    resource_type: str
    code: EhrCodingResponse | None
    status: str | None
    effective_at: str | None
    value: str | None

    @classmethod
    def from_domain(cls, resource: EhrResourceSummary) -> EhrResourceResponse:
        return cls(
            reference=resource.reference,
            resource_type=resource.resource_type,
            code=(
                EhrCodingResponse(
                    system=resource.code.system,
                    code=resource.code.code,
                    display=resource.code.display,
                )
                if resource.code
                else None
            ),
            status=resource.status,
            effective_at=resource.effective_at,
            value=resource.value,
        )


class PatientContextResponse(BaseModel):
    patient_ref: str
    encounter_ref: str
    encounter_start: str | None
    patient_name: str | None
    birth_date: str | None
    gender: str | None
    conditions: list[EhrResourceResponse]
    allergies: list[EhrResourceResponse]
    medications: list[EhrResourceResponse]
    observations: list[EhrResourceResponse]

    @classmethod
    def from_domain(cls, context: PatientContext) -> PatientContextResponse:
        return cls(
            patient_ref=context.patient_ref,
            encounter_ref=context.encounter_ref,
            encounter_start=context.encounter_start,
            patient_name=context.patient_name,
            birth_date=context.birth_date,
            gender=context.gender,
            conditions=[EhrResourceResponse.from_domain(r) for r in context.conditions],
            allergies=[EhrResourceResponse.from_domain(r) for r in context.allergies],
            medications=[EhrResourceResponse.from_domain(r) for r in context.medications],
            observations=[EhrResourceResponse.from_domain(r) for r in context.observations],
        )


class ReportWorkflowResponse(BaseModel):
    report_id: str
    dialogue_id: str
    patient_ref: str | None
    encounter_ref: str | None
    approval_status: str
    approved_by: str | None
    approved_at: datetime | None
    sync_status: str
    remote_reference: str | None
    remote_version_id: str | None
    last_error: str | None
    context_status: str = "not-linked"
    context_error: str | None = None

    @classmethod
    def from_domain(cls, record: ReportRecord) -> ReportWorkflowResponse:
        return cls(
            report_id=record.report_id,
            dialogue_id=record.dialogue_id,
            patient_ref=record.patient_ref,
            encounter_ref=record.encounter_ref,
            approval_status=record.approval_status,
            approved_by=record.approved_by,
            approved_at=record.approved_at,
            sync_status=record.sync_status,
            remote_reference=record.remote_reference,
            remote_version_id=record.remote_version_id,
            last_error=record.last_error,
            context_status=record.report.context_status,
            context_error=record.report.context_error,
        )


class ContextReferenceResponse(BaseModel):
    reference: str
    resource_type: str
    category: str
    display: str | None = None
    code: str | None = None
    status: str | None = None
    effective_at: str | None = None
    value: str | None = None

    @classmethod
    def from_view(
        cls, reference: ContextReferenceView
    ) -> ContextReferenceResponse:
        return cls(
            reference=reference.reference,
            resource_type=reference.resource_type,
            category=reference.category,
            display=reference.display,
            code=reference.code,
            status=reference.status,
            effective_at=reference.effective_at,
            value=reference.value,
        )


class ClaimResponse(BaseModel):
    id: str
    claim: str
    evidence_text: str
    turn_id: str
    grounding_score: float | None
    is_flagged: bool
    context_references: list[ContextReferenceResponse] = Field(
        default_factory=list
    )
    invalid_context_references: list[str] = Field(default_factory=list)

    @classmethod
    def from_view(cls, claim: ClaimView) -> ClaimResponse:
        return cls(
            id=str(claim.id),
            claim=claim.claim,
            evidence_text=claim.evidence_text,
            turn_id=str(claim.turn_id),
            grounding_score=claim.grounding_score,
            is_flagged=claim.is_flagged,
            context_references=[
                ContextReferenceResponse.from_view(reference)
                for reference in claim.context_references
            ],
            invalid_context_references=claim.invalid_context_references,
        )


class ClassifierResponse(BaseModel):
    system: str
    name: str
    version: str | None
    index_oid: str | None
    index_version: str | None


class CodingResponse(BaseModel):
    code: str
    title: str
    matched_formulation: str
    score: float
    classifier: ClassifierResponse


class AssessmentResponse(BaseModel):
    id: str
    claim: str
    evidence_text: str
    turn_id: str
    codings: list[CodingResponse]
    grounding_score: float | None
    is_flagged: bool
    context_references: list[ContextReferenceResponse] = Field(
        default_factory=list
    )
    invalid_context_references: list[str] = Field(default_factory=list)

    @classmethod
    def from_view(cls, assessment: AssessmentView) -> AssessmentResponse:
        return cls(
            id=str(assessment.id),
            claim=assessment.claim,
            evidence_text=assessment.evidence_text,
            turn_id=str(assessment.turn_id),
            grounding_score=assessment.grounding_score,
            is_flagged=assessment.is_flagged,
            context_references=[
                ContextReferenceResponse.from_view(reference)
                for reference in assessment.context_references
            ],
            invalid_context_references=assessment.invalid_context_references,
            codings=[
                CodingResponse(
                    code=c.code,
                    title=c.title,
                    matched_formulation=c.matched_formulation,
                    score=c.score.score,
                    classifier=ClassifierResponse(
                        system=c.classifier.system,
                        name=c.classifier.name,
                        version=c.classifier.version,
                        index_oid=c.classifier.index_oid,
                        index_version=c.classifier.index_version,
                    ),
                )
                for c in assessment.codings
            ],
        )


class Tier0Response(BaseModel):
    passed: bool
    empty_sections: list[str]
    citations_total: int
    citations_resolved: int

    @classmethod
    def from_view(cls, tier0: Tier0View) -> Tier0Response:
        return cls(
            passed=tier0.passed,
            empty_sections=tier0.empty_sections,
            citations_total=tier0.citations_total,
            citations_resolved=tier0.citations_resolved,
        )


class NoteResponse(BaseModel):
    id: str
    subjective: ClaimResponse
    objective: ClaimResponse
    assessment: AssessmentResponse
    plan: ClaimResponse
    tier0: Tier0Response
    needs_review: bool
    confidence: float | None

    @classmethod
    def from_view(cls, note: NoteView) -> NoteResponse:
        return cls(
            id=str(note.id),
            subjective=ClaimResponse.from_view(note.subjective),
            objective=ClaimResponse.from_view(note.objective),
            assessment=AssessmentResponse.from_view(note.assessment),
            plan=ClaimResponse.from_view(note.plan),
            tier0=Tier0Response.from_view(note.tier0),
            needs_review=note.needs_review,
            confidence=note.confidence,
        )


class ReportResponse(BaseModel):
    id: str
    soap_notes: list[NoteResponse]
    created_at: datetime
    updated_at: datetime
    context_status: str = "not-linked"
    context_error: str | None = None

    @classmethod
    def from_domain(cls, view: ReportView) -> ReportResponse:
        return cls(
            id=str(view.id),
            soap_notes=[NoteResponse.from_view(note) for note in view.notes],
            created_at=view.created_at,
            updated_at=view.updated_at,
            context_status=view.context_status,
            context_error=view.context_error,
        )
