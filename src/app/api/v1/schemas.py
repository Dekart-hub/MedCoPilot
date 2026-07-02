from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from dialogue import Dialogue
from soap import ReportView
from soap.view import AssessmentView, ClaimView, NoteView

# --------------------------------------------------------------------------- #
# Запросы.
# --------------------------------------------------------------------------- #


class TurnRequest(BaseModel):
    role: str
    content: str


class CreateDialogueRequest(BaseModel):
    turns: list[TurnRequest]


class CreateDialogueFromTextRequest(BaseModel):
    text: str


class CreateReportRequest(BaseModel):
    dialogue_id: str


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
        )


class ClaimResponse(BaseModel):
    id: str
    claim: str
    evidence_text: str
    turn_id: str

    @classmethod
    def from_view(cls, claim: ClaimView) -> ClaimResponse:
        return cls(
            id=str(claim.id),
            claim=claim.claim,
            evidence_text=claim.evidence_text,
            turn_id=str(claim.turn_id),
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


def _coding_response(c) -> CodingResponse:
    return CodingResponse(
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


class AssessmentResponse(BaseModel):
    id: str
    claim: str
    evidence_text: str
    turn_id: str
    codings: list[CodingResponse]
    selected: CodingResponse | None = None
    rationale: str | None = None

    @classmethod
    def from_view(cls, assessment: AssessmentView) -> AssessmentResponse:
        return cls(
            id=str(assessment.id),
            claim=assessment.claim,
            evidence_text=assessment.evidence_text,
            turn_id=str(assessment.turn_id),
            codings=[_coding_response(c) for c in assessment.codings],
            selected=(
                _coding_response(assessment.selected)
                if assessment.selected
                else None
            ),
            rationale=assessment.rationale,
        )


class NoteResponse(BaseModel):
    id: str
    subjective: ClaimResponse
    objective: ClaimResponse
    assessment: AssessmentResponse
    plan: ClaimResponse
    confidence: float | None

    @classmethod
    def from_view(cls, note: NoteView) -> NoteResponse:
        return cls(
            id=str(note.id),
            subjective=ClaimResponse.from_view(note.subjective),
            objective=ClaimResponse.from_view(note.objective),
            assessment=AssessmentResponse.from_view(note.assessment),
            plan=ClaimResponse.from_view(note.plan),
            confidence=note.confidence,
        )


class ReportResponse(BaseModel):
    id: str
    soap_notes: list[NoteResponse]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, view: ReportView) -> ReportResponse:
        return cls(
            id=str(view.id),
            soap_notes=[NoteResponse.from_view(note) for note in view.notes],
            created_at=view.created_at,
            updated_at=view.updated_at,
        )
