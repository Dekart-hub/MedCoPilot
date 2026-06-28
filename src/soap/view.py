"""Линеаризованный read-model отчёта.

Доменные агрегаты (``SoapReport``, ``SoapConfidenceReport``, ``SoapCodingReport``)
остаются нормализованными write-model: их пишут три независимые подсистемы
(extract / score / normalize). Здесь — единственное место, где они джойнятся
по id в одно дерево «нота → всё про неё внутри»: ``confidence`` и
``assessment.codings`` уезжают внутрь ноты. Это потребляют и HTTP-API, и
будущая запись в ЭМК, поэтому view живёт в домене, а не в слое API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from dialogue import DialogueTurnId

from .coding.coding import DiagnosisCoding, SoapCodingReport
from .score.score import SoapConfidenceReport
from .soap import SoapClaim, SoapClaimId, SoapNoteId, SoapReport, SoapReportId


@dataclass(frozen=True, slots=True)
class ClaimView:
    id: SoapClaimId
    claim: str
    evidence_text: str
    turn_id: DialogueTurnId


@dataclass(frozen=True, slots=True)
class AssessmentView:
    """Как ``ClaimView``, но несёт коды классификатора инлайн."""

    id: SoapClaimId
    claim: str
    evidence_text: str
    turn_id: DialogueTurnId
    codings: list[DiagnosisCoding] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class NoteView:
    id: SoapNoteId
    subjective: ClaimView
    objective: ClaimView
    assessment: AssessmentView
    plan: ClaimView
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ReportView:
    id: SoapReportId
    notes: list[NoteView]
    created_at: datetime
    updated_at: datetime


def _claim(claim: SoapClaim) -> ClaimView:
    return ClaimView(
        id=claim.id,
        claim=claim.claim,
        evidence_text=claim.evidence.text,
        turn_id=claim.evidence.turn_id,
    )


def _assessment(claim: SoapClaim, codings: list[DiagnosisCoding]) -> AssessmentView:
    return AssessmentView(
        id=claim.id,
        claim=claim.claim,
        evidence_text=claim.evidence.text,
        turn_id=claim.evidence.turn_id,
        codings=codings,
    )


def to_view(
    report: SoapReport,
    confidence: SoapConfidenceReport,
    coding: SoapCodingReport,
) -> ReportView:
    """Джойнит три агрегата в линеаризованное дерево отчёта."""
    confidence_by_note = {
        score.soap_note_id: score.score.score
        for score in confidence.confidence_scores
    }
    codings_by_claim = {
        note_coding.soap_claim_id: note_coding.candidates
        for note_coding in coding.codings
    }
    return ReportView(
        id=report.id,
        notes=[
            NoteView(
                id=note.id,
                subjective=_claim(note.subjective),
                objective=_claim(note.objective),
                assessment=_assessment(
                    note.assessment, codings_by_claim.get(note.assessment.id, [])
                ),
                plan=_claim(note.plan),
                confidence=confidence_by_note.get(note.id),
            )
            for note in report.soap_notes
        ],
        created_at=report.created_at,
        updated_at=report.updated_at,
    )
