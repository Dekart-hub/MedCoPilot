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
from .score.score import ClaimConfidenceScore, SoapConfidenceReport
from .score.tier0 import SoapTier0Report, Tier0NoteResult
from .soap import SoapClaim, SoapClaimId, SoapNoteId, SoapReport, SoapReportId


@dataclass(frozen=True, slots=True)
class Tier0View:
    """Structural gate verdict for one note (deterministic, model-free)."""

    passed: bool
    empty_sections: list[str]
    citations_total: int
    citations_resolved: int


@dataclass(frozen=True, slots=True)
class ClaimView:
    id: SoapClaimId
    claim: str
    evidence_text: str
    turn_id: DialogueTurnId
    grounding_score: float | None = None
    is_flagged: bool = False


@dataclass(frozen=True, slots=True)
class AssessmentView:
    """Like ``ClaimView`` but carries classifier codes inline."""

    id: SoapClaimId
    claim: str
    evidence_text: str
    turn_id: DialogueTurnId
    codings: list[DiagnosisCoding] = field(default_factory=list)
    grounding_score: float | None = None
    is_flagged: bool = False


@dataclass(frozen=True, slots=True)
class NoteView:
    id: SoapNoteId
    subjective: ClaimView
    objective: ClaimView
    assessment: AssessmentView
    plan: ClaimView
    tier0: Tier0View
    needs_review: bool
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ReportView:
    id: SoapReportId
    notes: list[NoteView]
    created_at: datetime
    updated_at: datetime


def _grounding(
    claim: SoapClaim,
    claim_scores: dict[SoapClaimId, ClaimConfidenceScore],
    unresolved: frozenset[SoapClaimId],
) -> tuple[float | None, bool]:
    cs = claim_scores.get(claim.id)
    score = cs.score.score if cs is not None else None
    flagged = (cs.is_flagged if cs is not None else False) or claim.id in unresolved
    return score, flagged


def _claim(
    claim: SoapClaim,
    claim_scores: dict[SoapClaimId, ClaimConfidenceScore],
    unresolved: frozenset[SoapClaimId],
) -> ClaimView:
    score, flagged = _grounding(claim, claim_scores, unresolved)
    return ClaimView(
        id=claim.id,
        claim=claim.claim,
        evidence_text=claim.evidence.text,
        turn_id=claim.evidence.turn_id,
        grounding_score=score,
        is_flagged=flagged,
    )


def _assessment(
    claim: SoapClaim,
    codings: list[DiagnosisCoding],
    claim_scores: dict[SoapClaimId, ClaimConfidenceScore],
    unresolved: frozenset[SoapClaimId],
) -> AssessmentView:
    score, flagged = _grounding(claim, claim_scores, unresolved)
    return AssessmentView(
        id=claim.id,
        claim=claim.claim,
        evidence_text=claim.evidence.text,
        turn_id=claim.evidence.turn_id,
        codings=codings,
        grounding_score=score,
        is_flagged=flagged,
    )


_TIER0_FALLBACK = Tier0View(
    passed=True, empty_sections=[], citations_total=0, citations_resolved=0
)


def _tier0_view(result: Tier0NoteResult | None) -> Tier0View:
    if result is None:
        return _TIER0_FALLBACK
    return Tier0View(
        passed=result.passed,
        empty_sections=list(result.empty_sections),
        citations_total=result.citations_total,
        citations_resolved=result.citations_resolved,
    )


def to_view(
    report: SoapReport,
    confidence: SoapConfidenceReport,
    coding: SoapCodingReport,
    tier0: SoapTier0Report,
) -> ReportView:
    """Joins the four side-car aggregates into one linearized report tree."""
    confidence_by_note = {
        score.soap_note_id: score for score in confidence.confidence_scores
    }
    codings_by_claim = {
        note_coding.soap_claim_id: note_coding.candidates
        for note_coding in coding.codings
    }
    tier0_by_note = {result.soap_note_id: result for result in tier0.results}

    notes: list[NoteView] = []
    for note in report.soap_notes:
        note_score = confidence_by_note.get(note.id)
        claim_scores = {
            cs.claim_id: cs for cs in (note_score.claim_scores if note_score else [])
        }
        t0 = tier0_by_note.get(note.id)
        unresolved = frozenset(t0.unresolved_claim_ids) if t0 else frozenset()

        subjective = _claim(note.subjective, claim_scores, unresolved)
        objective = _claim(note.objective, claim_scores, unresolved)
        assessment = _assessment(
            note.assessment,
            codings_by_claim.get(note.assessment.id, []),
            claim_scores,
            unresolved,
        )
        plan = _claim(note.plan, claim_scores, unresolved)

        tier0_view = _tier0_view(t0)
        any_flagged = any(
            c.is_flagged for c in (subjective, objective, assessment, plan)
        )
        notes.append(
            NoteView(
                id=note.id,
                subjective=subjective,
                objective=objective,
                assessment=assessment,
                plan=plan,
                tier0=tier0_view,
                needs_review=(
                    not tier0_view.passed
                    or bool(tier0_view.empty_sections)
                    or any_flagged
                ),
                confidence=note_score.score.score if note_score else None,
            )
        )

    return ReportView(
        id=report.id,
        notes=notes,
        created_at=report.created_at,
        updated_at=report.updated_at,
    )
