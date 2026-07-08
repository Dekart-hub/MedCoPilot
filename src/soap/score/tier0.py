"""Tier 0 — deterministic structural gate (proposal §5, Tier 0).

Checks that every cited evidence quote actually resolves to a span of the
dialogue turn it references (plain normalized substring match — this alone
catches fabricated citations, no model involved). Schema validity is
guaranteed by construction (typed domain model), so it is not re-checked
here. Per requirements clarification 3.3, a legitimately empty section is
recorded as a flag, not a failure.
"""

from __future__ import annotations

from dataclasses import dataclass

from dialogue import Dialogue, DialogueTurn, DialogueTurnId
from shared.entity import Entity
from shared.value_objects import Id

from ..soap import SoapClaimId, SoapNote, SoapNoteId, SoapReport, SoapReportId

type SoapTier0ReportId = Id[SoapTier0Report]


@dataclass(frozen=True, slots=True)
class Tier0NoteResult:
    soap_note_id: SoapNoteId
    passed: bool
    empty_sections: list[str]
    unresolved_claim_ids: list[SoapClaimId]
    citations_total: int
    citations_resolved: int


@dataclass(eq=False, slots=True)
class SoapTier0Report(Entity[SoapTier0ReportId]):
    """Side-car aggregate, mirrors SoapConfidenceReport / SoapCodingReport."""

    id: SoapTier0ReportId
    soap_report_id: SoapReportId
    results: list[Tier0NoteResult]


def _normalize(text: str) -> str:
    """Case-insensitive, whitespace-collapsed form for substring matching."""
    return " ".join(text.lower().split())


def _check_note(
    note: SoapNote, turns_by_id: dict[DialogueTurnId, DialogueTurn]
) -> Tier0NoteResult:
    empty_sections: list[str] = []
    unresolved: list[SoapClaimId] = []
    total = 0
    resolved = 0

    for section, claim in note.sections():
        if not claim.claim.strip():
            empty_sections.append(section)
            continue
        total += 1
        turn = turns_by_id.get(claim.evidence.turn_id)
        quote = _normalize(claim.evidence.text)
        if turn is not None and quote and quote in _normalize(turn.content):
            resolved += 1
        else:
            unresolved.append(claim.id)

    return Tier0NoteResult(
        soap_note_id=note.id,
        passed=not unresolved,
        empty_sections=empty_sections,
        unresolved_claim_ids=unresolved,
        citations_total=total,
        citations_resolved=resolved,
    )


def run_tier0(dialogue: Dialogue, report: SoapReport) -> SoapTier0Report:
    turns_by_id = {turn.id: turn for turn in dialogue.turns}
    return SoapTier0Report(
        id=Id.new(),
        soap_report_id=report.id,
        results=[_check_note(note, turns_by_id) for note in report.soap_notes],
    )
