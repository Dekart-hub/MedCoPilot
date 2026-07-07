from __future__ import annotations

from datetime import datetime, timezone

from dialogue import DialogueTurnId
from shared.value_objects import FloatRangedScore, Id
from soap.coding.coding import (
    DiagnosisCoding,
    SoapCodingReport,
    SoapNoteCoding,
)
from soap.score.score import (
    ClaimConfidenceScore,
    SoapConfidenceReport,
    SoapNoteConfidenceScore,
)
from soap.score.tier0 import SoapTier0Report, Tier0NoteResult
from soap.soap import SoapClaim, SoapEvidence, SoapNote, SoapReport
from soap.view import to_view


def _tier0_ok(report: SoapReport) -> SoapTier0Report:
    return SoapTier0Report(
        id=Id.new(),
        soap_report_id=report.id,
        results=[
            Tier0NoteResult(
                soap_note_id=note.id,
                passed=True,
                empty_sections=[],
                unresolved_claim_ids=[],
                citations_total=4,
                citations_resolved=4,
            )
            for note in report.soap_notes
        ],
    )


def _claim(text: str) -> SoapClaim:
    turn_id: DialogueTurnId = Id.new()
    return SoapClaim(
        id=Id.new(),
        claim=text,
        evidence=SoapEvidence(text=text, turn_id=turn_id),
    )


def _note() -> SoapNote:
    return SoapNote(
        id=Id.new(),
        subjective=_claim("болит шея"),
        objective=_claim("отек"),
        assessment=_claim("флегмона шеи"),
        plan=_claim("антибиотик"),
    )


def _report(note: SoapNote) -> SoapReport:
    now = datetime.now(timezone.utc)
    return SoapReport(id=Id.new(), soap_notes=[note], created_at=now, updated_at=now)


def _coding(claim_id) -> DiagnosisCoding:
    return DiagnosisCoding(
        code="L03.8",
        title="Флегмона других локализаций",
        matched_formulation="Флегмона шеи",
        score=FloatRangedScore(1.0),
    )


def test_confidence_and_codings_move_inline_per_note():
    note = _note()
    report = _report(note)

    confidence = SoapConfidenceReport(
        id=Id.new(),
        soap_report_id=report.id,
        confidence_scores=[
            SoapNoteConfidenceScore(
                id=Id.new(), score=FloatRangedScore(0.75), soap_note_id=note.id
            )
        ],
    )
    coding = SoapCodingReport(
        id=Id.new(),
        soap_report_id=report.id,
        codings=[
            SoapNoteCoding(
                id=Id.new(),
                soap_claim_id=note.assessment.id,
                candidates=[_coding(note.assessment.id)],
            )
        ],
    )

    view = to_view(report, confidence, coding, _tier0_ok(report))

    assert len(view.notes) == 1
    note_view = view.notes[0]
    # Score уехал внутрь ноты.
    assert note_view.confidence == 0.75
    # Коды уехали внутрь ассессмента.
    assert note_view.assessment.codings[0].code == "L03.8"
    # Остальные секции — обычные клеймы без кодов.
    assert note_view.subjective.claim == "болит шея"


def test_missing_enrichment_yields_none_and_empty():
    note = _note()
    report = _report(note)
    empty_conf = SoapConfidenceReport(
        id=Id.new(), soap_report_id=report.id, confidence_scores=[]
    )
    empty_coding = SoapCodingReport(
        id=Id.new(), soap_report_id=report.id, codings=[]
    )

    view = to_view(report, empty_conf, empty_coding, _tier0_ok(report))

    assert view.notes[0].confidence is None
    assert view.notes[0].assessment.codings == []


def _confidence_with_claims(
    report: SoapReport, note: SoapNote, flagged_claim_id, *, flagged: bool
) -> SoapConfidenceReport:
    claim_scores = [
        ClaimConfidenceScore(
            claim_id=claim.id,
            section=section,
            score=FloatRangedScore(0.2 if claim.id == flagged_claim_id else 0.9),
            is_flagged=flagged and claim.id == flagged_claim_id,
        )
        for section, claim in note.sections()
    ]
    return SoapConfidenceReport(
        id=Id.new(),
        soap_report_id=report.id,
        confidence_scores=[
            SoapNoteConfidenceScore(
                id=Id.new(),
                score=FloatRangedScore(0.725),
                soap_note_id=note.id,
                claim_scores=claim_scores,
            )
        ],
    )


def _empty_coding(report: SoapReport) -> SoapCodingReport:
    return SoapCodingReport(id=Id.new(), soap_report_id=report.id, codings=[])


def test_claim_grounding_scores_and_flags_reach_the_view():
    note = _note()
    report = _report(note)
    confidence = _confidence_with_claims(
        report, note, note.objective.id, flagged=True
    )

    view = to_view(report, confidence, _empty_coding(report), _tier0_ok(report))
    note_view = view.notes[0]

    assert note_view.subjective.grounding_score == 0.9
    assert note_view.subjective.is_flagged is False
    assert note_view.objective.grounding_score == 0.2
    assert note_view.objective.is_flagged is True
    assert note_view.needs_review is True


def test_unresolved_citation_flags_claim_and_note():
    note = _note()
    report = _report(note)
    confidence = _confidence_with_claims(
        report, note, note.plan.id, flagged=False
    )
    tier0 = SoapTier0Report(
        id=Id.new(),
        soap_report_id=report.id,
        results=[
            Tier0NoteResult(
                soap_note_id=note.id,
                passed=False,
                empty_sections=[],
                unresolved_claim_ids=[note.plan.id],
                citations_total=4,
                citations_resolved=3,
            )
        ],
    )

    view = to_view(report, confidence, _empty_coding(report), tier0)
    note_view = view.notes[0]

    assert note_view.tier0.passed is False
    assert note_view.tier0.citations_resolved == 3
    assert note_view.plan.is_flagged is True
    assert note_view.needs_review is True


def test_clean_note_does_not_need_review():
    note = _note()
    report = _report(note)
    confidence = _confidence_with_claims(
        report, note, note.plan.id, flagged=False
    )

    view = to_view(report, confidence, _empty_coding(report), _tier0_ok(report))
    note_view = view.notes[0]

    assert note_view.tier0.passed is True
    assert note_view.needs_review is False
