from __future__ import annotations

from datetime import datetime, timezone

from dialogue import DialogueTurnId
from shared.value_objects import FloatRangedScore, Id
from soap.coding.coding import (
    DiagnosisCoding,
    SoapCodingReport,
    SoapNoteCoding,
)
from soap.score.score import SoapConfidenceReport, SoapNoteConfidenceScore
from soap.soap import SoapClaim, SoapEvidence, SoapNote, SoapReport
from soap.view import to_view


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

    view = to_view(report, confidence, coding)

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

    view = to_view(report, empty_conf, empty_coding)

    assert view.notes[0].confidence is None
    assert view.notes[0].assessment.codings == []
