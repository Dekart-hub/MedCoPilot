from __future__ import annotations

from datetime import datetime, timezone

from shared.value_objects import Id
from soap.coding.coding import SoapCodingReport
from soap.context import (
    ClinicalContextInput,
    ClinicalContextResource,
    ContextStatus,
    PreparedClinicalContext,
    RequestedContextSupport,
    SoapExtraction,
    validate_context_support,
)
from soap.score.score import SoapConfidenceReport
from soap.score.tier0 import SoapTier0Report, Tier0NoteResult
from soap.soap import SoapClaim, SoapEvidence, SoapNote, SoapReport
from soap.view import to_view


def _claim(text: str) -> SoapClaim:
    return SoapClaim(
        id=Id.new(),
        claim=text,
        evidence=SoapEvidence(text=text, turn_id=Id.new()),
    )


def _report() -> tuple[SoapReport, SoapNote]:
    note = SoapNote(
        id=Id.new(),
        subjective=_claim("headache"),
        objective=_claim("normal temperature"),
        assessment=_claim("tension headache"),
        plan=_claim("follow up"),
    )
    now = datetime.now(timezone.utc)
    return (
        SoapReport(
            id=Id.new(),
            soap_notes=[note],
            created_at=now,
            updated_at=now,
        ),
        note,
    )


def _view(report: SoapReport, context_support):
    confidence = SoapConfidenceReport(
        id=Id.new(), soap_report_id=report.id, confidence_scores=[]
    )
    coding = SoapCodingReport(id=Id.new(), soap_report_id=report.id, codings=[])
    tier0 = SoapTier0Report(
        id=Id.new(),
        soap_report_id=report.id,
        results=[
            Tier0NoteResult(
                soap_note_id=report.soap_notes[0].id,
                passed=True,
                empty_sections=[],
                unresolved_claim_ids=[],
                citations_total=4,
                citations_resolved=4,
            )
        ],
    )
    return to_view(report, confidence, coding, tier0, context_support)


def test_context_gate_resolves_exact_refs_and_flags_unknown_refs():
    report, note = _report()
    known = ClinicalContextResource(
        reference="Condition/history",
        resource_type="Condition",
        category="condition",
        display="Migraine",
    )
    extraction = SoapExtraction(
        report=report,
        requested_context=[
            RequestedContextSupport(
                soap_note_id=note.id,
                section="assessment",
                references=[
                    " Condition/history ",
                    "Condition/history",
                    "Condition/not-in-snapshot",
                ],
            )
        ],
    )
    prepared = PreparedClinicalContext(
        status=ContextStatus.AVAILABLE,
        context=ClinicalContextInput(
            patient_ref="Patient/p1",
            encounter_ref="Encounter/e1",
            resources=(known,),
        ),
    )

    support = validate_context_support(extraction, prepared)
    view = _view(report, support)

    assessment = view.notes[0].assessment
    assert [ref.reference for ref in assessment.context_references] == [
        "Condition/history"
    ]
    assert assessment.invalid_context_references == [
        "Condition/not-in-snapshot"
    ]
    assert assessment.evidence_text == "tension headache"
    assert view.context_status is ContextStatus.AVAILABLE
    assert view.notes[0].needs_review is True


def test_unavailable_context_keeps_report_and_forces_review():
    report, _ = _report()
    extraction = SoapExtraction(report=report)
    prepared = PreparedClinicalContext(
        status=ContextStatus.UNAVAILABLE,
        error="Mock EHR is unavailable",
    )

    support = validate_context_support(extraction, prepared)
    view = _view(report, support)

    assert view.context_status is ContextStatus.UNAVAILABLE
    assert view.context_error == "Mock EHR is unavailable"
    assert view.notes[0].assessment.context_references == []
    assert view.notes[0].plan.context_references == []
    assert view.notes[0].needs_review is True
