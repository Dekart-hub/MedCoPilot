from __future__ import annotations

from datetime import datetime, timezone

from app.api.v1.schemas import ReportResponse
from shared.value_objects import Id
from soap.view import AssessmentView, ClaimView, NoteView, ReportView, Tier0View


def _claim_view(text: str, *, score: float | None, flagged: bool) -> ClaimView:
    return ClaimView(
        id=Id.new(),
        claim=text,
        evidence_text=text,
        turn_id=Id.new(),
        grounding_score=score,
        is_flagged=flagged,
    )


def test_report_response_carries_gate_and_flags():
    now = datetime.now(timezone.utc)
    note = NoteView(
        id=Id.new(),
        subjective=_claim_view("headache", score=0.9, flagged=False),
        objective=_claim_view("fever of 39", score=0.1, flagged=True),
        assessment=AssessmentView(
            id=Id.new(),
            claim="tension headache",
            evidence_text="tension headache",
            turn_id=Id.new(),
            grounding_score=0.8,
            is_flagged=False,
        ),
        plan=_claim_view("rest", score=None, flagged=False),
        tier0=Tier0View(
            passed=False,
            empty_sections=["objective"],
            citations_total=3,
            citations_resolved=2,
        ),
        needs_review=True,
        confidence=0.6,
    )
    view = ReportView(id=Id.new(), notes=[note], created_at=now, updated_at=now)

    resp = ReportResponse.from_domain(view)
    body = resp.model_dump()
    note_body = body["soap_notes"][0]

    assert note_body["needs_review"] is True
    assert note_body["tier0"] == {
        "passed": False,
        "empty_sections": ["objective"],
        "citations_total": 3,
        "citations_resolved": 2,
    }
    assert note_body["subjective"]["grounding_score"] == 0.9
    assert note_body["objective"]["is_flagged"] is True
    assert note_body["assessment"]["grounding_score"] == 0.8
    assert note_body["plan"]["grounding_score"] is None
