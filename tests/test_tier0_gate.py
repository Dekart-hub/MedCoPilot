from __future__ import annotations

from datetime import datetime, timezone

from dialogue import Dialogue, DialogueTurn
from shared.value_objects import Id
from soap.score.tier0 import run_tier0
from soap.soap import SoapClaim, SoapEvidence, SoapNote, SoapReport


def _turn(content: str) -> DialogueTurn:
    return DialogueTurn(
        id=Id.new(),
        role="patient",
        content=content,
        timestamp=datetime.now(timezone.utc),
    )


def _claim(text: str, quote: str, turn: DialogueTurn) -> SoapClaim:
    return SoapClaim(
        id=Id.new(),
        claim=text,
        evidence=SoapEvidence(text=quote, turn_id=turn.id),
    )


def _note(s: SoapClaim, o: SoapClaim, a: SoapClaim, p: SoapClaim) -> SoapNote:
    return SoapNote(id=Id.new(), subjective=s, objective=o, assessment=a, plan=p)


def _report(*notes: SoapNote) -> SoapReport:
    now = datetime.now(timezone.utc)
    return SoapReport(id=Id.new(), soap_notes=list(notes), created_at=now, updated_at=now)


def _dialogue(*turns: DialogueTurn) -> Dialogue:
    return Dialogue(id=Id.new(), turns=list(turns), created_at=datetime.now(timezone.utc))


def test_verbatim_citations_pass_the_gate():
    t = _turn("My chest feels tight on the stairs since yesterday")
    note = _note(
        _claim("chest tightness on exertion", "chest feels tight on the stairs", t),
        _claim("no exam performed", "since yesterday", t),
        _claim("query angina", "tight on the stairs", t),
        _claim("refer to cardiology", "My chest feels tight", t),
    )
    result = run_tier0(_dialogue(t), _report(note)).results[0]

    assert result.passed is True
    assert result.citations_total == 4
    assert result.citations_resolved == 4
    assert result.empty_sections == []
    assert result.unresolved_claim_ids == []


def test_match_ignores_case_and_extra_whitespace():
    t = _turn("Blood pressure is 130 over 85,  temperature normal")
    note = _note(
        _claim("s", "blood PRESSURE is 130 over 85", t),
        _claim("o", "Temperature   normal", t),
        _claim("a", "130 over 85", t),
        _claim("p", "blood pressure", t),
    )
    result = run_tier0(_dialogue(t), _report(note)).results[0]

    assert result.passed is True
    assert result.citations_resolved == 4


def test_fabricated_quote_fails_the_gate():
    t = _turn("I have a headache")
    good = _claim("headache", "I have a headache", t)
    fabricated = _claim("fever", "temperature is 39", t)
    note = _note(good, fabricated, good, good)
    result = run_tier0(_dialogue(t), _report(note)).results[0]

    assert result.passed is False
    assert result.citations_resolved == 3
    assert result.unresolved_claim_ids == [fabricated.id]


def test_quote_referencing_missing_turn_fails():
    t = _turn("I have a headache")
    orphan_turn = _turn("this turn is not part of the dialogue")
    orphan = _claim("headache", "I have a headache", orphan_turn)
    good = _claim("headache", "I have a headache", t)
    note = _note(orphan, good, good, good)
    result = run_tier0(_dialogue(t), _report(note)).results[0]

    assert result.passed is False
    assert orphan.id in result.unresolved_claim_ids


def test_empty_section_is_flagged_not_failed():
    t = _turn("I have a headache")
    good = _claim("headache", "I have a headache", t)
    empty = _claim("", "", t)
    note = _note(good, empty, good, good)
    result = run_tier0(_dialogue(t), _report(note)).results[0]

    assert result.passed is True
    assert result.empty_sections == ["objective"]
    assert result.citations_total == 3
    assert result.citations_resolved == 3


def test_gate_runs_per_note():
    t = _turn("I have a headache")
    good = _claim("headache", "I have a headache", t)
    fabricated = _claim("fever", "temperature is 39", t)
    ok_note = _note(good, good, good, good)
    bad_note = _note(good, fabricated, good, good)
    report = run_tier0(_dialogue(t), _report(ok_note, bad_note))

    assert [r.passed for r in report.results] == [True, False]
    assert report.results[0].soap_note_id == ok_note.id
    assert report.results[1].soap_note_id == bad_note.id
