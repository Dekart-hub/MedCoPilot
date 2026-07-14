"""Unit tests for the SOAP domain invariants and JSON serialization."""

from __future__ import annotations

import json

import pytest

from dialogue.dialogue import Dialogue
from shared.value_objects import Id
from soap.serialization import report_to_dict
from soap.soap import (
    AssessmentClaim,
    IcdCoding,
    SoapClaim,
    SoapNote,
    SoapReport,
    SoapSection,
    TurnCitation,
)


def _dialogue() -> Dialogue:
    dialogue = Dialogue.start()
    dialogue.add_turn("patient", "I've had a headache for three days.")
    dialogue.add_turn("doctor", "Blood pressure is 140 over 90.")
    return dialogue


def test_claim_must_cite_at_least_one_turn() -> None:
    with pytest.raises(ValueError):
        SoapClaim(id=Id.new(), text="Headache for three days.", citations=[])


def test_claim_keeps_its_turn_citations() -> None:
    dialogue = _dialogue()
    turn = dialogue.turns[0]

    claim = SoapClaim(
        id=Id.new(),
        text="Headache for three days.",
        citations=[TurnCitation(turn_id=turn.id, quote="headache for three days")],
    )

    assert [citation.turn_id for citation in claim.citations] == [turn.id]


def test_a_citation_may_omit_the_verbatim_quote() -> None:
    turn = _dialogue().turns[0]

    citation = TurnCitation(turn_id=turn.id)

    assert citation.quote is None


def test_section_membership_is_unambiguous() -> None:
    dialogue = _dialogue()
    subjective = SoapClaim(
        id=Id.new(),
        text="Headache for three days.",
        citations=[TurnCitation(turn_id=dialogue.turns[0].id)],
    )
    assessment = AssessmentClaim(
        id=Id.new(),
        text="Essential hypertension.",
        citations=[TurnCitation(turn_id=dialogue.turns[1].id)],
        icd=IcdCoding(code="I10", name="Essential hypertension", classifier_url="icd://I10"),
    )
    note = SoapNote(id=Id.new(), subjective=[subjective], assessment=[assessment])

    section_of = {section: claims for section, claims in note.sections() if claims}
    assert list(section_of) == [SoapSection.SUBJECTIVE, SoapSection.ASSESSMENT]
    assert section_of[SoapSection.SUBJECTIVE] == [subjective]
    assert section_of[SoapSection.ASSESSMENT] == [assessment]


def test_report_is_a_list_of_notes_and_reserves_confidence() -> None:
    note = SoapNote(id=Id.new())
    report = SoapReport(id=Id.new(), notes=[note])

    assert report.notes == [note]
    assert note.confidence is None


def test_report_serializes_to_a_json_roundtrippable_dict() -> None:
    dialogue = _dialogue()
    assessment = AssessmentClaim(
        id=Id.new(),
        text="Essential hypertension.",
        citations=[TurnCitation(turn_id=dialogue.turns[1].id, quote="140 over 90")],
        icd=IcdCoding(code="I10", name="Essential hypertension", classifier_url="icd://I10"),
    )
    note = SoapNote(
        id=Id.new(),
        subjective=[
            SoapClaim(
                id=Id.new(),
                text="Headache for three days.",
                citations=[TurnCitation(turn_id=dialogue.turns[0].id)],
            )
        ],
        assessment=[assessment],
    )
    report = SoapReport(id=Id.new(), notes=[note])

    payload = report_to_dict(report)

    assert json.loads(json.dumps(payload)) == payload
    assert set(payload["notes"][0]["sections"]) == {section.value for section in SoapSection}
    assert payload["notes"][0]["sections"]["assessment"][0]["icd"]["code"] == "I10"
    assert payload["notes"][0]["sections"]["subjective"][0]["citations"][0]["quote"] is None
