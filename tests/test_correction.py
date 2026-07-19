"""Unit tests for the SOAP-correction domain: lineage, states and invariants."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from shared.value_objects import Id
from soap.correction import (
    CorrectedNote,
    CorrectionNotEditable,
    CorrectionStatus,
    DuplicateSourceNote,
    EmptyDoctorId,
    NoteNotInCorrection,
    SoapReportCorrection,
)
from soap.soap import (
    AssessmentClaim,
    IcdCoding,
    SoapClaim,
    SoapNote,
    SoapNoteId,
    SoapReport,
    TurnCitation,
)

_CREATED = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)
_LATER = datetime(2026, 7, 14, 10, 0, tzinfo=UTC)


def _claim(text: str = "Headache for three days.") -> SoapClaim:
    return SoapClaim(id=Id.new(), text=text, citations=[TurnCitation(turn_id=Id.new())])


def _assessment(text: str = "Essential hypertension.") -> AssessmentClaim:
    return AssessmentClaim(
        id=Id.new(),
        text=text,
        citations=[TurnCitation(turn_id=Id.new())],
        icd=IcdCoding(code="I10", name="Essential hypertension", classifier_url="icd://I10"),
    )


def _source_report() -> SoapReport:
    note = SoapNote(id=Id.new(), subjective=[_claim()], assessment=[_assessment()])
    return SoapReport(id=Id.new(), notes=[note])


def test_start_produces_a_draft_copy_linked_to_the_source() -> None:
    source = _source_report()

    correction = SoapReportCorrection.start(source, created_at=_CREATED)

    assert correction.status is CorrectionStatus.DRAFT
    assert correction.source_report_id == source.id
    assert correction.verified_by is None and correction.verified_at is None
    assert [note.source_note_id for note in correction.notes] == [source.notes[0].id]


def test_editing_the_correction_does_not_mutate_the_source_report() -> None:
    source = _source_report()
    correction = SoapReportCorrection.start(source, created_at=_CREATED)

    corrected_claim = correction.notes[0].subjective[0]
    assert corrected_claim is not source.notes[0].subjective[0]
    corrected_claim.text = "Rewritten by the doctor."
    correction.delete_note(correction.notes[0].id, at=_LATER)

    assert source.notes[0].subjective[0].text == "Headache for three days."
    assert len(source.notes) == 1


def test_added_note_carries_no_source_lineage() -> None:
    correction = SoapReportCorrection.start(_source_report(), created_at=_CREATED)

    added = correction.add_note(at=_LATER, subjective=[_claim("New finding.")])

    assert added.source_note_id is None
    assert correction.updated_at == _LATER


def test_deleted_note_is_determinable_from_missing_source_lineage() -> None:
    source = _source_report()
    correction = SoapReportCorrection.start(source, created_at=_CREATED)

    correction.delete_note(correction.notes[0].id, at=_LATER)

    live_sources = {note.source_note_id for note in correction.notes}
    assert source.notes[0].id not in live_sources


def test_update_note_replaces_section_claims_while_keeping_identity() -> None:
    source = _source_report()
    correction = SoapReportCorrection.start(source, created_at=_CREATED)
    note = correction.notes[0]

    correction.update_note(note.id, at=_LATER, plan=[_claim("Start amlodipine.")])

    assert note.source_note_id == source.notes[0].id
    assert [claim.text for claim in note.plan] == ["Start amlodipine."]
    assert note.subjective == []


@pytest.mark.parametrize(
    "edit",
    [
        lambda c, note_id: c.add_note(at=_LATER),
        lambda c, note_id: c.update_note(note_id, at=_LATER),
        lambda c, note_id: c.delete_note(note_id, at=_LATER),
    ],
)
def test_editing_a_verified_correction_is_rejected(
    edit: Callable[[SoapReportCorrection, SoapNoteId], object],
) -> None:
    correction = SoapReportCorrection.start(_source_report(), created_at=_CREATED)
    note_id = correction.notes[0].id
    correction.verify("dr-house", at=_LATER)

    with pytest.raises(CorrectionNotEditable):
        edit(correction, note_id)


def test_editing_a_note_absent_from_the_correction_is_rejected() -> None:
    correction = SoapReportCorrection.start(_source_report(), created_at=_CREATED)

    with pytest.raises(NoteNotInCorrection):
        correction.update_note(Id.new(), at=_LATER)


@pytest.mark.parametrize("doctor_id", ["", "   "])
def test_verify_requires_a_non_empty_doctor_id(doctor_id: str) -> None:
    correction = SoapReportCorrection.start(_source_report(), created_at=_CREATED)

    with pytest.raises(EmptyDoctorId):
        correction.verify(doctor_id, at=_LATER)
    assert correction.status is CorrectionStatus.DRAFT


def test_verify_transitions_to_verified_and_stamps_the_doctor() -> None:
    correction = SoapReportCorrection.start(_source_report(), created_at=_CREATED)

    correction.verify("dr-house", at=_LATER)

    assert correction.status is CorrectionStatus.VERIFIED
    assert correction.verified_by == "dr-house"
    assert correction.verified_at == _LATER


def test_reopen_returns_to_draft_and_clears_verification() -> None:
    correction = SoapReportCorrection.start(_source_report(), created_at=_CREATED)
    correction.verify("dr-house", at=_LATER)

    correction.reopen(at=_LATER)

    assert correction.status is CorrectionStatus.DRAFT
    assert correction.verified_by is None and correction.verified_at is None


def test_reopening_restores_editability() -> None:
    correction = SoapReportCorrection.start(_source_report(), created_at=_CREATED)
    correction.verify("dr-house", at=_LATER)
    correction.reopen(at=_LATER)

    added = correction.add_note(at=_LATER, objective=[_claim("BP 120/80.")])

    assert added in correction.notes


def test_a_source_note_may_appear_at_most_once() -> None:
    shared_source: SoapNoteId = Id.new()

    with pytest.raises(DuplicateSourceNote):
        SoapReportCorrection(
            id=Id.new(),
            source_report_id=Id.new(),
            status=CorrectionStatus.DRAFT,
            created_at=_CREATED,
            updated_at=_CREATED,
            notes=[
                CorrectedNote(id=Id.new(), source_note_id=shared_source),
                CorrectedNote(id=Id.new(), source_note_id=shared_source),
            ],
        )
