"""Exact-value tests for the deterministic SOAP online-quality diff (T22)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from shared.value_objects import Id
from soap.correction import CorrectedNote, CorrectionStatus, SoapReportCorrection
from soap.quality import (
    CorrectionNotVerifiedError,
    CorrectionSourceMismatchError,
    SoapReportQualityDiff,
    calculate_soap_report_diff,
)
from soap.soap import (
    AssessmentClaim,
    IcdCoding,
    SoapClaim,
    SoapNote,
    SoapReport,
    TurnCitation,
)

_CREATED = datetime(2026, 7, 20, 9, 0, tzinfo=UTC)
_VERIFIED = datetime(2026, 7, 20, 10, 0, tzinfo=UTC)
_I10 = IcdCoding(
    code="I10",
    name="Essential hypertension",
    classifier_url="https://icd.example/I10",
)


def _claim(text: str) -> SoapClaim:
    return SoapClaim(
        id=Id.new(),
        text=text,
        citations=[TurnCitation(turn_id=Id.new(), quote="source")],
    )


def _assessment(text: str, icd: IcdCoding | None = _I10) -> AssessmentClaim:
    return AssessmentClaim(
        id=Id.new(),
        text=text,
        citations=[TurnCitation(turn_id=Id.new(), quote="source")],
        icd=icd,
    )


def _note(
    *,
    subjective: list[str] | None = None,
    objective: list[str] | None = None,
    assessment: list[tuple[str, IcdCoding | None]] | None = None,
    plan: list[str] | None = None,
    confidence: float | None = None,
) -> SoapNote:
    return SoapNote(
        id=Id.new(),
        subjective=[_claim(text) for text in subjective or []],
        objective=[_claim(text) for text in objective or []],
        assessment=[_assessment(text, icd) for text, icd in assessment or []],
        plan=[_claim(text) for text in plan or []],
        confidence=confidence,
    )


def _report(*notes: SoapNote) -> SoapReport:
    return SoapReport(id=Id.new(), notes=list(notes))


def _verified_correction(
    source: SoapReport,
    edit: Callable[[SoapReportCorrection], None] | None = None,
) -> SoapReportCorrection:
    correction = SoapReportCorrection.start(source, created_at=_CREATED)
    if edit is not None:
        edit(correction)
    correction.verify("doctor-1", at=_VERIFIED)
    return correction


def _single_subjective_diff(source_text: str, corrected_text: str) -> SoapReportQualityDiff:
    source = _report(_note(subjective=[source_text]))

    def edit(correction: SoapReportCorrection) -> None:
        correction.notes[0].subjective = [_claim(corrected_text)]

    return calculate_soap_report_diff(source, _verified_correction(source, edit))


def test_unchanged_report_has_zero_metrics() -> None:
    source = _report(
        _note(
            subjective=["Headache for three days."],
            objective=["BP 140/90."],
            assessment=[("Essential hypertension.", _I10)],
            plan=["Monitor blood pressure."],
            confidence=0.97,
        )
    )

    result = calculate_soap_report_diff(source, _verified_correction(source))

    assert result.notes_added == 0
    assert result.notes_removed == 0
    assert result.changed_characters == 0
    assert result.diagnosis_changes == 0
    assert len(result.note_diffs) == 1
    assert result.note_diffs[0].changed_characters == 0
    assert result.note_diffs[0].diagnosis_changed is False


def test_quality_remains_available_while_pending_and_after_publication() -> None:
    source = _report(_note(subjective=["Headache."]))
    correction = _verified_correction(source)
    correction.begin_publication(at=_VERIFIED)

    pending = calculate_soap_report_diff(source, correction)
    correction.mark_published(at=_VERIFIED)
    published = calculate_soap_report_diff(source, correction)

    assert pending == published


@pytest.mark.parametrize(
    ("source_text", "corrected_text"),
    [
        ("pain", "paint"),  # insertion
        ("paint", "pain"),  # deletion
        ("pain", "gain"),  # substitution
        ("боль", "соль"),  # Cyrillic substitution
    ],
)
def test_single_character_edit_has_exact_distance_one(
    source_text: str,
    corrected_text: str,
) -> None:
    result = _single_subjective_diff(source_text, corrected_text)

    assert result.changed_characters == 1
    assert result.note_diffs[0].changed_characters == 1


def test_only_line_endings_are_normalized() -> None:
    result = _single_subjective_diff("first\r\nsecond\rthird", "first\nsecond\nthird")

    assert result.changed_characters == 0


def test_spaces_are_preserved_and_count_as_changes() -> None:
    result = _single_subjective_diff("headache", "headache ")

    assert result.changed_characters == 1


def test_empty_sections_have_zero_distance() -> None:
    source = _report(_note())

    result = calculate_soap_report_diff(source, _verified_correction(source))

    assert result.changed_characters == 0
    assert result.note_diffs[0].changed_characters == 0


def test_reordering_claims_has_exact_character_distance() -> None:
    source = _report(_note(subjective=["a", "b"]))

    def edit(correction: SoapReportCorrection) -> None:
        correction.notes[0].subjective.reverse()

    result = calculate_soap_report_diff(source, _verified_correction(source, edit))

    # Canonical text changes from "a\nb" to "b\na": two substitutions.
    assert result.changed_characters == 2


def test_added_and_removed_notes_are_counted_but_excluded_from_diff() -> None:
    kept = _note(subjective=["Kept note."])
    removed = _note(assessment=[("Removed diagnosis.", _I10)])
    source = _report(kept, removed)

    def edit(correction: SoapReportCorrection) -> None:
        correction.delete_note(correction.notes[1].id, at=_VERIFIED)
        correction.add_note(
            at=_VERIFIED,
            assessment=[
                _assessment(
                    "Doctor-added diagnosis.",
                    IcdCoding("J18.9", "Pneumonia", "https://icd.example/J18.9"),
                )
            ],
        )

    result = calculate_soap_report_diff(source, _verified_correction(source, edit))

    assert result.notes_added == 1
    assert result.notes_removed == 1
    assert result.changed_characters == 0
    assert result.diagnosis_changes == 0
    assert len(result.note_diffs) == 1
    assert result.note_diffs[0].source_note_id == kept.id


@pytest.mark.parametrize(
    "corrected_icd",
    [
        IcdCoding("I11", _I10.name, _I10.classifier_url),
        IcdCoding(_I10.code, "Renamed hypertension", _I10.classifier_url),
        IcdCoding(_I10.code, _I10.name, "https://icd.example/changed"),
        None,
    ],
)
def test_icd_only_change_is_a_diagnosis_change_but_not_a_character_change(
    corrected_icd: IcdCoding | None,
) -> None:
    source = _report(_note(assessment=[("Essential hypertension.", _I10)]))

    def edit(correction: SoapReportCorrection) -> None:
        correction.notes[0].assessment[0].icd = corrected_icd

    result = calculate_soap_report_diff(source, _verified_correction(source, edit))

    assert result.changed_characters == 0
    assert result.diagnosis_changes == 1
    assert result.note_diffs[0].diagnosis_changed is True


def test_assessment_text_change_affects_both_metrics() -> None:
    source = _report(_note(assessment=[("Hypertension", _I10)]))

    def edit(correction: SoapReportCorrection) -> None:
        correction.notes[0].assessment[0].text = "hypertension"

    result = calculate_soap_report_diff(source, _verified_correction(source, edit))

    assert result.changed_characters == 1
    assert result.diagnosis_changes == 1
    assert result.note_diffs[0].diagnosis_changed is True


def test_report_aggregates_are_sums_of_matched_note_diffs() -> None:
    first = _note(subjective=["pain"])
    second = _note(plan=["rest"])
    source = _report(first, second)

    def edit(correction: SoapReportCorrection) -> None:
        correction.notes[0].subjective[0].text = "gain"
        correction.notes[1].plan[0].text = "rests"

    result = calculate_soap_report_diff(source, _verified_correction(source, edit))

    assert [diff.source_note_id for diff in result.note_diffs] == [first.id, second.id]
    assert [diff.changed_characters for diff in result.note_diffs] == [1, 1]
    assert result.changed_characters == 2
    assert result.diagnosis_changes == 0


def test_ids_citations_confidence_and_icd_are_excluded_from_character_diff() -> None:
    source = _report(
        _note(
            subjective=["Same claim."],
            assessment=[("Same diagnosis.", _I10)],
            confidence=0.01,
        )
    )

    def edit(correction: SoapReportCorrection) -> None:
        subjective = correction.notes[0].subjective[0]
        subjective.id = Id.new()
        subjective.citations = [TurnCitation(turn_id=Id.new(), quote="different evidence")]
        assessment = correction.notes[0].assessment[0]
        assessment.id = Id.new()
        assessment.citations = [TurnCitation(turn_id=Id.new(), quote="other evidence")]
        assessment.icd = IcdCoding("I11", "Other code", "https://icd.example/I11")

    result = calculate_soap_report_diff(source, _verified_correction(source, edit))

    assert result.changed_characters == 0
    assert result.diagnosis_changes == 1


def test_draft_correction_is_rejected() -> None:
    source = _report(_note(subjective=["Draft."]))
    correction = SoapReportCorrection.start(source, created_at=_CREATED)

    with pytest.raises(CorrectionNotVerifiedError):
        calculate_soap_report_diff(source, correction)


def test_correction_for_another_report_is_rejected() -> None:
    source = _report(_note(subjective=["First report."]))
    other = _report(_note(subjective=["Other report."]))

    with pytest.raises(CorrectionSourceMismatchError):
        calculate_soap_report_diff(source, _verified_correction(other))


def test_result_keeps_corrected_note_identity() -> None:
    source = _report(_note(subjective=["Stable."]))
    correction = _verified_correction(source)

    result = calculate_soap_report_diff(source, correction)

    assert result.note_diffs[0].source_note_id == source.notes[0].id
    assert result.note_diffs[0].corrected_note_id == correction.notes[0].id


def test_verified_status_is_required_even_when_fields_are_populated() -> None:
    source = _report(_note())
    correction = SoapReportCorrection(
        id=Id.new(),
        source_report_id=source.id,
        status=CorrectionStatus.DRAFT,
        created_at=_CREATED,
        updated_at=_VERIFIED,
        notes=[CorrectedNote(id=Id.new(), source_note_id=source.notes[0].id)],
        verified_by="doctor-1",
        verified_at=_VERIFIED,
    )

    with pytest.raises(CorrectionNotVerifiedError):
        calculate_soap_report_diff(source, correction)
