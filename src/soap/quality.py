"""Deterministic online-quality diff for generated and doctor-verified SOAP.

The original :class:`~soap.soap.SoapReport` is compared with its verified
:class:`~soap.correction.SoapReportCorrection`.  Corrected notes are matched to
their originals exclusively by ``source_note_id``; text similarity is never
used for lineage.  Doctor-added and removed notes are counted separately and
therefore do not inflate the character-level edit distance.

The diff is deliberately pure: no database, API, LLM or wall-clock access.
Only claim text contributes to the Levenshtein distance.  Identities,
citations, confidence values and ICD metadata are excluded; ICD changes are
reported by the separate diagnosis metric.
"""

from __future__ import annotations

from dataclasses import dataclass

from .correction import CorrectedNote, CorrectionStatus, SoapReportCorrection
from .soap import AssessmentClaim, IcdCoding, SoapNote, SoapNoteId, SoapReport


class SoapQualityError(Exception):
    """Base class for deterministic SOAP-quality diff errors."""


class CorrectionNotVerifiedError(SoapQualityError):
    """Raised when quality is requested for an editable draft correction."""


class CorrectionSourceMismatchError(SoapQualityError):
    """Raised when a correction belongs to a different source report."""


@dataclass(frozen=True, slots=True)
class SoapNoteQualityDiff:
    """Character and diagnosis changes for one matched source/corrected note."""

    source_note_id: SoapNoteId
    corrected_note_id: SoapNoteId
    changed_characters: int
    diagnosis_changed: bool


@dataclass(frozen=True, slots=True)
class SoapReportQualityDiff:
    """Online-quality aggregates plus per-note detail for one SOAP report."""

    notes_added: int
    notes_removed: int
    changed_characters: int
    diagnosis_changes: int
    note_diffs: list[SoapNoteQualityDiff]


def calculate_soap_report_diff(
    source: SoapReport,
    correction: SoapReportCorrection,
) -> SoapReportQualityDiff:
    """Compare ``source`` with its VERIFIED doctor correction.

    Matched notes follow the source report's order, making the result stable
    regardless of dictionary ordering.  Added and removed notes are counted but
    omitted from ``note_diffs`` and from both change aggregates.
    """
    if correction.status is CorrectionStatus.DRAFT:
        raise CorrectionNotVerifiedError("SOAP quality requires a verified correction")
    if correction.source_report_id != source.id:
        raise CorrectionSourceMismatchError(
            "SOAP correction does not belong to the supplied source report"
        )

    corrected_by_source = {
        note.source_note_id: note for note in correction.notes if note.source_note_id is not None
    }
    notes_added = sum(note.source_note_id is None for note in correction.notes)

    note_diffs: list[SoapNoteQualityDiff] = []
    notes_removed = 0
    for source_note in source.notes:
        corrected_note = corrected_by_source.get(source_note.id)
        if corrected_note is None:
            notes_removed += 1
            continue
        note_diffs.append(_diff_note(source_note, corrected_note))

    return SoapReportQualityDiff(
        notes_added=notes_added,
        notes_removed=notes_removed,
        changed_characters=sum(diff.changed_characters for diff in note_diffs),
        diagnosis_changes=sum(diff.diagnosis_changed for diff in note_diffs),
        note_diffs=note_diffs,
    )


def _diff_note(source: SoapNote, corrected: CorrectedNote) -> SoapNoteQualityDiff:
    return SoapNoteQualityDiff(
        source_note_id=source.id,
        corrected_note_id=corrected.id,
        changed_characters=_levenshtein_distance(
            _canonical_note_text(source),
            _canonical_note_text(corrected),
        ),
        diagnosis_changed=(
            _assessment_text(source.assessment) != _assessment_text(corrected.assessment)
            or _icd_signature(source.assessment) != _icd_signature(corrected.assessment)
        ),
    )


def _canonical_note_text(note: SoapNote | CorrectedNote) -> str:
    """Flatten claim text in canonical S/O/A/P and within-section order."""
    return "\n".join(
        _normalize_newlines(claim.text) for _, claims in note.sections() for claim in claims
    )


def _assessment_text(claims: list[AssessmentClaim]) -> str:
    return "\n".join(_normalize_newlines(claim.text) for claim in claims)


def _icd_signature(
    claims: list[AssessmentClaim],
) -> tuple[tuple[str, str, str] | None, ...]:
    return tuple(_icd_tuple(claim.icd) for claim in claims)


def _icd_tuple(icd: IcdCoding | None) -> tuple[str, str, str] | None:
    if icd is None:
        return None
    return (icd.code, icd.name, icd.classifier_url)


def _normalize_newlines(text: str) -> str:
    """Normalize CRLF and bare CR only; preserve every other character."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _levenshtein_distance(left: str, right: str) -> int:
    """Return unit-cost character Levenshtein distance using two DP rows."""
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_character != right_character)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]
