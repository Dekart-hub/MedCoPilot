"""JSON serialization for the SOAP domain.

Kept out of the domain proper: turning a :class:`SoapReport` into a
JSON-compatible ``dict`` is a boundary concern. Identities and enums are
rendered as strings so the result is directly ``json.dumps``-able.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .correction import CorrectedNote, SoapReportCorrection
from .quality_use_cases import DialogueSoapQuality
from .repository import ReportSummary
from .soap import (
    AssessmentClaim,
    IcdCoding,
    SoapClaim,
    SoapNote,
    SoapReport,
    TurnCitation,
)


def report_to_dict(report: SoapReport) -> dict[str, Any]:
    """Serialize a report to a JSON-compatible dict."""
    return {
        "id": str(report.id),
        "notes": [_note_to_dict(note) for note in report.notes],
    }


def report_summary_to_dict(summary: ReportSummary) -> dict[str, Any]:
    """Serialize a report summary for the list view, ``created_at`` as ISO-8601."""
    return {
        "report_id": str(summary.report_id),
        "dialogue_id": str(summary.dialogue_id),
        "created_at": summary.created_at.isoformat(),
    }


def correction_to_dict(correction: SoapReportCorrection) -> dict[str, Any]:
    """Serialize a doctor's correction, including status and verification stamp."""
    return {
        "id": str(correction.id),
        "source_report_id": str(correction.source_report_id),
        "status": correction.status.value,
        "verified_by": correction.verified_by,
        "verified_at": _iso(correction.verified_at),
        "notes": [_corrected_note_to_dict(note) for note in correction.notes],
    }


def quality_to_dict(quality: DialogueSoapQuality) -> dict[str, Any]:
    """Serialize dialogue-level online SOAP quality to the REST response shape."""
    return {
        "dialogue_id": str(quality.dialogue_id),
        "report_id": str(quality.report_id),
        "correction_id": str(quality.correction_id),
        "notes_added": quality.notes_added,
        "notes_removed": quality.notes_removed,
        "changed_characters": quality.changed_characters,
        "diagnosis_changes": quality.diagnosis_changes,
        "note_diffs": [
            {
                "source_note_id": str(note.source_note_id),
                "corrected_note_id": str(note.corrected_note_id),
                "changed_characters": note.changed_characters,
                "diagnosis_changed": note.diagnosis_changed,
            }
            for note in quality.note_diffs
        ],
    }


def _corrected_note_to_dict(note: CorrectedNote) -> dict[str, Any]:
    return {
        "id": str(note.id),
        "source_note_id": str(note.source_note_id) if note.source_note_id is not None else None,
        "sections": {
            section.value: [_claim_to_dict(claim) for claim in claims]
            for section, claims in note.sections()
        },
    }


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment is not None else None


def _note_to_dict(note: SoapNote) -> dict[str, Any]:
    return {
        "id": str(note.id),
        "confidence": note.confidence,
        "sections": {
            section.value: [_claim_to_dict(claim) for claim in claims]
            for section, claims in note.sections()
        },
    }


def _claim_to_dict(claim: SoapClaim) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": str(claim.id),
        "text": claim.text,
        "citations": [_citation_to_dict(citation) for citation in claim.citations],
    }
    if isinstance(claim, AssessmentClaim):
        data["icd"] = _icd_to_dict(claim.icd) if claim.icd is not None else None
    return data


def _citation_to_dict(citation: TurnCitation) -> dict[str, Any]:
    return {"turn_id": str(citation.turn_id), "quote": citation.quote}


def _icd_to_dict(icd: IcdCoding) -> dict[str, str]:
    return {"code": icd.code, "name": icd.name, "classifier_url": icd.classifier_url}
