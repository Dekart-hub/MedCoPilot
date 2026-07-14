"""JSON serialization for the SOAP domain.

Kept out of the domain proper: turning a :class:`SoapReport` into a
JSON-compatible ``dict`` is a boundary concern. Identities and enums are
rendered as strings so the result is directly ``json.dumps``-able.
"""

from __future__ import annotations

from typing import Any

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
