#!/usr/bin/env python3
"""End-to-end happy-path scenario against a LIVE MedCoPilot stack  [#7/NFR-4].

A QA-runs-it-by-hand integration check — **not** a pytest test and not wired
into CI. It proves the whole stack works together over HTTP: it POSTs a
doctor-patient dialogue, asks the service to extract its SOAP report, reads the
persisted report back, and asserts the response is a well-formed clinical
document — at least one note, all four S/O/A/P sections populated, an ICD coding
on the Assessment, a per-note groundedness confidence, and every claim traced to
a real dialogue turn. All breaches are collected and printed together; the
process exits non-zero if any assertion fails.

Point it at a running app (see the README "Demo / E2E" section):

    BASE_URL=http://localhost:8000 PYTHONPATH=src uv run python scripts/e2e_smoke.py

The app must be wired to a live MedGemma (``VLLM_BASE_URL``) with the NLI
confidence scorer enabled (``NLI_CONFIDENCE_ENABLED=1``) so ``confidence`` is
populated.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
PATIENT_ID = os.environ.get("PATIENT_ID", "P001")
# One extraction fans out several schema-guided MedGemma calls, so allow a
# generous wall-clock budget for the report POST.
REQUEST_TIMEOUT = float(os.environ.get("E2E_TIMEOUT", "180"))

SOAP_SECTIONS = ("subjective", "objective", "assessment", "plan")

# A full-encounter dialogue that exercises all four SOAP dimensions: a subjective
# complaint, objective exam findings, a codeable assessment (pneumonia) and a
# plan. Kept verbatim here so the scenario is self-contained and reproducible.
DIALOGUE_TURNS: list[tuple[str, str]] = [
    ("doctor", "What brings you in today?"),
    (
        "patient",
        "I've had a fever and a productive cough for three days, and I feel short of breath.",
    ),
    (
        "doctor",
        "Any chest pain? Let me listen. I hear crackles in the right lower lobe, "
        "and your temperature is 38.6.",
    ),
    ("patient", "Yes, it hurts when I breathe deeply."),
    (
        "doctor",
        "This looks like community-acquired pneumonia. I'll start amoxicillin and "
        "order a chest X-ray. Follow up in three days.",
    ),
]


def _create_dialogue(client: httpx.Client) -> str:
    """POST the dialogue and return its server-assigned id."""
    payload = {"turns": [{"speaker": speaker, "text": text} for speaker, text in DIALOGUE_TURNS]}
    response = client.post("/dialogues", json=payload)
    response.raise_for_status()
    return str(response.json()["id"])


def _extract_report(client: httpx.Client, dialogue_id: str) -> str:
    """Ask the service to extract the report for a dialogue and return its id."""
    response = client.post(f"/dialogues/{dialogue_id}/report", params={"patient_id": PATIENT_ID})
    response.raise_for_status()
    return str(response.json()["id"])


def _fetch_report(client: httpx.Client, report_id: str) -> dict[str, Any]:
    """Read the persisted report back, proving the round-trip through storage."""
    response = client.get(f"/reports/{report_id}")
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def _validate(report: dict[str, Any]) -> list[str]:
    """Return every happy-path assertion breach in ``report`` (empty ⇒ pass)."""
    notes = report.get("notes", [])
    if not notes:
        return ["report has no SOAP notes (expected >= 1)"]
    failures: list[str] = []
    for position, note in enumerate(notes, start=1):
        failures.extend(_validate_note(note, position))
    failures.extend(_validate_report_sections(notes))
    failures.extend(_validate_report_icd(notes))
    return failures


def _validate_note(note: dict[str, Any], position: int) -> list[str]:
    label = f"note {position}"
    return [
        *_validate_sections_present(note, label),
        *_validate_confidence(note, label),
        *_validate_citations(note, label),
    ]


def _validate_sections_present(note: dict[str, Any], label: str) -> list[str]:
    """Every note must expose all four SOAP section keys."""
    sections = note.get("sections", {})
    missing = [section for section in SOAP_SECTIONS if section not in sections]
    if missing:
        return [f"{label}: missing SOAP sections {missing}"]
    return []


def _validate_confidence(note: dict[str, Any], label: str) -> list[str]:
    """The NLI scorer must land a per-note confidence in [0, 1]."""
    confidence = note.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        return [f"{label}: confidence is {confidence!r}, expected a number in [0, 1]"]
    if not 0.0 <= confidence <= 1.0:
        return [f"{label}: confidence {confidence} is outside [0, 1]"]
    return []


def _validate_citations(note: dict[str, Any], label: str) -> list[str]:
    failures: list[str] = []
    sections = note.get("sections", {})
    for section in SOAP_SECTIONS:
        for claim in sections.get(section, []):
            failures.extend(_validate_claim_citations(claim, f"{label}/{section}"))
    return failures


def _validate_claim_citations(claim: dict[str, Any], where: str) -> list[str]:
    """Every claim must cite at least one real dialogue turn.

    The service grounds claims by construction — an unresolvable turn drops the
    claim — so a turn id reaching the client is genuine; here we assert it is
    present and a well-formed identity, the strongest check the HTTP surface
    (which never echoes the dialogue's turn ids back) allows.
    """
    citations = claim.get("citations", [])
    if not citations:
        return [f"{where}: claim {claim.get('text')!r} cites no dialogue turn"]
    return [
        f"{where}: claim cites malformed turn id {citation.get('turn_id')!r}"
        for citation in citations
        if not _is_uuid(citation.get("turn_id"))
    ]


def _validate_report_sections(notes: list[dict[str, Any]]) -> list[str]:
    """The report as a whole must have a claim in each of the four sections."""
    populated = {
        section
        for note in notes
        for section in SOAP_SECTIONS
        if note.get("sections", {}).get(section)
    }
    missing = [section for section in SOAP_SECTIONS if section not in populated]
    if missing:
        return [f"report has no claims in section(s) {missing} (expected all of S/O/A/P)"]
    return []


def _validate_report_icd(notes: list[dict[str, Any]]) -> list[str]:
    """At least one Assessment claim must carry a complete ICD coding."""
    codings = [
        claim.get("icd")
        for note in notes
        for claim in note.get("sections", {}).get("assessment", [])
    ]
    if not any(_is_complete_icd(icd) for icd in codings):
        return ["no Assessment claim carries a complete ICD coding (code + name + classifier_url)"]
    return []


def _is_complete_icd(icd: Any) -> bool:
    fields = ("code", "name", "classifier_url")
    return isinstance(icd, dict) and all(icd.get(field) for field in fields)


def _is_uuid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _print_overview(report: dict[str, Any]) -> None:
    notes = report["notes"]
    print(f"report {report['id']}: {len(notes)} note(s)")
    for position, note in enumerate(notes, start=1):
        counts = ", ".join(
            f"{section}={len(note['sections'].get(section, []))}" for section in SOAP_SECTIONS
        )
        print(f"  note {position}: confidence={note.get('confidence')} | {counts}")
        for claim in note["sections"].get("assessment", []):
            icd = claim.get("icd")
            if _is_complete_icd(icd):
                print(f"    ICD {icd['code']} {icd['name']!r} <- {claim['text']!r}")


def main() -> int:
    print(f"E2E happy-path against {BASE_URL} (patient {PATIENT_ID})")
    with httpx.Client(base_url=BASE_URL, timeout=REQUEST_TIMEOUT) as client:
        dialogue_id = _create_dialogue(client)
        print(f"created dialogue {dialogue_id} ({len(DIALOGUE_TURNS)} turns)")
        report_id = _extract_report(client, dialogue_id)
        print(f"extracted report {report_id}")
        report = _fetch_report(client, report_id)
    _print_overview(report)
    failures = _validate(report)
    print("-" * 60)
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("PASS: dialogue -> report with S/O/A/P + ICD + confidence + grounded citations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
