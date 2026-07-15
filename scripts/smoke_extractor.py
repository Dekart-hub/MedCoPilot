#!/usr/bin/env python3
"""Manual acceptance runner for the SOAP extractor  [#7/FR-2][#7/NFR-2].

A QA-runs-it-by-hand smoke test — not a pytest suite and not wired into CI. It
pushes a few varied doctor-patient dialogues through the *real* LLM extractor
(MedGemma served by vLLM), checks each :class:`~soap.soap.SoapReport` against the
SOAP schema invariants, and prints a human-readable pass/fail report. All
invariant breaches are collected per dialogue rather than failing on the first,
so one run surfaces every problem. The process exits non-zero if any dialogue
fails.

Run it against a live vLLM server (see the README section "Manual acceptance"):

    VLLM_BASE_URL=http://localhost:8001/v1 MODEL_ID=google/medgemma-4b-it \
        PYTHONPATH=src uv run python scripts/smoke_extractor.py
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from config.settings import Settings
from dialogue.dialogue import Dialogue, DialogueTurnId
from infra.llm import build_llm_extractor
from soap.llm_extractor import LlmSoapExtractor
from soap.serialization import report_to_dict
from soap.soap import SoapClaim, SoapNote, SoapReport, SoapSection


@dataclass(frozen=True)
class Fixture:
    """A named dialogue plus the patient context to extract it under."""

    name: str
    patient_context: str
    turns: list[tuple[str, str]]

    def dialogue(self) -> Dialogue:
        dialogue = Dialogue.start()
        for speaker, text in self.turns:
            dialogue.add_turn(speaker, text)
        return dialogue


FIXTURES: list[Fixture] = [
    Fixture(
        name="pneumonia",
        patient_context="45-year-old male, no chronic conditions, no known allergies.",
        turns=[
            ("doctor", "What brings you in today?"),
            (
                "patient",
                "I've had a fever and a productive cough for three days, and I feel short "
                "of breath.",
            ),
            (
                "doctor",
                "Any chest pain? Let me listen. I hear crackles in the right lower lobe, "
                "temp is 38.6.",
            ),
            ("patient", "Yes, it hurts when I breathe deeply."),
            (
                "doctor",
                "This looks like community-acquired pneumonia. I'll start amoxicillin and "
                "order a chest X-ray. Follow up in three days.",
            ),
        ],
    ),
    Fixture(
        name="hypertension",
        patient_context="58-year-old woman, BMI 29, family history of stroke, on no "
        "antihypertensive.",
        turns=[
            (
                "doctor",
                "Your home readings have been running high — what numbers are you seeing?",
            ),
            (
                "patient",
                "Most mornings it's around 150 over 95, sometimes higher, and I get "
                "headaches at the back of my head.",
            ),
            ("doctor", "Any chest pain, palpitations, or visual changes?"),
            (
                "patient",
                "No chest pain, but occasional palpitations. My father had a stroke at 60.",
            ),
            (
                "doctor",
                "In clinic today your blood pressure is 152 over 96, pulse 78 and regular, "
                "heart and lungs normal.",
            ),
            (
                "doctor",
                "This is stage 2 hypertension. I'll start amlodipine 5 mg daily, cut back on "
                "salt, and recheck in four weeks.",
            ),
        ],
    ),
    Fixture(
        name="back-pain",
        patient_context="34-year-old warehouse worker, no prior back problems, no red-flag "
        "symptoms.",
        turns=[
            ("doctor", "What's going on with your back?"),
            (
                "patient",
                "I lifted a heavy box at work four days ago and my lower back has ached "
                "ever since.",
            ),
            (
                "doctor",
                "Does the pain travel down your legs? Any numbness, tingling, or bladder trouble?",
            ),
            ("patient", "It stays in the lower back, no numbness, and my bladder is fine."),
            (
                "doctor",
                "On exam there's tenderness over the lumbar muscles, straight-leg raise is "
                "negative, and reflexes and strength are normal.",
            ),
            (
                "doctor",
                "This is a mechanical lumbar strain. Take ibuprofen with food, stay gently "
                "active, and come back if you develop leg weakness or numbness.",
            ),
        ],
    ),
]


def _validate(report: SoapReport, turn_ids: set[DialogueTurnId]) -> list[str]:
    """Return every SOAP-invariant breach in ``report`` (empty list means pass)."""
    if not report.notes:
        return ["report has no SOAP notes (expected >= 1)"]
    failures: list[str] = []
    for position, note in enumerate(report.notes, start=1):
        failures.extend(_validate_note(note, position, turn_ids))
    return failures


def _validate_note(note: SoapNote, position: int, turn_ids: set[DialogueTurnId]) -> list[str]:
    label = f"note {position}"
    failures: list[str] = []
    sections = note.sections()
    present = [section for section, _ in sections]
    if present != list(SoapSection):
        failures.append(f"{label}: sections are {present}, expected {list(SoapSection)}")
    if not any(claims for _, claims in sections):
        failures.append(f"{label}: has no claims in any section")
    for section, claims in sections:
        for claim in claims:
            failures.extend(_validate_claim(claim, f"{label}/{section.value}", turn_ids))
    return failures


def _validate_claim(claim: SoapClaim, where: str, turn_ids: set[DialogueTurnId]) -> list[str]:
    if not claim.citations:
        return [f"{where}: claim {claim.text!r} cites no dialogue turn"]
    return [
        f"{where}: claim {claim.text!r} cites unknown turn {citation.turn_id}"
        for citation in claim.citations
        if citation.turn_id not in turn_ids
    ]


def _report_overview(report: SoapReport) -> str:
    lines = [f"notes: {len(report.notes)}"]
    for position, note in enumerate(report.notes, start=1):
        counts = ", ".join(f"{section.value}={len(claims)}" for section, claims in note.sections())
        lines.append(f"  note {position}: {counts}")
    return "\n".join(lines)


def _dump(report: SoapReport) -> str:
    return json.dumps(report_to_dict(report), ensure_ascii=False, indent=2)


async def _check(extractor: LlmSoapExtractor, fixture: Fixture) -> bool:
    dialogue = fixture.dialogue()
    print(f"\n=== {fixture.name} === ({len(dialogue.turns)} turns)")
    try:
        report = await extractor.extract(dialogue, fixture.patient_context)
    except Exception as exc:
        # A QA runner reports the failure and moves on; it never crashes mid-run.
        print(f"FAIL\n  - extraction raised: {exc}")
        return False
    failures = _validate(report, {turn.id for turn in dialogue.turns})
    print(_report_overview(report))
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  - {failure}")
        print(_dump(report))
        return False
    print("PASS")
    return True


async def _run() -> int:
    extractor = build_llm_extractor(Settings())
    results = [await _check(extractor, fixture) for fixture in FIXTURES]
    passed = sum(results)
    print(f"\n{'=' * 40}")
    print(f"RESULT: {passed}/{len(results)} dialogues passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
