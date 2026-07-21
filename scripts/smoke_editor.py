#!/usr/bin/env python3
"""Manual acceptance runner for the LLM SOAP edit agent  [#12/FR-1][#12/FR-2][#12/FR-3].

A QA-runs-it-by-hand smoke test — not a pytest suite and not wired into CI. It
builds a doctor correction from a canned dialogue, then asks the *real* edit
agent (MedGemma served by vLLM) to satisfy a plain edit request, and checks the
returned :class:`~soap.llm_editor.ProposalDraft` against the agent's contract:
every operation is add/update/delete, each update/delete targets a note that
exists in the correction, every proposed claim cites a real dialogue turn, and no
ICD coding can appear (the output schema has no channel for it). Breaches are
collected per fixture rather than failing on the first, so one run surfaces every
problem. The process exits non-zero if any fixture fails.

Run it against a live vLLM server (see the README section "Manual acceptance"):

    VLLM_BASE_URL=http://localhost:8001/v1 MODEL_ID=google/medgemma-4b-it \
        PYTHONPATH=src uv run python scripts/smoke_editor.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from config.settings import Settings
from dialogue.dialogue import Dialogue, DialogueTurnId
from infra.llm import build_soap_edit_agent
from shared.value_objects import Id
from soap.correction import SoapReportCorrection
from soap.llm_editor import EditContext, ProposalDraft, SoapEditAgent
from soap.proposal import (
    CorrectionEditorSession,
    DeleteNoteOperation,
    OperationSpec,
    ProposedClaim,
    UpdateNoteOperation,
)
from soap.soap import SoapClaim, SoapNote, SoapNoteId, SoapReport, TurnCitation


@dataclass(frozen=True)
class Fixture:
    """A dialogue, the notes to correct over it, and the edit request to run."""

    name: str
    patient_id: str
    request: str
    turns: list[tuple[str, str]]
    # Each note is section -> list of (text, 1-based turn number it cites).
    notes: list[dict[str, list[tuple[str, int]]]] = field(default_factory=list)

    def dialogue(self) -> Dialogue:
        dialogue = Dialogue.start()
        for speaker, text in self.turns:
            dialogue.add_turn(speaker, text)
        return dialogue

    def report(self, dialogue: Dialogue) -> SoapReport:
        return SoapReport(id=Id.new(), notes=[self._note(spec, dialogue) for spec in self.notes])

    def _note(self, spec: dict[str, list[tuple[str, int]]], dialogue: Dialogue) -> SoapNote:
        section = {
            name: [self._claim(text, turn, dialogue) for text, turn in claims]
            for name, claims in spec.items()
        }
        return SoapNote(id=Id.new(), **section)

    def _claim(self, text: str, turn: int, dialogue: Dialogue) -> SoapClaim:
        citation = TurnCitation(turn_id=dialogue.turns[turn - 1].id)
        return SoapClaim(id=Id.new(), text=text, citations=[citation])


FIXTURES: list[Fixture] = [
    Fixture(
        name="add-followup-plan",
        patient_id="P001",
        request="Add a short follow-up plan note telling the patient when to return.",
        turns=[
            ("doctor", "What brings you in today?"),
            ("patient", "I've had a fever and a productive cough for three days."),
            ("doctor", "I hear crackles in the right lower lobe; your temperature is 38.6."),
            ("doctor", "This looks like community-acquired pneumonia. Follow up in three days."),
        ],
        notes=[
            {
                "subjective": [("Fever and productive cough for three days.", 2)],
                "objective": [("Crackles in the right lower lobe, temperature 38.6.", 3)],
                "assessment": [("Community-acquired pneumonia.", 4)],
            }
        ],
    ),
    Fixture(
        name="tighten-assessment",
        patient_id="P002",
        request="Reword the assessment to name the hypertension stage more precisely.",
        turns=[
            ("patient", "My home readings are around 150 over 95 most mornings."),
            ("doctor", "In clinic today your blood pressure is 152 over 96, pulse 78."),
            ("doctor", "This is stage 2 hypertension. I'll start amlodipine 5 mg daily."),
        ],
        notes=[
            {
                "subjective": [("Home readings around 150/95 in the mornings.", 1)],
                "objective": [("In-clinic blood pressure 152/96, pulse 78.", 2)],
                "assessment": [("Hypertension.", 3)],
                "plan": [("Start amlodipine 5 mg daily.", 3)],
            }
        ],
    ),
]


def _validate(
    draft: ProposalDraft, note_ids: set[SoapNoteId], turn_ids: set[DialogueTurnId]
) -> list[str]:
    if not draft.operations:
        return ["draft carries no operations"]
    failures: list[str] = []
    for position, operation in enumerate(draft.operations, start=1):
        failures.extend(_validate_operation(operation, position, note_ids, turn_ids))
    return failures


def _validate_operation(
    operation: OperationSpec,
    position: int,
    note_ids: set[SoapNoteId],
    turn_ids: set[DialogueTurnId],
) -> list[str]:
    label = f"op {position}"
    if (
        isinstance(operation, DeleteNoteOperation | UpdateNoteOperation)
        and operation.target_note_id not in note_ids
    ):
        return [f"{label}: targets a note absent from the correction"]
    if isinstance(operation, DeleteNoteOperation):
        return []
    failures: list[str] = []
    for claim in _claims(operation):
        failures.extend(_validate_claim(claim, label, turn_ids))
        assert not hasattr(claim, "icd"), "a proposed claim must never carry an ICD coding"
    return failures


def _claims(operation: OperationSpec) -> list[ProposedClaim]:
    content = operation.content
    return [*content.subjective, *content.objective, *content.assessment, *content.plan]


def _validate_claim(claim: ProposedClaim, label: str, turn_ids: set[DialogueTurnId]) -> list[str]:
    if not claim.citations:
        return [f"{label}: proposed claim cites no dialogue turn"]
    return [
        f"{label}: proposed claim cites unknown turn {citation.turn_id}"
        for citation in claim.citations
        if citation.turn_id not in turn_ids
    ]


def _kind(operation: OperationSpec) -> str:
    if isinstance(operation, DeleteNoteOperation):
        return "delete"
    return "update" if isinstance(operation, UpdateNoteOperation) else "add"


async def _check(agent: SoapEditAgent, fixture: Fixture) -> bool:
    now = datetime.now(UTC)
    dialogue = fixture.dialogue()
    report = fixture.report(dialogue)
    correction = SoapReportCorrection.start(report, created_at=now)
    session = CorrectionEditorSession.start(correction.id, created_at=now)
    context = EditContext(
        dialogue=dialogue,
        report=report,
        correction=correction,
        session=session,
        patient_id=fixture.patient_id,
    )
    print(f"\n=== {fixture.name} === ({len(dialogue.turns)} turns)  request: {fixture.request!r}")
    try:
        draft = await agent.propose(context, user_request=fixture.request)
    except Exception as exc:
        print(f"FAIL\n  - propose raised: {exc}")
        return False
    note_ids = {note.id for note in correction.notes}
    turn_ids = {turn.id for turn in dialogue.turns}
    failures = _validate(draft, note_ids, turn_ids)
    kinds = ", ".join(_kind(operation) for operation in draft.operations)
    print(f"operations: {len(draft.operations)} [{kinds}] via {draft.model_id}")
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return False
    print("PASS")
    return True


async def _run() -> int:
    agent = build_soap_edit_agent(Settings())
    results = [await _check(agent, fixture) for fixture in FIXTURES]
    passed = sum(results)
    print(f"\n{'=' * 40}")
    print(f"RESULT: {passed}/{len(results)} fixtures passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
