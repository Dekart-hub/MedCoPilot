from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from dialogue import Dialogue, DialogueTurn
from shared.value_objects import Id
from soap.context import ClinicalContextInput, ClinicalContextResource
from soap.extractor.llm_extractor import (
    ClaimOut,
    ContextualClaimOut,
    LlmSoapExtractor,
    NoteOut,
    _render_clinical_context,
)


class StubAgent:
    """Подменяет LangGraphAgent: запоминает вход и возвращает готовое состояние."""

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state
        self.received_messages: Any = None

    async def run(
        self, messages: Any, config: Any = None
    ) -> dict[str, Any]:
        self.received_messages = messages
        return self._state


def _turn(role: str, content: str) -> DialogueTurn:
    return DialogueTurn(
        id=Id.new(),
        role=role,
        content=content,
        timestamp=datetime.now(timezone.utc),
    )


def _dialogue(*turns: DialogueTurn) -> Dialogue:
    return Dialogue(
        id=Id.new(),
        turns=list(turns),
        created_at=datetime.now(timezone.utc),
    )


def _note_out(turn_index: int) -> NoteOut:
    def claim(text: str) -> ClaimOut:
        return ClaimOut(claim=text, evidence_text=text, turn_index=turn_index)

    def contextual(text: str) -> ContextualClaimOut:
        return ContextualClaimOut(
            claim=text,
            evidence_text=text,
            turn_index=turn_index,
        )

    return NoteOut(
        subjective=claim("sharp headache"),
        objective=claim("bp 130 over 85"),
        assessment=contextual("tension headache"),
        plan=contextual("take ibuprofen"),
    )


def _extract_result(
    agent: StubAgent,
    dialogue: Dialogue,
    context: ClinicalContextInput | None = None,
):
    return asyncio.run(LlmSoapExtractor(agent).extract(dialogue, context))


def _extract(agent: StubAgent, dialogue: Dialogue):
    return _extract_result(agent, dialogue).report


def test_state_notes_become_soap_report():
    turn = _turn("patient", "I have a sharp headache")
    dialogue = _dialogue(turn)
    agent = StubAgent({"notes": [_note_out(1)]})

    report = _extract(agent, dialogue)

    assert len(report.soap_notes) == 1
    note = report.soap_notes[0]
    assert note.subjective.claim == "sharp headache"
    assert note.plan.claim == "take ibuprofen"


def test_turn_index_maps_back_to_turn_id():
    first = _turn("patient", "I have a sharp headache")
    second = _turn("doctor", "Since when?")
    dialogue = _dialogue(first, second)
    # Все claim'ы ссылаются на вторую реплику (индекс 2, 1-based).
    agent = StubAgent({"notes": [_note_out(2)]})

    report = _extract(agent, dialogue)

    evidence = report.soap_notes[0].subjective.evidence
    assert evidence.turn_id == second.id
    assert evidence.text == "sharp headache"


def test_out_of_range_turn_index_is_dangling():
    turn = _turn("patient", "I have a sharp headache")
    dialogue = _dialogue(turn)
    # Индекс 99 не существует -> «висячий» id, не равный реальной реплике.
    agent = StubAgent({"notes": [_note_out(99)]})

    report = _extract(agent, dialogue)

    assert report.soap_notes[0].subjective.evidence.turn_id != turn.id


def test_evidence_prefix_is_stripped():
    turn = _turn("person", "Болит голова")
    dialogue = _dialogue(turn)

    def claim(text: str) -> ClaimOut:
        return ClaimOut(claim="c", evidence_text=text, turn_index=1)

    def contextual(text: str) -> ContextualClaimOut:
        return ContextualClaimOut(claim="c", evidence_text=text, turn_index=1)

    note = NoteOut(
        subjective=claim("[1] person: Болит голова"),
        objective=claim("[2] medic: Давление 135 на 88"),
        assessment=contextual("[3]"),
        plan=contextual("[16], [17]"),
    )
    agent = StubAgent({"notes": [note]})

    report = _extract(agent, dialogue)
    result = report.soap_notes[0]

    assert result.subjective.evidence.text == "Болит голова"
    assert result.objective.evidence.text == "Давление 135 на 88"
    # Голые ссылки без текста схлопываются в пустую цитату.
    assert result.assessment.evidence.text == ""
    assert result.plan.evidence.text == ""


def test_multiple_notes_are_all_mapped():
    turn = _turn("patient", "headache and cough")
    dialogue = _dialogue(turn)
    agent = StubAgent({"notes": [_note_out(1), _note_out(1)]})

    report = _extract(agent, dialogue)

    assert len(report.soap_notes) == 2
    # Каждая нота получает собственный идентификатор.
    assert report.soap_notes[0].id != report.soap_notes[1].id


def test_empty_notes_give_empty_report():
    dialogue = _dialogue(_turn("patient", "hi"))
    agent = StubAgent({"notes": []})

    report = _extract(agent, dialogue)

    assert report.soap_notes == []


def test_agent_receives_flattened_turns():
    turn = _turn("patient", "I have a sharp headache")
    dialogue = _dialogue(turn)
    agent = StubAgent({"notes": []})

    _extract(agent, dialogue)

    assert agent.received_messages["turns"] == [
        {"id": str(turn.id), "role": "patient", "content": "I have a sharp headache"}
    ]
    assert agent.received_messages["clinical_context"] is None
    assert agent.received_messages["segments"] == []
    assert agent.received_messages["notes"] == []


def test_assessment_and_plan_context_refs_are_returned_as_sidecar_requests():
    turn = _turn("patient", "I have a headache")
    dialogue = _dialogue(turn)
    note = _note_out(1)
    note.assessment.context_refs = ["Condition/history"]
    note.plan.context_refs = ["MedicationRequest/active"]
    context = ClinicalContextInput(
        patient_ref="Patient/p1",
        encounter_ref="Encounter/e1",
        resources=(
            ClinicalContextResource(
                reference="Condition/history",
                resource_type="Condition",
                category="condition",
                display="Migraine",
            ),
        ),
    )
    agent = StubAgent({"notes": [note]})

    extraction = _extract_result(agent, dialogue, context)

    assert agent.received_messages["clinical_context"] is context
    assert [item.section for item in extraction.requested_context] == [
        "assessment",
        "plan",
    ]
    assert extraction.requested_context[0].references == ["Condition/history"]
    assert extraction.requested_context[1].references == [
        "MedicationRequest/active"
    ]
    assert extraction.report.soap_notes[0].assessment.evidence.text == (
        "tension headache"
    )


def test_context_render_uses_exact_refs_without_patient_demographics():
    context = ClinicalContextInput(
        patient_ref="Patient/secret",
        encounter_ref="Encounter/e1",
        resources=(
            ClinicalContextResource(
                reference="AllergyIntolerance/a1",
                resource_type="AllergyIntolerance",
                category="allergy",
                display="Penicillin",
                status="active",
            ),
        ),
    )

    rendered = _render_clinical_context(context)

    assert "[AllergyIntolerance/a1]" in rendered
    assert "Penicillin" in rendered
    assert "Patient/secret" not in rendered
