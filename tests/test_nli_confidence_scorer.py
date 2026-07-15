"""Unit tests for the NLI-backed confidence scorer, with a mock NLI (no network)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from dialogue.dialogue import Dialogue
from shared.value_objects import Id
from soap.llm_client import LlmClient
from soap.llm_extractor import LlmSoapExtractor
from soap.nli_scorer import NliConfidenceScorer, render_dialogue, render_note
from soap.soap import AssessmentClaim, SoapClaim, SoapNote, TurnCitation


class _FakeNli:
    """Records every (inference, reference) pair and returns a fixed score."""

    def __init__(self, score: float = 0.73) -> None:
        self._score = score
        self.calls: list[tuple[str, str]] = []

    async def calc_nli_score(self, inference: str, reference: str) -> float:
        self.calls.append((inference, reference))
        return self._score


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _dialogue() -> Dialogue:
    dialogue = Dialogue.start()
    dialogue.add_turn("patient", "I've had a headache for three days.")
    dialogue.add_turn("doctor", "Blood pressure is 140 over 90.")
    return dialogue


def _note() -> SoapNote:
    turn = Id.new()
    citation = [TurnCitation(turn_id=turn)]
    return SoapNote(
        id=Id.new(),
        subjective=[SoapClaim(id=Id.new(), text="Headache for three days.", citations=citation)],
        objective=[SoapClaim(id=Id.new(), text="Blood pressure 140/90.", citations=citation)],
        assessment=[AssessmentClaim(id=Id.new(), text="Tension headache.", citations=citation)],
        plan=[SoapClaim(id=Id.new(), text="Ibuprofen as needed.", citations=citation)],
    )


def test_score_is_a_probability_in_unit_range() -> None:
    scorer = NliConfidenceScorer(_FakeNli(score=0.42))  # type: ignore[arg-type]
    score = _run(scorer.score(_dialogue(), _note()))
    assert 0.0 <= score <= 1.0
    assert score == 0.42


def test_inference_is_the_whole_note_not_a_single_claim() -> None:
    nli = _FakeNli()
    _run(NliConfidenceScorer(nli).score(_dialogue(), _note()))  # type: ignore[arg-type]

    assert len(nli.calls) == 1  # one call for the whole note, not one per claim
    inference, _ = nli.calls[0]
    # Claims from every section landed together in a single inference string.
    assert "Headache for three days." in inference
    assert "Blood pressure 140/90." in inference
    assert "Tension headache." in inference
    assert "Ibuprofen as needed." in inference


def test_reference_is_the_rendered_dialogue_without_turn_markers() -> None:
    nli = _FakeNli()
    dialogue = _dialogue()
    _run(NliConfidenceScorer(nli).score(dialogue, _note()))  # type: ignore[arg-type]

    _, reference = nli.calls[0]
    assert "patient: I've had a headache for three days." in reference
    # No citation/turn-index markers leak into the reference.
    assert "[1]" not in reference and "[2]" not in reference


def test_note_renders_section_labelled_claim_block() -> None:
    text = render_note(_note())
    assert text.splitlines() == [
        "Subjective: Headache for three days.",
        "Objective: Blood pressure 140/90.",
        "Assessment: Tension headache.",
        "Plan: Ibuprofen as needed.",
    ]


def test_note_rendering_skips_empty_sections() -> None:
    turn = [TurnCitation(turn_id=Id.new())]
    note = SoapNote(
        id=Id.new(),
        subjective=[SoapClaim(id=Id.new(), text="Only subjective here.", citations=turn)],
    )
    assert render_note(note) == "Subjective: Only subjective here."


def test_dialogue_rendering_is_role_text_lines() -> None:
    assert render_dialogue(_dialogue()).splitlines() == [
        "patient: I've had a headache for three days.",
        "doctor: Blood pressure is 140 over 90.",
    ]


# --- through the extractor's per-note fan-out ------------------------------- #

_NOTE_PAYLOAD: dict[str, Any] = {
    "subjective": [{"text": "Headache for three days.", "turn_index": 1, "quote": None}],
    "objective": [{"text": "Blood pressure 140/90.", "turn_index": 2, "quote": None}],
    "assessment": [{"text": "Tension headache.", "turn_index": 1, "quote": None}],
    "plan": [{"text": "Ibuprofen as needed.", "turn_index": 2, "quote": None}],
}


class _FakeLlmClient(LlmClient):
    """Mock LLM: names ``problems`` in the plan call, canned note otherwise."""

    def __init__(self, problems: list[str]) -> None:
        self._problems = problems

    async def complete_json(
        self, *, instructions: str, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        if "problems" in schema.get("properties", {}):
            return {"problems": [{"title": title} for title in self._problems]}
        return _NOTE_PAYLOAD


def test_every_note_gets_a_confidence_in_unit_range_via_extractor() -> None:
    nli = _FakeNli(score=0.66)
    extractor = LlmSoapExtractor(
        _FakeLlmClient(problems=["Headache", "Hypertension"]),
        NliConfidenceScorer(nli),  # type: ignore[arg-type]
    )

    report = _run(extractor.extract(_dialogue(), "45yo"))

    assert len(report.notes) == 2
    for note in report.notes:
        assert note.confidence is not None
        assert 0.0 <= note.confidence <= 1.0
        assert note.confidence == 0.66
    assert len(nli.calls) == 2  # scored once per note, whole-note inference
