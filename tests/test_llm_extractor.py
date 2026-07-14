"""Unit tests for the LLM SOAP extractor, with a mock LLM (no network)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable, Iterator
from time import perf_counter
from typing import Any

import pytest

from dialogue.dialogue import Dialogue
from soap.llm_client import LlmClient
from soap.llm_extractor import LlmSoapExtractor, SoapExtractionError
from soap.scorer import NullConfidenceScorer
from soap.soap import SoapClaim, SoapNote, SoapReport

# A note draft whose claims cite real turns of the two-turn dialogue below.
_NOTE_PAYLOAD: dict[str, Any] = {
    "subjective": [{"text": "Headache for three days.", "turn_index": 1, "quote": "headache"}],
    "objective": [{"text": "Blood pressure 140/90.", "turn_index": 2, "quote": None}],
    "assessment": [{"text": "Tension headache.", "turn_index": 1, "quote": None}],
    "plan": [{"text": "Ibuprofen as needed.", "turn_index": 2, "quote": None}],
}


class FakeLlmClient(LlmClient):
    """Mock LLM: canned plan/note payloads, optional delay and failures."""

    def __init__(
        self,
        *,
        problems: list[str],
        note_payload: dict[str, Any] = _NOTE_PAYLOAD,
        delay: float = 0.0,
        fail_note_titles: Iterable[str] = (),
        fail_plan: bool = False,
    ) -> None:
        self._problems = problems
        self._note_payload = note_payload
        self._delay = delay
        self._fail_note_titles = tuple(fail_note_titles)
        self._fail_plan = fail_plan
        self.calls: list[str] = []

    async def complete_json(
        self, *, instructions: str, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        await asyncio.sleep(self._delay)
        if "problems" in schema.get("properties", {}):
            self.calls.append("plan")
            if self._fail_plan:
                raise TimeoutError("planner unavailable")
            return {"problems": [{"title": title} for title in self._problems]}
        self.calls.append("note")
        if any(bad in prompt for bad in self._fail_note_titles):
            raise TimeoutError("extractor unavailable")
        return self._note_payload


def _dialogue() -> Dialogue:
    dialogue = Dialogue.start()
    dialogue.add_turn("patient", "I've had a headache for three days.")
    dialogue.add_turn("doctor", "Blood pressure is 140 over 90.")
    return dialogue


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _extract(client: FakeLlmClient, dialogue: Dialogue) -> SoapReport:
    extractor = LlmSoapExtractor(client, NullConfidenceScorer())
    return _run(extractor.extract(dialogue, "45yo, no known allergies"))


def _all_claims(report: SoapReport) -> Iterator[SoapClaim]:
    for note in report.notes:
        for _, claims in note.sections():
            yield from claims


def test_extract_produces_a_valid_report() -> None:
    report = _extract(FakeLlmClient(problems=["Headache"]), _dialogue())

    assert isinstance(report, SoapReport)
    assert len(report.notes) == 1
    note = report.notes[0]
    assert isinstance(note, SoapNote)
    assert note.confidence is None  # NullConfidenceScorer leaves it unscored.
    assert [claim.text for claim in note.subjective] == ["Headache for three days."]
    assert [claim.text for claim in note.assessment] == ["Tension headache."]


def test_non_empty_dialogue_yields_at_least_one_note() -> None:
    # The model names no problem; the extractor must still emit one note.
    report = _extract(FakeLlmClient(problems=[]), _dialogue())

    assert len(report.notes) >= 1


def test_every_claim_cites_a_real_turn() -> None:
    dialogue = _dialogue()
    turn_ids = {turn.id for turn in dialogue.turns}

    report = _extract(FakeLlmClient(problems=["Headache"]), dialogue)

    claims = list(_all_claims(report))
    assert claims  # sanity: the note is not empty
    for claim in claims:
        assert claim.citations
        assert all(citation.turn_id in turn_ids for citation in claim.citations)


def test_ungrounded_claims_are_dropped() -> None:
    payload = {
        "subjective": [{"text": "grounded", "turn_index": 1, "quote": None}],
        "objective": [{"text": "dangling", "turn_index": 99, "quote": None}],
        "assessment": [],
        "plan": [],
    }
    report = _extract(FakeLlmClient(problems=["Headache"], note_payload=payload), _dialogue())

    note = report.notes[0]
    assert [claim.text for claim in note.subjective] == ["grounded"]
    assert note.objective == []  # turn_index 99 does not exist -> dropped


def test_note_tasks_fan_out_concurrently() -> None:
    delay = 0.1
    problems = ["A", "B", "C", "D"]
    client = FakeLlmClient(problems=problems, delay=delay)

    start = perf_counter()
    report = _extract(client, _dialogue())
    elapsed = perf_counter() - start

    assert len(report.notes) == len(problems)
    assert client.calls.count("note") == len(problems)
    # Concurrent fan-out costs ~plan + one note (~2*delay); running the four
    # note calls serially would cost ~5*delay. The threshold sits between.
    assert elapsed < delay * 3


def test_one_note_failure_does_not_abort_the_report() -> None:
    client = FakeLlmClient(problems=["Headache", "Rash"], fail_note_titles=["Rash"])

    report = _extract(client, _dialogue())

    # The failing note is logged and skipped; the healthy one survives.
    assert len(report.notes) == 1


def test_planning_failure_surfaces_a_clear_error() -> None:
    client = FakeLlmClient(problems=["Headache"], fail_plan=True)
    extractor = LlmSoapExtractor(client, NullConfidenceScorer())

    with pytest.raises(SoapExtractionError):
        _run(extractor.extract(_dialogue(), ""))


def test_empty_dialogue_yields_an_empty_report() -> None:
    client = FakeLlmClient(problems=["Headache"])
    extractor = LlmSoapExtractor(client, NullConfidenceScorer())

    report = _run(extractor.extract(Dialogue.start(), ""))

    assert report.notes == []
    assert client.calls == []  # no LLM call for an empty dialogue
