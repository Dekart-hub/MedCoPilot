from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from bench.judges import (
    DEFAULT_BENCH_PROMPTS,
    CaseVerdict,
    JudgeError,
    LlmJudge,
    render_candidate_note,
)
from shared.prompts import InMemoryPromptStore
from shared.value_objects import Id
from soap import AssessmentView, ClaimView, NoteView, ReportView


class StubStructured:
    """Stands in for the runnable from with_structured_output: yields outcomes in order."""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str):
        self.prompts.append(prompt)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class StubModel:
    def __init__(self, outcomes: list) -> None:
        self.structured = StubStructured(outcomes)
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured


def _claim(text: str) -> ClaimView:
    return ClaimView(id=Id.new(), claim=text, evidence_text=text, turn_id=Id.new())


def _view(*note_prefixes: str) -> ReportView:
    now = datetime.now(timezone.utc)
    notes = [
        NoteView(
            id=Id.new(),
            subjective=_claim(f"{p} subj"),
            objective=_claim(f"{p} obj"),
            assessment=AssessmentView(
                id=Id.new(), claim=f"{p} asmt", evidence_text=f"{p} asmt",
                turn_id=Id.new(),
            ),
            plan=_claim(f"{p} plan"),
            confidence=0.9,
        )
        for p in note_prefixes
    ]
    return ReportView(id=Id.new(), notes=notes, created_at=now, updated_at=now)


def _verdict() -> CaseVerdict:
    return CaseVerdict(
        label="normal", coverage=4, correctness=4, hallucination_free=5,
        hallucinations=[], rationale="fine",
    )


def _judge(outcomes: list) -> tuple[LlmJudge, StubModel]:
    model = StubModel(outcomes)
    return LlmJudge(model, InMemoryPromptStore(DEFAULT_BENCH_PROMPTS)), model


def test_render_candidate_note_lists_all_notes_and_sections():
    text = render_candidate_note(_view("one", "two"))
    assert "Note 1" in text and "Note 2" in text
    assert "one subj" in text and "two plan" in text
    assert "Subjective" in text and "Plan" in text


def test_judge_prompt_contains_all_three_artifacts():
    judge, model = _judge([_verdict()])
    asyncio.run(judge.judge("THE-TRANSCRIPT", "THE-GOLD-NOTE", _view("n")))
    prompt = model.structured.prompts[0]
    assert "THE-TRANSCRIPT" in prompt
    assert "THE-GOLD-NOTE" in prompt
    assert "n subj" in prompt
    assert model.schema is CaseVerdict


def test_judge_returns_verdict():
    judge, _ = _judge([_verdict()])
    verdict = asyncio.run(judge.judge("t", "g", _view("n")))
    assert verdict.label == "normal"


def test_judge_retries_once_then_succeeds():
    judge, model = _judge([RuntimeError("boom"), _verdict()])
    verdict = asyncio.run(judge.judge("t", "g", _view("n")))
    assert verdict.label == "normal"
    assert len(model.structured.prompts) == 2


def test_judge_raises_after_two_failures():
    judge, _ = _judge([RuntimeError("one"), RuntimeError("two")])
    with pytest.raises(JudgeError):
        asyncio.run(judge.judge("t", "g", _view("n")))


def test_verdict_rejects_out_of_range_scores():
    with pytest.raises(Exception):
        CaseVerdict(
            label="bad", coverage=6, correctness=1, hallucination_free=1,
            hallucinations=[], rationale="x",
        )
