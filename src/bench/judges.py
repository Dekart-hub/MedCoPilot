"""Benchmark LLM judge: (transcript, gold, our note) → CaseVerdict.

Three artifacts in one prompt: the gold note is the anchor for content
completeness, the transcript is the source of truth for hallucinations. The
judge does not penalize format/style (our SOAP is flat, the gold is richer —
decision 3.2 in REQUIREMENTS_CLARIFICATIONS.md); multi-note output is graded as
a single document. The judge may be the same model as the generator
(self-preference bias) — this is recorded in the report limitations, see
bench.report.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from shared.prompts import PromptStore
from soap import ReportView

LABELS: tuple[str, ...] = ("excellent", "normal", "bad")

JUDGE_PROMPT_KEY = "bench.judge"

DEFAULT_BENCH_PROMPTS: dict[str, str] = {
    JUDGE_PROMPT_KEY: (
        "You are an experienced clinician grading a machine-generated SOAP note "
        "against the source consultation transcript and a reference note written "
        "by a human scribe.\n\n"
        "TRANSCRIPT (the only source of truth for facts):\n"
        "{{ transcript }}\n\n"
        "REFERENCE NOTE (anchor for content completeness; its format is richer "
        "than plain SOAP):\n"
        "{{ gold_note }}\n\n"
        "CANDIDATE NOTE (machine-generated; may contain several SOAP notes for "
        "one visit — grade them together as one document):\n"
        "{{ candidate_note }}\n\n"
        "Grading rules:\n"
        "1. Use the REFERENCE NOTE only to check that clinically important "
        "content is covered. Do NOT penalize the candidate for different "
        "formatting, section structure, ordering, or brevity of style.\n"
        "2. Any statement in the candidate that is not supported by the "
        "TRANSCRIPT is a hallucination.\n"
        "3. Score 1 (worst) to 5 (best): coverage (important findings, "
        "assessments and plans are present), correctness (facts match the "
        "transcript), hallucination_free (5 = nothing fabricated).\n"
        "4. List every fabricated fact in `hallucinations` (quote the candidate "
        "verbatim); empty list if none.\n"
        "5. Overall label: \"excellent\" = safe to file after a quick skim; "
        "\"normal\" = usable after minor edits; \"bad\" = misleading or missing "
        "critical content.\n"
        "6. Keep `rationale` to one or two sentences."
    ),
}


class CaseVerdict(BaseModel):
    """Structured judge verdict for a single case."""

    label: Literal["excellent", "normal", "bad"] = Field(
        description="Overall 3-class quality label"
    )
    coverage: int = Field(ge=1, le=5, description="Important content is present")
    correctness: int = Field(ge=1, le=5, description="Facts match the transcript")
    hallucination_free: int = Field(
        ge=1, le=5, description="5 = no fabricated facts at all"
    )
    hallucinations: list[str] = Field(
        default_factory=list, description="Verbatim fabricated statements, if any"
    )
    rationale: str = Field(description="One or two sentences explaining the label")


class JudgeError(RuntimeError):
    """The judge failed to produce a valid verdict (after retry)."""


def render_candidate_note(view: ReportView) -> str:
    """Linearizes a ReportView into flat text for the judge prompt."""
    lines: list[str] = []
    for i, note in enumerate(view.notes, start=1):
        lines.append(f"Note {i}:")
        lines.append(f"  Subjective: {note.subjective.claim}")
        lines.append(f"  Objective: {note.objective.claim}")
        lines.append(f"  Assessment: {note.assessment.claim}")
        lines.append(f"  Plan: {note.plan.claim}")
    return "\n".join(lines)


class LlmJudge:
    """One LLM call per case, structured output, one retry."""

    def __init__(self, model: BaseChatModel, prompts: PromptStore) -> None:
        self._structured = model.with_structured_output(CaseVerdict)
        self._prompts = prompts

    async def judge(
        self, transcript: str, gold_note: str, view: ReportView
    ) -> CaseVerdict:
        prompt = await self._prompts.get(
            JUDGE_PROMPT_KEY,
            transcript=transcript,
            gold_note=gold_note,
            candidate_note=render_candidate_note(view),
        )
        last_error: Exception | None = None
        for _ in range(2):
            try:
                verdict = await self._structured.ainvoke(prompt)
            except Exception as exc:  # network, parsing, validation — retry all
                last_error = exc
                continue
            if isinstance(verdict, CaseVerdict):
                return verdict
            last_error = TypeError(f"Unexpected judge output: {type(verdict)!r}")
        raise JudgeError(f"Judge failed after retry: {last_error!r}")
