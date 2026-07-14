"""LLM-backed SOAP extractor with an async per-note fan-out.

Turning a dialogue into a :class:`~soap.soap.SoapReport` happens in two phases:

1. **Plan** — one LLM call names the distinct clinical problems in the dialogue.
2. **Fan-out** — each problem becomes an independent async task that extracts its
   own S/O/A/P claims and scores the note's confidence. The tasks run
   concurrently, so a report of *N* notes costs roughly one note's latency, not
   *N* times it (NFR-1).

The domain stays pure: this module depends only on the :class:`LlmClient` and
:class:`~soap.scorer.ConfidenceScorer` ports, never on a concrete transport.
Claims are grounded by construction — a claim whose source turn cannot be
resolved is dropped rather than emitted ungrounded.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

import structlog
from pydantic import BaseModel, Field

from dialogue.dialogue import Dialogue, DialogueTurn
from shared.value_objects import Id

from .extractor import SoapExtractor
from .llm_client import LlmClient
from .scorer import ConfidenceScorer
from .soap import (
    AssessmentClaim,
    SoapClaim,
    SoapNote,
    SoapReport,
    TurnCitation,
)

_LOG = structlog.get_logger(__name__)

_FALLBACK_PROBLEM = "Clinical encounter"


class SoapExtractionError(RuntimeError):
    """Raised when extraction fails in a way that yields no usable report."""


# --------------------------------------------------------------------------- #
# Structured-output DTOs: the shape the model returns, decoupled from the
# domain. The model references turns by their 1-based transcript index; the
# extractor resolves those indices back to real turn identities.
# --------------------------------------------------------------------------- #


class _ClaimOut(BaseModel):
    text: str = Field(description="A single clinical statement for this section.")
    turn_index: int = Field(
        description="1-based index ([N]) of the dialogue turn this statement rests on."
    )
    quote: str | None = Field(default=None, description="Optional verbatim span from that turn.")


class _NoteOut(BaseModel):
    subjective: list[_ClaimOut] = Field(default_factory=list)
    objective: list[_ClaimOut] = Field(default_factory=list)
    assessment: list[_ClaimOut] = Field(default_factory=list)
    plan: list[_ClaimOut] = Field(default_factory=list)


class _ProblemOut(BaseModel):
    title: str = Field(description="Short name of a distinct clinical problem.")


class _ProblemsOut(BaseModel):
    problems: list[_ProblemOut] = Field(default_factory=list)


class LlmSoapExtractor(SoapExtractor):
    """Extracts a SOAP report from a dialogue via schema-guided LLM calls."""

    def __init__(
        self,
        client: LlmClient,
        scorer: ConfidenceScorer,
        *,
        request_timeout: float = 60.0,
    ) -> None:
        self._client = client
        self._scorer = scorer
        self._request_timeout = request_timeout

    async def extract(self, dialogue: Dialogue, patient_context: str) -> SoapReport:
        if not dialogue.turns:
            return SoapReport(id=Id.new())
        problems = await self._identify_problems(dialogue, patient_context)
        notes = await self._extract_notes(dialogue, patient_context, problems)
        return SoapReport(id=Id.new(), notes=notes)

    async def _identify_problems(
        self, dialogue: Dialogue, patient_context: str
    ) -> list[_ProblemOut]:
        try:
            result = await self._complete(
                _ProblemsOut,
                instructions=_PLAN_INSTRUCTIONS,
                prompt=_plan_prompt(dialogue, patient_context),
            )
        except Exception as exc:
            _LOG.error("soap.problem_identification_failed", error=str(exc))
            raise SoapExtractionError("failed to identify SOAP problems") from exc
        # A non-empty dialogue must yield at least one note (#7): if the model
        # names no problem, fall back to a single note over the whole encounter.
        return result.problems or [_ProblemOut(title=_FALLBACK_PROBLEM)]

    async def _extract_notes(
        self,
        dialogue: Dialogue,
        patient_context: str,
        problems: list[_ProblemOut],
    ) -> list[SoapNote]:
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(self._build_note(dialogue, patient_context, problem))
                for problem in problems
            ]
        notes = [note for task in tasks if (note := task.result()) is not None]
        if not notes:
            raise SoapExtractionError("every SOAP note extraction failed")
        return notes

    async def _build_note(
        self,
        dialogue: Dialogue,
        patient_context: str,
        problem: _ProblemOut,
    ) -> SoapNote | None:
        # One note's failure is logged and isolated: it must not cancel the
        # sibling tasks (hence a caught exception rather than a propagated one).
        try:
            draft = await self._complete(
                _NoteOut,
                instructions=_EXTRACT_INSTRUCTIONS,
                prompt=_extract_prompt(dialogue, patient_context, problem),
            )
            note = _to_note(draft, dialogue.turns)
            note.confidence = await self._scorer.score(dialogue, note)
            return note
        except Exception as exc:
            _LOG.warning("soap.note_extraction_failed", problem=problem.title, error=str(exc))
            return None

    async def _complete[ModelT: BaseModel](
        self, schema: type[ModelT], *, instructions: str, prompt: str
    ) -> ModelT:
        async with asyncio.timeout(self._request_timeout):
            data = await self._client.complete_json(
                instructions=instructions,
                prompt=prompt,
                schema=schema.model_json_schema(),
            )
        return schema.model_validate(data)


# --------------------------------------------------------------------------- #
# DTO -> domain assembly.
# --------------------------------------------------------------------------- #


def _to_note(draft: _NoteOut, turns: list[DialogueTurn]) -> SoapNote:
    return SoapNote(
        id=Id.new(),
        subjective=_build_section(draft.subjective, turns, _to_soap_claim),
        objective=_build_section(draft.objective, turns, _to_soap_claim),
        assessment=_build_section(draft.assessment, turns, _to_assessment_claim),
        plan=_build_section(draft.plan, turns, _to_soap_claim),
    )


def _build_section[ClaimT: SoapClaim](
    claims: Sequence[_ClaimOut],
    turns: list[DialogueTurn],
    to_claim: Callable[[_ClaimOut, list[DialogueTurn]], ClaimT | None],
) -> list[ClaimT]:
    built = (to_claim(claim, turns) for claim in claims)
    return [claim for claim in built if claim is not None]


def _to_soap_claim(claim: _ClaimOut, turns: list[DialogueTurn]) -> SoapClaim | None:
    citation = _citation(claim, turns)
    if citation is None:
        return None
    return SoapClaim(id=Id.new(), text=claim.text, citations=[citation])


def _to_assessment_claim(claim: _ClaimOut, turns: list[DialogueTurn]) -> AssessmentClaim | None:
    citation = _citation(claim, turns)
    if citation is None:
        return None
    # ICD stays unset here; T10 populates it during the coding step.
    return AssessmentClaim(id=Id.new(), text=claim.text, citations=[citation])


def _citation(claim: _ClaimOut, turns: list[DialogueTurn]) -> TurnCitation | None:
    # Resolve the 1-based transcript index back to a real turn. An index the
    # dialogue cannot back drops the claim rather than fabricating grounding.
    if not 1 <= claim.turn_index <= len(turns):
        return None
    quote = claim.quote.strip() if claim.quote else None
    return TurnCitation(turn_id=turns[claim.turn_index - 1].id, quote=quote or None)


# --------------------------------------------------------------------------- #
# Prompts. The transcript is rendered with 1-based ``[index]`` markers so the
# model can cite turns by number and the extractor can resolve them.
# --------------------------------------------------------------------------- #


_PLAN_INSTRUCTIONS = (
    "You are a clinical scribe. Read a doctor-patient dialogue and list the "
    "distinct clinical problems or diagnoses it addresses. Most encounters "
    "cover a single problem; return several only when the dialogue clearly "
    "discusses independent complaints. Never split one problem into parts."
)

_PLAN_TEMPLATE = (
    "Patient context:\n{context}\n\n"
    "Dialogue (each line is [index] speaker: text):\n{transcript}\n\n"
    "List the distinct clinical problems."
)

_EXTRACT_INSTRUCTIONS = (
    "You are a clinical scribe writing a SOAP note for one clinical problem. "
    "Fill the Subjective, Objective, Assessment and Plan sections using only "
    "what the dialogue supports. For every statement, cite its source turn by "
    "the [index] it appears on, and optionally quote it verbatim. Leave a "
    "section empty when the dialogue offers nothing for it."
)

_EXTRACT_TEMPLATE = (
    "Problem: {problem}\n\n"
    "Patient context:\n{context}\n\n"
    "Dialogue (each line is [index] speaker: text):\n{transcript}\n\n"
    "Write the SOAP note for this problem."
)


def _plan_prompt(dialogue: Dialogue, patient_context: str) -> str:
    return _PLAN_TEMPLATE.format(
        context=patient_context or "(none)", transcript=_render_turns(dialogue)
    )


def _extract_prompt(dialogue: Dialogue, patient_context: str, problem: _ProblemOut) -> str:
    return _EXTRACT_TEMPLATE.format(
        problem=problem.title,
        context=patient_context or "(none)",
        transcript=_render_turns(dialogue),
    )


def _render_turns(dialogue: Dialogue) -> str:
    return "\n".join(
        f"[{index}] {turn.speaker}: {turn.text}"
        for index, turn in enumerate(dialogue.turns, start=1)
    )
