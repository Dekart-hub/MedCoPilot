"""NLI-backed confidence scorer: how well a whole note is grounded in the dialogue.

The real :class:`~soap.scorer.ConfidenceScorer` behind the extractor's per-note
fan-out [#7/FR-4][#11/FR-1][#11/FR-2]. Scoring is **whole-note, dialogue-grounded**:
the entire dialogue is the reference and the entire note (all its S/O/A/P claims
rendered together) is the inference, and the pair is fed once to a
:class:`~nli.VllmNliScorer`. It is deliberately *not* per-claim and does not use
turn citations -- a note's confidence answers "does the dialogue entail this
note?", not "is each sentence traceable?" (that is what citations are for).
"""

from __future__ import annotations

from dialogue.dialogue import Dialogue
from nli import VllmNliScorer

from .scorer import ConfidenceScorer
from .soap import SoapNote


def render_dialogue(dialogue: Dialogue) -> str:
    """Render a dialogue as plain ``speaker: text`` lines.

    Mirrors the extractor's transcript rendering but drops the ``[index]``
    markers: whole-note NLI grounds against the conversation as a whole and never
    references turns by number.
    """
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in dialogue.turns)


def render_note(note: SoapNote) -> str:
    """Flatten a note into a ``Section: claims`` block, one line per non-empty section."""
    lines = [
        f"{section.value.capitalize()}: {' '.join(claim.text for claim in claims)}"
        for section, claims in note.sections()
        if claims
    ]
    return "\n".join(lines)


class NliConfidenceScorer(ConfidenceScorer):
    """Scores a note's confidence as ``P(dialogue entails whole note)`` via NLI."""

    def __init__(self, nli: VllmNliScorer) -> None:
        self._nli = nli

    async def score(self, dialogue: Dialogue, note: SoapNote) -> float:
        return await self._nli.calc_nli_score(
            inference=render_note(note),
            reference=render_dialogue(dialogue),
        )
