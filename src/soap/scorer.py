"""Port for scoring a SOAP note's confidence, plus a no-op default.

A note's confidence reflects how well its claims are grounded in the source
dialogue. The real scorer is an NLI groundedness model (T12); until then the
:class:`NullConfidenceScorer` keeps the extractor's per-note fan-out shape
honest without inventing numbers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dialogue.dialogue import Dialogue

from .soap import SoapNote


class ConfidenceScorer(ABC):
    """Scores how well a note is supported by its source dialogue."""

    @abstractmethod
    async def score(self, dialogue: Dialogue, note: SoapNote) -> float | None:
        """Return a confidence in ``[0, 1]``, or ``None`` when left unscored."""
        ...


class NullConfidenceScorer(ConfidenceScorer):
    """No-op scorer: always ``None``. Placeholder until the T12 NLI scorer."""

    async def score(self, dialogue: Dialogue, note: SoapNote) -> float | None:
        return None
