"""Port for coding a diagnosis to an ICD-10 code, plus a no-op default.

A coder maps free-text diagnosis wording to a single best classifier entry
(top-1). Keeping the port pure lets the extractor depend on the capability, not
on the retrieval backend (T10 ships a local BM25 adapter). The
:class:`NullIcdCoder` keeps the extractor's coding slot honest — mirroring
:class:`~soap.scorer.NullConfidenceScorer` — when no dictionary is wired in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from soap.soap import IcdCoding


class IcdCoder(ABC):
    """Assigns an ICD-10 coding to a diagnosis text."""

    @abstractmethod
    def code(self, diagnosis_text: str) -> IcdCoding | None:
        """Return the best ICD-10 coding for ``diagnosis_text``, or ``None``.

        ``None`` means no confident match (empty or out-of-vocabulary text);
        the caller leaves the claim uncoded rather than inventing a code.
        """
        ...


class NullIcdCoder(IcdCoder):
    """No-op coder: always ``None``. Placeholder until a dictionary is wired in."""

    def code(self, diagnosis_text: str) -> IcdCoding | None:
        return None
