"""The extractor port: the boundary between a dialogue and its SOAP report.

Pure contract — a concrete extractor (LLM-backed, T8) implements it. Keeping
the domain unaware of *how* extraction happens lets the implementation and its
dependencies live at the edges.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dialogue.dialogue import Dialogue

from .soap import SoapReport


class SoapExtractor(ABC):
    """Turns a dialogue plus patient context into a structured SOAP report."""

    @abstractmethod
    async def extract(self, dialogue: Dialogue, patient_context: str) -> SoapReport:
        """Extract SOAP notes from ``dialogue``, grounded in the dialogue turns."""
        ...
