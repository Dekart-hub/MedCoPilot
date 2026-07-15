"""Port for reading a patient's clinical context from an EHR by id.

An EHR client resolves a ``patient_id`` to a free-text summary — history,
medications, allergies — that grounds the SOAP extractor's prompt. Keeping the
port pure lets the extractor depend on the capability, not on a concrete EHR
integration: T9 ships a read-only mock, a real FHIR adapter can follow without
reshaping the domain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EhrClient(ABC):
    """Resolves a patient id to free-text clinical context."""

    @abstractmethod
    async def get_patient_context(self, patient_id: str) -> str:
        """Return free-text context for ``patient_id``.

        An unknown id yields an empty string: the caller feeds the result
        straight to the extractor, which renders empty context as "(none)"
        rather than failing on a missing record.
        """
        ...
