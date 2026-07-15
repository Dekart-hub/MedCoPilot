from __future__ import annotations

from abc import ABC, abstractmethod


class EhrClient(ABC):
    """Read-only source of free-text patient context for the SOAP extractor.

    Given a ``patient_id`` it returns prior clinical context (history,
    medications, allergies) as plain text ready to splice into the extractor
    prompt. An unknown id resolves to an empty/default string rather than
    raising, so a missing record degrades to "no prior context".
    """

    @abstractmethod
    async def get_patient_context(self, patient_id: str) -> str:
        raise NotImplementedError
