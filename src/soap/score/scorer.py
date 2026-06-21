from abc import ABC, abstractmethod

from dialogue import Dialogue

from ..soap import SoapNote
from .score import SoapNoteConfidenceScore


class ConfidenceScorer(ABC):
    @abstractmethod
    async def score(
        self, dialogue: Dialogue, soap_note: SoapNote
    ) -> SoapNoteConfidenceScore:
        raise NotImplementedError
