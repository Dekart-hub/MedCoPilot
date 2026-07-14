from abc import ABC, abstractmethod

from dialogue import Dialogue

from ..context import ClinicalContextInput, SoapExtraction


class SoapExtractor(ABC):
    @abstractmethod
    async def extract(
        self,
        dialogue: Dialogue,
        clinical_context: ClinicalContextInput | None = None,
    ) -> SoapExtraction:
        raise NotImplementedError
