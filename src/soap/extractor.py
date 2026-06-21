from abc import ABC, abstractmethod

from dialogue import Dialogue

from .soap import SoapReport


class SoapExtractor(ABC):
    @abstractmethod
    async def extract(self, dialogue: Dialogue) -> SoapReport:
        raise NotImplementedError
