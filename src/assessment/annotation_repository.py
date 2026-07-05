from abc import ABC, abstractmethod

from soap.soap import SoapNoteId
from .annotation import CorrectedSoapNote, CorrectedSoapNoteId


class EntityNotFoundError(Exception):
    pass


class CorrectedSoapNoteRepository(ABC):
    """Интерфейс для сохранения и получения исправленных SOAP-нот."""

    @abstractmethod
    async def save(self, corrected_note: CorrectedSoapNote) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_id(
        self, corrected_note_id: CorrectedSoapNoteId
    ) -> CorrectedSoapNote:
        raise NotImplementedError

    @abstractmethod
    async def find_by_id(
        self, corrected_note_id: CorrectedSoapNoteId
    ) -> CorrectedSoapNote | None:
        raise NotImplementedError

    @abstractmethod
    async def find_by_original_note(
        self, note_id: SoapNoteId
    ) -> CorrectedSoapNote | None:
        raise NotImplementedError

    @abstractmethod
    async def find_by_corrected_note(
        self, note_id: SoapNoteId
    ) -> CorrectedSoapNote | None:
        raise NotImplementedError

    @abstractmethod
    async def list_all(self) -> list[CorrectedSoapNote]:
        raise NotImplementedError
