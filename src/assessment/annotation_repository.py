from abc import ABC, abstractmethod

from soap.soap import SoapNoteId, SoapReportId
from .annotation import (
    AddedSoapNote,
    AddedSoapNoteId,
    CorrectedSoapNote,
    CorrectedSoapNoteId,
)


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
    async def list_by_report(self, report_id: SoapReportId) -> list[CorrectedSoapNote]:
        raise NotImplementedError

    @abstractmethod
    async def list_all(self) -> list[CorrectedSoapNote]:
        raise NotImplementedError


class AddedSoapNoteRepository(ABC):
    """Интерфейс для сохранения SOAP-нот, добавленных врачом с нуля."""

    @abstractmethod
    async def save(self, added_note: AddedSoapNote) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_id(self, added_note_id: AddedSoapNoteId) -> AddedSoapNote:
        raise NotImplementedError

    @abstractmethod
    async def find_by_id(
        self, added_note_id: AddedSoapNoteId
    ) -> AddedSoapNote | None:
        raise NotImplementedError

    @abstractmethod
    async def find_by_added_note(
        self, note_id: SoapNoteId
    ) -> AddedSoapNote | None:
        raise NotImplementedError

    @abstractmethod
    async def list_by_report(self, report_id: SoapReportId) -> list[AddedSoapNote]:
        raise NotImplementedError

    @abstractmethod
    async def list_all(self) -> list[AddedSoapNote]:
        raise NotImplementedError
