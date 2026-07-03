from abc import ABC, abstractmethod

from soap.soap import SoapReportId
from .annotation import SoapAnnotation, AnnotationId


class EntityNotFoundError(Exception):
    pass


class AnnotationRepository(ABC):
    """Интерфейс для сохранения и получения разметки."""

    @abstractmethod
    async def save(self, annotation: SoapAnnotation) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_id(self, annotation_id: AnnotationId) -> SoapAnnotation:
        raise NotImplementedError

    @abstractmethod
    async def find_by_id(self, annotation_id: AnnotationId) -> SoapAnnotation | None:
        raise NotImplementedError

    @abstractmethod
    async def find_by_original_report(
        self, report_id: SoapReportId
    ) -> SoapAnnotation | None:
        raise NotImplementedError

    @abstractmethod
    async def find_by_corrected_report(
        self, report_id: SoapReportId
    ) -> SoapAnnotation | None:
        raise NotImplementedError

    @abstractmethod
    async def list_all(self) -> list[SoapAnnotation]:
        raise NotImplementedError
