from abc import ABC, abstractmethod
from typing import Optional

from soap.soap import SoapReportId
from .annotation import SoapAnnotation, AnnotationId



class EntityNotFoundError(Exception):
    pass

class AnnotationRepository(ABC):
    """Интерфейс для сохранения и получения разметки."""
    @abstractmethod
    async def save(self, annotation: SoapAnnotation) -> None:
        raise NotImplementedError

    # get - бросает исключение, если не найдено
    @abstractmethod
    async def get_by_id(self, annotation_id: AnnotationId) -> SoapAnnotation:
        raise NotImplementedError

    # find - возвращает Optional (в Python от этого пока не уйти, но это только для find)
    @abstractmethod
    async def find_by_id(self, annotation_id: AnnotationId) -> Optional[SoapAnnotation]:
        raise NotImplementedError

    @abstractmethod
    async def find_by_original_report(self, report_id: SoapReportId) -> Optional[SoapAnnotation]:
        raise NotImplementedError

    @abstractmethod
    async def get_completed_annotations(self) -> list[SoapAnnotation]:
        raise NotImplementedError