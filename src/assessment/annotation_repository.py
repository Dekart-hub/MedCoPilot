from abc import ABC, abstractmethod
from typing import Optional

from soap.soap import SoapReportId
from .annotation import SoapAnnotation, AnnotationId


class AnnotationRepository(ABC):
    """Интерфейс для сохранения и получения разметки."""

    @abstractmethod
    async def save(self, annotation: SoapAnnotation) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_id(self, annotation_id: AnnotationId) -> Optional[SoapAnnotation]:
        raise NotImplementedError

    @abstractmethod
    async def get_by_original_report(self, report_id: SoapReportId) -> Optional[SoapAnnotation]:
        raise NotImplementedError

    @abstractmethod
    async def get_completed_annotations(self) -> list[SoapAnnotation]:
        """Получить все завершённые разметки (для обучения Tier 3)."""
        raise NotImplementedError