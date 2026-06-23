# src/soap/annotation_repository.py

from abc import ABC, abstractmethod
from typing import Optional

class AnnotationRepository(ABC):
    @abstractmethod
    async def save(self, annotation: SoapAnnotation) -> None:
        pass
    
    @abstractmethod
    async def get_by_original_report(self, report_id: SoapReportId) -> Optional[SoapAnnotation]:
        pass
    
    @abstractmethod
    async def get_by_annotator(self, annotator_id: str) -> list[SoapAnnotation]:
        pass
    
    @abstractmethod
    async def get_labeled_dataset(self) -> list[SoapAnnotation]:
        """Получить все завершённые разметки для обучения."""
        pass