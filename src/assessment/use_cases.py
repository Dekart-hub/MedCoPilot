from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from shared.value_objects import Id
from soap.soap import SoapReport, SoapReportId, SoapNote

from .annotation import SoapAnnotation, AnnotationStatus, ChangeDescription
from .annotation_repository import AnnotationRepository


@dataclass(frozen=True, slots=True)
class AnnotationRequest:
    """DTO для запроса на создание разметки."""
    original_report_id: SoapReportId
    corrected_notes: list[SoapNote]
    annotator_id: str
    changes: list[ChangeDescription] 
    comments: Optional[str] = None


class CreateSoapAnnotation:
    """Use case: сохранение исправлений врача."""

    def __init__(self, repository: AnnotationRepository) -> None:
        self.repository = repository

    async def execute(self, request: AnnotationRequest) -> SoapAnnotation:
        # 1. Собираем исправленный отчёт из новых заметок
        corrected_report = SoapReport(
            id=SoapReportId.new(),
            soap_notes=request.corrected_notes,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # 2. Создаём сущность разметки
        annotation = SoapAnnotation(
            id=Id.new(),
            original_report_id=request.original_report_id,
            annotated_report=corrected_report,
            annotator_id=request.annotator_id,
            status=AnnotationStatus.COMPLETED,
            notes=request.comments,
        )

        # 3. Сохраняем через репозиторий
        await self.repository.save(annotation)

        return annotation