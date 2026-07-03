from dataclasses import dataclass
from datetime import datetime

from shared.value_objects import Id
from soap.soap import SoapReportId

from .annotation import SoapAnnotation
from .annotation_repository import AnnotationRepository
from shared.value_objects import AnnotatorId


@dataclass(frozen=True, slots=True)
class SubmitSoapAnnotationRequest:
    """Запрос на сохранение законченной врачебной разметки."""

    original_report_id: SoapReportId
    corrected_report_id: SoapReportId
    annotator_id: AnnotatorId
    comment: str = ""


class SubmitSoapAnnotation:
    def __init__(self, repository: AnnotationRepository) -> None:
        self.repository = repository

    async def execute(self, request: SubmitSoapAnnotationRequest) -> SoapAnnotation:
        annotation = SoapAnnotation(
            id=Id.new(),
            original_report_id=request.original_report_id,
            corrected_report_id=request.corrected_report_id,
            annotator_id=request.annotator_id,
            submitted_at=datetime.now(),
            comment=request.comment,
        )
        await self.repository.save(annotation)
        return annotation
