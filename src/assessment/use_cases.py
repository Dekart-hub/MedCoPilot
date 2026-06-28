from dataclasses import dataclass
from datetime import datetime

from shared.value_objects import Id
from soap.soap import SoapReport, SoapReportId, SoapNote, SoapNoteId

from .annotation import SoapAnnotation, AnnotationItem, AnnotationStatus
from .annotation_repository import AnnotationRepository
from shared.value_objects import AnnotatorId, ClaimId
from .changes import SoapChange, TextModified, NoteAdded


# --- DTO для запросов ---

@dataclass(frozen=True, slots=True)
class CorrectSoapNoteRequest:
    """Запрос на ИСПРАВЛЕНИЕ существующей заметки."""
    report_id: SoapReportId
    note_id: SoapNoteId
    annotator_id: AnnotatorId
    changes: list[SoapChange]  # Список конкретных изменений
    comment: str = ""


@dataclass(frozen=True, slots=True)
class CreateSoapNoteRequest:
    """Запрос на СОЗДАНИЕ новой заметки (если ИИ пропустил)."""
    report_id: SoapReportId
    annotator_id: AnnotatorId
    new_note: SoapNote
    comment: str = ""


# --- Use Cases ---

class CorrectSoapNote:
    def __init__(self, repository: AnnotationRepository) -> None:
        self.repository = repository

    async def execute(self, request: CorrectSoapNoteRequest) -> SoapAnnotation:
        # 1. Находим оригинальный отчет (get бросит ошибку, если нет)
        # В реальности здесь нужен SoapReportRepository, но для примера опустим
        # original_report = await report_repo.get_by_id(request.report_id)
        
        now = datetime.now()
        
        # 2. Создаем сущность разметки
        annotation = SoapAnnotation(
            id=Id.new(),
            original_report_id=request.report_id,
            annotated_report=SoapReport(id=SoapReportId.new(), soap_notes=[], created_at=now, updated_at=now), # Заглушка
            annotator_id=request.annotator_id,
        )
        
        annotation.start()
        
        # 3. Добавляем изменения
        for change in request.changes:
            item = AnnotationItem(
                id=Id.new(),
                change=change,
                comment=request.comment
            )
            annotation.changes.append(item)
            
        annotation.complete(now)
        await self.repository.save(annotation)
        return annotation


class CreateSoapNote:
    def __init__(self, repository: AnnotationRepository) -> None:
        self.repository = repository

    async def execute(self, request: CreateSoapNoteRequest) -> SoapAnnotation:
        now = datetime.now()
        
        annotation = SoapAnnotation(
            id=Id.new(),
            original_report_id=request.report_id,
            annotated_report=SoapReport(id=SoapReportId.new(), soap_notes=[request.new_note], created_at=now, updated_at=now),
            annotator_id=request.annotator_id,
        )
        
        annotation.start()
        
        # Создаем специальное изменение "Добавлена заметка"
        change = NoteAdded(claim_id=ClaimId("new"), note_text=request.new_note.assessment.text, section="assessment")
        item = AnnotationItem(id=Id.new(), change=change, comment=request.comment)
        annotation.changes.append(item)
        
        annotation.complete(now)
        await self.repository.save(annotation)
        return annotation