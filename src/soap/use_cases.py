import asyncio
from dataclasses import dataclass

from dialogue import Dialogue
from shared.value_objects import Id

from .extractor import SoapExtractor
from .score.score import SoapConfidenceReport
from .score.scorer import ConfidenceScorer
from .soap import SoapReport


from dataclasses import dataclass
from typing import Optional



@dataclass(frozen=True, slots=True)
class ScoredReport:
    """Результат use case: извлечённый отчёт вместе с оценками уверенности."""

    report: SoapReport
    confidence_report: SoapConfidenceReport


class ExtractScoredSoap:
    def __init__(self, extractor: SoapExtractor, scorer: ConfidenceScorer) -> None:
        self.extractor = extractor
        self.scorer = scorer

    async def execute(self, dialogue: Dialogue) -> ScoredReport:
        report = await self.extractor.extract(dialogue)
        note_scores = await asyncio.gather(
            *(self.scorer.score(dialogue, note) for note in report.soap_notes)
        )
        scores = SoapConfidenceReport(
            id=Id.new(),
            soap_report_id=report.id,
            confidence_scores=list(note_scores),
        )
        return ScoredReport(report=report, confidence_report=scores)

@dataclass(frozen=True, slots=True)
class AnnotationRequest:
    """Запрос на создание/обновление разметки."""
    original_report_id: SoapReportId
    corrected_notes: list[SoapNote]  # Исправленные заметки
    annotator_id: str
    changes: list[dict]  # Описание изменений
    comments: Optional[str] = None


class CreateSoapAnnotation:
    """Use case: создание разметки SOAP отчёта."""
    
    def __init__(self, annotation_repo: AnnotationRepository):
        self.repo = annotation_repo
    
    async def execute(self, request: AnnotationRequest) -> SoapAnnotation:
        # 1. Найти оригинальный report
        original = await self.repo.get_report(request.original_report_id)
        
        # 2. Создать исправленную версию
        corrected_report = SoapReport(
            id=SoapReportId.new(),
            soap_notes=request.corrected_notes,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        # 3. Создать annotation items для каждого изменения
        annotation_items = []
        for orig_note, corr_note in zip(original.soap_notes, request.corrected_notes):
            # Сравнить и создать items для каждого изменённого claim
            items = self._detect_changes(orig_note, corr_note)
            annotation_items.extend(items)
        
        # 4. Создать annotation
        annotation = SoapAnnotation(
            id=AnnotationId.new(),
            original_report_id=request.original_report_id,
            annotated_report=corrected_report,
            annotator_id=request.annotator_id,
            status=AnnotationStatus.COMPLETED,
            annotation_items=annotation_items,
            notes=request.comments,
        )
        annotation.complete()
        
        # 5. Сохранить
        await self.repo.save(annotation)
        
        return annotation
    
    def _detect_changes(self, original: SoapNote, corrected: SoapNote) -> list[AnnotationItem]:
        """Сравнить original и corrected, вернуть список изменений."""
        changes = []
        # Логика сравнения claims
        return changes