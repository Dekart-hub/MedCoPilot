from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from shared.entity import Entity
from shared.value_objects import Id
from soap.soap import SoapReport, SoapReportId, SoapClaim

from .enums import AnnotationStatus, ChangeType

# Фантомные типы для ID
type AnnotationId = Id[SoapAnnotation]
type AnnotationItemId = Id[AnnotationItem]


@dataclass(frozen=True, slots=True)
class ChangeDescription:
    """Строгое описание одного изменения."""
    change_type: ChangeType
    field_name: str  # Какое поле изменено: "claim", "evidence"
    old_value: Optional[str]
    new_value: Optional[str]
    claim_id: Optional[str] = None
    comment: Optional[str] = None


@dataclass(eq=False, slots=True)
class AnnotationItem(Entity[AnnotationItemId]):
    """Одно конкретное изменение в SOAP заметке."""
    id: AnnotationItemId
    original_claim: Optional[SoapClaim]
    corrected_claim: Optional[SoapClaim]
    change_type: ChangeType
    comment: Optional[str] = None
    annotated_at: datetime = field(default_factory=datetime.now)


@dataclass(eq=False, slots=True)
class SoapAnnotation(Entity[AnnotationId]):
    """Главная сущность разметки: связывает оригинал, исправление и врача."""
    id: AnnotationId
    original_report_id: SoapReportId
    annotated_report: SoapReport  # Полностью исправленная версия отчёта
    annotator_id: str  
    status: AnnotationStatus
    changes: list[ChangeDescription] = field(default_factory=list)
    annotation_items: list[AnnotationItem] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    notes: Optional[str] = None  # Общий комментарий врача к разметке

    def complete(self) -> None:
        """Переводит разметку в статус завершённой."""
        self.status = AnnotationStatus.COMPLETED
        self.completed_at = datetime.now()