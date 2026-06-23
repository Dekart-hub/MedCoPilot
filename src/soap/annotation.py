# src/soap/annotation.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional

from shared.entity import Entity
from shared.value_objects import Id
from .soap import SoapReport, SoapNote, SoapClaim, SoapReportId

type AnnotationId = Id[SoapAnnotation]
type AnnotationItemId = Id[AnnotationItem]


class AnnotationStatus(Enum):
    PENDING = auto()      # Ожидает разметки
    IN_PROGRESS = auto()  # В процессе
    COMPLETED = auto()    # Завершена врачом


class ChangeType(Enum):
    MODIFIED = auto()     # Изменён текст
    DELETED = auto()      # Удалён
    ADDED = auto()        # Добавлен новый
    EVIDENCE_CHANGED = auto()  # Изменена цитата
    SCORE_ADJUSTED = auto()    # Скорректирована оценка


@dataclass(eq=False, slots=True)
class AnnotationItem(Entity[AnnotationItemId]):
    """Одно изменение в SOAP note."""
    id: AnnotationItemId
    original_claim: Optional[SoapClaim]  # Оригинал (может быть None если добавлено)
    corrected_claim: Optional[SoapClaim]  # Исправление (может быть None если удалено)
    change_type: ChangeType
    comment: Optional[str] = None  # Комментарий врача почему изменил
    annotated_at: datetime = field(default_factory=datetime.now)


@dataclass(eq=False, slots=True)
class SoapAnnotation(Entity[AnnotationId]):
    """Разметка всего SOAP отчёта врачом."""
    id: AnnotationId
    original_report_id: SoapReportId  # Ссылка на оригинал
    annotated_report: SoapReport      # Исправленная версия
    annotator_id: str                  # ID врача (пока строка, потом можно сделать entity)
    status: AnnotationStatus
    annotation_items: list[AnnotationItem] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    notes: Optional[str] = None  # Общие заметки врача
    
    def complete(self) -> None:
        """Завершить разметку."""
        self.status = AnnotationStatus.COMPLETED
        self.completed_at = datetime.now()