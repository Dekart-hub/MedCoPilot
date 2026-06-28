from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from shared.entity import Entity
from shared.value_objects import Id, AnnotatorId, ClaimId
from soap.soap import SoapReport, SoapReportId, SoapClaim

from .enums import AnnotationStatus
from .changes import SoapChange

type AnnotationId = Id[SoapAnnotation]
type AnnotationItemId = Id[AnnotationItem]

_NOT_COMPLETED = datetime.min


@dataclass(eq=False, slots=True)
class AnnotationItem(Entity[AnnotationItemId]):
    """Сущность одного изменения."""
    id: AnnotationItemId
    change: SoapChange
    comment: str = ""

    def with_comment(self, comment: str) -> AnnotationItem:
        return AnnotationItem(
            id=self.id,
            change=self.change,
            comment=comment
        )


@dataclass(eq=False, slots=True)
class SoapAnnotation(Entity[AnnotationId]):
    """Главная сущность разметки."""
    id: AnnotationId
    original_report_id: SoapReportId
    annotated_report: SoapReport
    annotator_id: AnnotatorId  # строгий тип
    
    _status: AnnotationStatus = field(default=AnnotationStatus.PENDING, init=False)
    _started_at: datetime = field(default_factory=datetime.now, init=False)
    _completed_at: datetime = field(default=_NOT_COMPLETED, init=False)
    
    changes: list[AnnotationItem] = field(default_factory=list)
    notes: str = ""

    def start(self) -> None:
        if self._status != AnnotationStatus.PENDING:
            raise ValueError(f"Cannot start annotation in status {self._status}")
        self._status = AnnotationStatus.IN_PROGRESS

    def complete(self, completed_at: datetime) -> None:
        if self._status != AnnotationStatus.IN_PROGRESS:
            raise ValueError(f"Cannot complete annotation in status {self._status}")
        self._status = AnnotationStatus.COMPLETED
        self._completed_at = completed_at

    @property
    def status(self) -> AnnotationStatus:
        return self._status

    @property
    def started_at(self) -> datetime:
        return self._started_at

    @property
    def completed_at(self) -> datetime:
        if self._completed_at == _NOT_COMPLETED:
            raise ValueError("Annotation is not completed yet")
        return self._completed_at

    def is_completed(self) -> bool:
        return self._status == AnnotationStatus.COMPLETED