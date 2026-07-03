from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from shared.entity import Entity
from shared.value_objects import AnnotatorId, Id
from soap.soap import SoapReportId

type AnnotationId = Id[SoapAnnotation]


@dataclass(eq=False, slots=True)
class SoapAnnotation(Entity[AnnotationId]):
    """Факт врачебной разметки SOAP-отчёта.

    Исправления хранятся не как список патчей, а как отдельный исправленный
    ``SoapReport``. Эта сущность связывает исходную и исправленную версии и
    фиксирует автора разметки.
    """

    id: AnnotationId
    original_report_id: SoapReportId
    corrected_report_id: SoapReportId
    annotator_id: AnnotatorId
    submitted_at: datetime
    comment: str = ""
