from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from shared.entity import Entity
from shared.value_objects import AnnotatorId, Id
from soap.soap import SoapNote, SoapNoteId

type CorrectedSoapNoteId = Id[CorrectedSoapNote]


@dataclass(eq=False, slots=True)
class CorrectedSoapNote(Entity[CorrectedSoapNoteId]):
    """Исправленная врачом SOAP-нота.

    Разметка хранит не набор патчей, а готовую исправленную ``SoapNote``.
    Ссылка на оригинальную ноту остаётся рядом, чтобы потом можно было
    сравнить исходный результат модели с врачебной правкой.
    """

    id: CorrectedSoapNoteId
    original_note_id: SoapNoteId
    corrected_note: SoapNote
    annotator_id: AnnotatorId
    submitted_at: datetime
    comment: str = ""
