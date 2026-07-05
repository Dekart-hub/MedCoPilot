from dataclasses import dataclass
from datetime import datetime

from shared.value_objects import Id
from soap.soap import SoapNote, SoapNoteId

from .annotation import CorrectedSoapNote
from .annotation_repository import CorrectedSoapNoteRepository
from shared.value_objects import AnnotatorId


@dataclass(frozen=True, slots=True)
class SubmitCorrectedSoapNoteRequest:
    """Запрос на сохранение законченной врачебной правки SOAP-ноты."""

    original_note_id: SoapNoteId
    corrected_note: SoapNote
    annotator_id: AnnotatorId
    comment: str = ""


class SubmitCorrectedSoapNote:
    def __init__(self, repository: CorrectedSoapNoteRepository) -> None:
        self.repository = repository

    async def execute(
        self, request: SubmitCorrectedSoapNoteRequest
    ) -> CorrectedSoapNote:
        corrected_note = CorrectedSoapNote(
            id=Id.new(),
            original_note_id=request.original_note_id,
            corrected_note=request.corrected_note,
            annotator_id=request.annotator_id,
            submitted_at=datetime.now(),
            comment=request.comment,
        )
        await self.repository.save(corrected_note)
        return corrected_note
