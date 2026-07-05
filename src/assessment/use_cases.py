from dataclasses import dataclass
from datetime import datetime

from shared.value_objects import Id
from soap.soap import SoapClaim, SoapNote, SoapNoteId, SoapReportId

from .annotation import AddedSoapNote, CorrectedSoapNote
from .annotation_repository import AddedSoapNoteRepository, CorrectedSoapNoteRepository
from shared.value_objects import AnnotatorId


@dataclass(frozen=True, slots=True)
class SubmitCorrectedSoapNoteRequest:
    """Запрос на сохранение законченной врачебной правки SOAP-ноты."""

    report_id: SoapReportId
    original_note_id: SoapNoteId
    subjective: SoapClaim
    objective: SoapClaim
    assessment: SoapClaim
    plan: SoapClaim
    annotator_id: AnnotatorId
    comment: str = ""


@dataclass(frozen=True, slots=True)
class SubmitAddedSoapNoteRequest:
    """Запрос на сохранение SOAP-ноты, которую агент не извлёк."""

    report_id: SoapReportId
    subjective: SoapClaim
    objective: SoapClaim
    assessment: SoapClaim
    plan: SoapClaim
    annotator_id: AnnotatorId
    comment: str = ""


def _build_note(
    subjective: SoapClaim,
    objective: SoapClaim,
    assessment: SoapClaim,
    plan: SoapClaim,
) -> SoapNote:
    return SoapNote(
        id=Id.new(),
        subjective=subjective,
        objective=objective,
        assessment=assessment,
        plan=plan,
    )


class SubmitCorrectedSoapNote:
    def __init__(self, repository: CorrectedSoapNoteRepository) -> None:
        self.repository = repository

    async def execute(
        self, request: SubmitCorrectedSoapNoteRequest
    ) -> CorrectedSoapNote:
        note = _build_note(
            subjective=request.subjective,
            objective=request.objective,
            assessment=request.assessment,
            plan=request.plan,
        )
        corrected_note = CorrectedSoapNote(
            id=Id.new(),
            report_id=request.report_id,
            original_note_id=request.original_note_id,
            corrected_note=note,
            annotator_id=request.annotator_id,
            submitted_at=datetime.now(),
            comment=request.comment,
        )
        await self.repository.save(corrected_note)
        return corrected_note


class SubmitAddedSoapNote:
    def __init__(self, repository: AddedSoapNoteRepository) -> None:
        self.repository = repository

    async def execute(self, request: SubmitAddedSoapNoteRequest) -> AddedSoapNote:
        note = _build_note(
            subjective=request.subjective,
            objective=request.objective,
            assessment=request.assessment,
            plan=request.plan,
        )
        added_note = AddedSoapNote(
            id=Id.new(),
            report_id=request.report_id,
            added_note=note,
            annotator_id=request.annotator_id,
            submitted_at=datetime.now(),
            comment=request.comment,
        )
        await self.repository.save(added_note)
        return added_note
