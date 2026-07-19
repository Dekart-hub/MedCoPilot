"""REST endpoints for dialogues and their extracted SOAP reports.

Stateless: every handler resolves its collaborators from the request-scoped
session and commits the unit of work itself once the use case succeeds — the
repositories never commit. Extraction is idempotent, so ``POST`` on a dialogue's
report is safe to retry: it returns the stored report without a second LLM call.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from dialogue.dialogue import DialogueId, DialogueTurnId
from dialogue.use_cases import AddDialogue, AddDialogueCommand, TurnInput
from shared.value_objects import Id
from soap.correction import CorrectionId, SoapReportCorrection
from soap.correction_repository import SoapReportCorrectionRepository
from soap.correction_use_cases import (
    AddCorrectedNote,
    AddCorrectedNoteCommand,
    DeleteCorrectedNote,
    DeleteCorrectedNoteCommand,
    ReopenSoapCorrection,
    ReopenSoapCorrectionCommand,
    StartSoapCorrection,
    StartSoapCorrectionCommand,
    UpdateCorrectedNote,
    UpdateCorrectedNoteCommand,
    VerifySoapCorrection,
    VerifySoapCorrectionCommand,
)
from soap.repository import SoapReportRepository
from soap.serialization import correction_to_dict, report_to_dict
from soap.soap import (
    AssessmentClaim,
    IcdCoding,
    SoapClaim,
    SoapNoteId,
    SoapReportId,
    TurnCitation,
)
from soap.use_cases import (
    DialogueNotFoundError,
    ExtractSoapReport,
    ExtractSoapReportCommand,
)

from .dependencies import (
    SessionDep,
    get_add_corrected_note,
    get_add_dialogue,
    get_correction_repository,
    get_delete_corrected_note,
    get_extract_soap_report,
    get_reopen_soap_correction,
    get_soap_report_repository,
    get_start_soap_correction,
    get_update_corrected_note,
    get_verify_soap_correction,
)
from .errors import CorrectionForReportNotFound

router = APIRouter()


class TurnPayload(BaseModel):
    speaker: str
    text: str


class DialoguePayload(BaseModel):
    turns: list[TurnPayload]


class DialogueCreated(BaseModel):
    id: str


@router.post("/dialogues", status_code=201, tags=["dialogues"])
async def create_dialogue(
    payload: DialoguePayload,
    session: SessionDep,
    add_dialogue: Annotated[AddDialogue, Depends(get_add_dialogue)],
) -> DialogueCreated:
    """Persist a new dialogue and return its id."""
    command = AddDialogueCommand(
        turns=[TurnInput(speaker=turn.speaker, text=turn.text) for turn in payload.turns]
    )
    dialogue_id = await add_dialogue.execute(command)
    await session.commit()
    return DialogueCreated(id=str(dialogue_id))


@router.post("/dialogues/{dialogue_id}/report", tags=["reports"])
async def extract_report(
    dialogue_id: UUID,
    session: SessionDep,
    extract: Annotated[ExtractSoapReport, Depends(get_extract_soap_report)],
    patient_id: str | None = None,
) -> dict[str, Any]:
    """Extract (idempotently) the SOAP report for a dialogue and return it."""
    dialogue_key: DialogueId = Id(dialogue_id)
    try:
        report = await extract.execute(
            ExtractSoapReportCommand(dialogue_id=dialogue_key, patient_id=patient_id)
        )
    except DialogueNotFoundError as exc:
        raise HTTPException(status_code=404, detail="dialogue not found") from exc
    await session.commit()
    return report_to_dict(report)


@router.get("/reports/{report_id}", tags=["reports"])
async def get_report(
    report_id: UUID,
    reports: Annotated[SoapReportRepository, Depends(get_soap_report_repository)],
) -> dict[str, Any]:
    """Return a persisted SOAP report by id."""
    report_key: SoapReportId = Id(report_id)
    report = await reports.get(report_key)
    if report is None:
        raise HTTPException(status_code=404, detail="report not found")
    return report_to_dict(report)


class CitationPayload(BaseModel):
    turn_id: UUID
    quote: str | None = None


class ClaimPayload(BaseModel):
    text: str
    citations: list[CitationPayload] = Field(min_length=1)


class IcdPayload(BaseModel):
    code: str
    name: str
    classifier_url: str


class AssessmentClaimPayload(ClaimPayload):
    icd: IcdPayload | None = None


class NoteContentPayload(BaseModel):
    """The four SOAP sections of a corrected note; each claim keeps its citations."""

    subjective: list[ClaimPayload] = Field(default_factory=list)
    objective: list[ClaimPayload] = Field(default_factory=list)
    assessment: list[AssessmentClaimPayload] = Field(default_factory=list)
    plan: list[ClaimPayload] = Field(default_factory=list)


class VerifyPayload(BaseModel):
    doctor_id: str


CorrectionRepositoryDep = Annotated[
    SoapReportCorrectionRepository, Depends(get_correction_repository)
]


@router.post("/reports/{report_id}/correction", tags=["corrections"])
async def start_correction(
    report_id: UUID,
    session: SessionDep,
    start: Annotated[StartSoapCorrection, Depends(get_start_soap_correction)],
) -> dict[str, Any]:
    """Open the doctor's draft correction of a report, or resume the existing one."""
    report_key: SoapReportId = Id(report_id)
    correction = await start.execute(StartSoapCorrectionCommand(report_id=report_key))
    await session.commit()
    return correction_to_dict(correction)


@router.get("/reports/{report_id}/correction", tags=["corrections"])
async def get_correction(
    report_id: UUID,
    corrections: CorrectionRepositoryDep,
) -> dict[str, Any]:
    """Return the doctor's draft or verified correction of a report."""
    correction = await _resolve_correction(corrections, report_id)
    return correction_to_dict(correction)


@router.put("/reports/{report_id}/correction/notes/{note_id}", tags=["corrections"])
async def update_note(
    report_id: UUID,
    note_id: UUID,
    payload: NoteContentPayload,
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    update: Annotated[UpdateCorrectedNote, Depends(get_update_corrected_note)],
) -> dict[str, Any]:
    """Replace a corrected note's sections, citations and Assessment ICD."""
    correction = await _resolve_correction(corrections, report_id)
    note_key: SoapNoteId = Id(note_id)
    await update.execute(
        UpdateCorrectedNoteCommand(
            correction_id=correction.id, note_id=note_key, **_sections(payload)
        )
    )
    await session.commit()
    return await _reload(corrections, correction.id)


@router.post("/reports/{report_id}/correction/notes", tags=["corrections"])
async def add_note(
    report_id: UUID,
    payload: NoteContentPayload,
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    add: Annotated[AddCorrectedNote, Depends(get_add_corrected_note)],
) -> dict[str, Any]:
    """Add a doctor-authored note (no source lineage) to the correction."""
    correction = await _resolve_correction(corrections, report_id)
    await add.execute(AddCorrectedNoteCommand(correction_id=correction.id, **_sections(payload)))
    await session.commit()
    return await _reload(corrections, correction.id)


@router.delete("/reports/{report_id}/correction/notes/{note_id}", tags=["corrections"])
async def delete_note(
    report_id: UUID,
    note_id: UUID,
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    delete: Annotated[DeleteCorrectedNote, Depends(get_delete_corrected_note)],
) -> dict[str, Any]:
    """Drop a note from the doctor's version of the correction."""
    correction = await _resolve_correction(corrections, report_id)
    note_key: SoapNoteId = Id(note_id)
    await delete.execute(DeleteCorrectedNoteCommand(correction_id=correction.id, note_id=note_key))
    await session.commit()
    return await _reload(corrections, correction.id)


@router.post("/reports/{report_id}/correction/verify", tags=["corrections"])
async def verify_correction(
    report_id: UUID,
    payload: VerifyPayload,
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    verify: Annotated[VerifySoapCorrection, Depends(get_verify_soap_correction)],
) -> dict[str, Any]:
    """Move the correction DRAFT → VERIFIED under the given ``doctor_id``."""
    correction = await _resolve_correction(corrections, report_id)
    verified = await verify.execute(
        VerifySoapCorrectionCommand(correction_id=correction.id, doctor_id=payload.doctor_id)
    )
    await session.commit()
    return correction_to_dict(verified)


@router.post("/reports/{report_id}/correction/reopen", tags=["corrections"])
async def reopen_correction(
    report_id: UUID,
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    reopen: Annotated[ReopenSoapCorrection, Depends(get_reopen_soap_correction)],
) -> dict[str, Any]:
    """Move the correction VERIFIED → DRAFT so the doctor can edit again."""
    correction = await _resolve_correction(corrections, report_id)
    reopened = await reopen.execute(ReopenSoapCorrectionCommand(correction_id=correction.id))
    await session.commit()
    return correction_to_dict(reopened)


async def _resolve_correction(
    corrections: SoapReportCorrectionRepository, report_id: UUID
) -> SoapReportCorrection:
    report_key: SoapReportId = Id(report_id)
    correction = await corrections.get_by_source_report_id(report_key)
    if correction is None:
        raise CorrectionForReportNotFound(report_key)
    return correction


async def _reload(
    corrections: SoapReportCorrectionRepository, correction_id: CorrectionId
) -> dict[str, Any]:
    correction = await corrections.get(correction_id)
    if correction is None:
        raise CorrectionForReportNotFound(correction_id)
    return correction_to_dict(correction)


def _sections(payload: NoteContentPayload) -> dict[str, Any]:
    return {
        "subjective": [_to_claim(claim) for claim in payload.subjective],
        "objective": [_to_claim(claim) for claim in payload.objective],
        "assessment": [_to_assessment(claim) for claim in payload.assessment],
        "plan": [_to_claim(claim) for claim in payload.plan],
    }


def _to_claim(payload: ClaimPayload) -> SoapClaim:
    return SoapClaim(
        id=Id.new(),
        text=payload.text,
        citations=[_to_citation(citation) for citation in payload.citations],
    )


def _to_assessment(payload: AssessmentClaimPayload) -> AssessmentClaim:
    return AssessmentClaim(
        id=Id.new(),
        text=payload.text,
        citations=[_to_citation(citation) for citation in payload.citations],
        icd=_to_icd(payload.icd) if payload.icd is not None else None,
    )


def _to_citation(payload: CitationPayload) -> TurnCitation:
    turn_id: DialogueTurnId = Id(payload.turn_id)
    return TurnCitation(turn_id=turn_id, quote=payload.quote)


def _to_icd(payload: IcdPayload) -> IcdCoding:
    return IcdCoding(code=payload.code, name=payload.name, classifier_url=payload.classifier_url)
