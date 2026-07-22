"""REST endpoints for dialogues and their extracted SOAP reports.

Stateless: every handler resolves its collaborators from the request-scoped
session and commits the unit of work itself once the use case succeeds — the
repositories never commit. Extraction is idempotent, so ``POST`` on a dialogue's
report is safe to retry: it returns the stored report without a second LLM call.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from dialogue.dialogue import DialogueId, DialogueTurnId
from dialogue.repository import DialogueRepository
from dialogue.serialization import dialogue_to_dict
from dialogue.use_cases import AddDialogue, AddDialogueCommand, TurnInput
from ehr.publication import PublicationStatus
from ehr.publication_use_cases import (
    GetEhrPublication,
    RequestEhrPublication,
    RequestEhrPublicationCommand,
)
from ehr.serialization import publication_to_dict
from icd.dictionary import classifier_url as icd_classifier_url
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
from soap.editor_use_cases import (
    AcceptanceMetricQuery,
    AcceptProposalOperation,
    ComputeAcceptanceMetric,
    DecideOperationCommand,
    EnsureNoPendingProposal,
    GetCurrentProposal,
    ProposeCorrectionEdit,
    RejectPendingProposals,
    RejectProposalOperation,
)
from soap.quality_use_cases import GetDialogueSoapQuality
from soap.repository import SoapReportRepository
from soap.serialization import (
    acceptance_metric_to_dict,
    correction_to_dict,
    proposal_to_dict,
    quality_to_dict,
    report_summary_to_dict,
    report_to_dict,
)
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
    get_accept_proposal_operation,
    get_acceptance_metric,
    get_add_corrected_note,
    get_add_dialogue,
    get_correction_repository,
    get_current_proposal,
    get_delete_corrected_note,
    get_dialogue_repository,
    get_dialogue_soap_quality,
    get_ehr_publication,
    get_ensure_no_pending_proposal,
    get_extract_soap_report,
    get_propose_correction_edit,
    get_reject_pending_proposals,
    get_reject_proposal_operation,
    get_reopen_soap_correction,
    get_request_ehr_publication,
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


@router.get("/dialogues/{dialogue_id}", tags=["dialogues"])
async def get_dialogue(
    dialogue_id: UUID,
    dialogues: Annotated[DialogueRepository, Depends(get_dialogue_repository)],
) -> dict[str, Any]:
    """Return a persisted dialogue with its ordered turns by id."""
    dialogue_key: DialogueId = Id(dialogue_id)
    dialogue = await dialogues.get(dialogue_key)
    if dialogue is None:
        raise HTTPException(status_code=404, detail="dialogue not found")
    return dialogue_to_dict(dialogue)


@router.get("/dialogues/{dialogue_id}/quality", tags=["quality"])
async def get_dialogue_quality(
    dialogue_id: UUID,
    quality: Annotated[GetDialogueSoapQuality, Depends(get_dialogue_soap_quality)],
) -> dict[str, Any]:
    """Calculate SOAP quality from the dialogue's current verified correction."""
    dialogue_key: DialogueId = Id(dialogue_id)
    result = await quality.execute(dialogue_key)
    return quality_to_dict(result)


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


@router.get("/reports", tags=["reports"])
async def list_reports(
    reports: Annotated[SoapReportRepository, Depends(get_soap_report_repository)],
) -> list[dict[str, Any]]:
    """Return every persisted SOAP report as a summary, newest first."""
    summaries = await reports.list_summaries()
    return [report_summary_to_dict(summary) for summary in summaries]


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
    """A manually entered coding. ``classifier_url`` is accepted for backward
    compatibility but ignored: the server derives the reference URL from the
    code, so a client can never store an arbitrary link."""

    code: str
    name: str
    classifier_url: str | None = None


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


RejectPendingDep = Annotated[RejectPendingProposals, Depends(get_reject_pending_proposals)]

_DOCTOR_EDIT = "doctor_edit"


@router.put("/reports/{report_id}/correction/notes/{note_id}", tags=["corrections"])
async def update_note(
    report_id: UUID,
    note_id: UUID,
    payload: NoteContentPayload,
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    update: Annotated[UpdateCorrectedNote, Depends(get_update_corrected_note)],
    reject_pending: RejectPendingDep,
) -> dict[str, Any]:
    """Replace a corrected note's sections, citations and Assessment ICD."""
    correction = await _resolve_correction(corrections, report_id)
    note_key: SoapNoteId = Id(note_id)
    await update.execute(
        UpdateCorrectedNoteCommand(
            correction_id=correction.id, note_id=note_key, **_sections(payload)
        )
    )
    await reject_pending.execute(correction, reason=_DOCTOR_EDIT)
    await session.commit()
    return await _reload(corrections, correction.id)


@router.post("/reports/{report_id}/correction/notes", tags=["corrections"])
async def add_note(
    report_id: UUID,
    payload: NoteContentPayload,
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    add: Annotated[AddCorrectedNote, Depends(get_add_corrected_note)],
    reject_pending: RejectPendingDep,
) -> dict[str, Any]:
    """Add a doctor-authored note (no source lineage) to the correction."""
    correction = await _resolve_correction(corrections, report_id)
    await add.execute(AddCorrectedNoteCommand(correction_id=correction.id, **_sections(payload)))
    await reject_pending.execute(correction, reason=_DOCTOR_EDIT)
    await session.commit()
    return await _reload(corrections, correction.id)


@router.delete("/reports/{report_id}/correction/notes/{note_id}", tags=["corrections"])
async def delete_note(
    report_id: UUID,
    note_id: UUID,
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    delete: Annotated[DeleteCorrectedNote, Depends(get_delete_corrected_note)],
    reject_pending: RejectPendingDep,
) -> dict[str, Any]:
    """Drop a note from the doctor's version of the correction."""
    correction = await _resolve_correction(corrections, report_id)
    note_key: SoapNoteId = Id(note_id)
    await delete.execute(DeleteCorrectedNoteCommand(correction_id=correction.id, note_id=note_key))
    await reject_pending.execute(correction, reason=_DOCTOR_EDIT)
    await session.commit()
    return await _reload(corrections, correction.id)


@router.post("/reports/{report_id}/correction/verify", tags=["corrections"])
async def verify_correction(
    report_id: UUID,
    payload: VerifyPayload,
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    verify: Annotated[VerifySoapCorrection, Depends(get_verify_soap_correction)],
    ensure_no_pending: Annotated[EnsureNoPendingProposal, Depends(get_ensure_no_pending_proposal)],
) -> dict[str, Any]:
    """Move the correction DRAFT → VERIFIED under the given ``doctor_id``."""
    correction = await _resolve_correction(corrections, report_id)
    await ensure_no_pending.execute(correction)
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


class ProposeEditPayload(BaseModel):
    user_request: str
    patient_id: str


@router.post("/reports/{report_id}/correction/editor/proposals", status_code=201, tags=["editor"])
async def create_proposal(
    report_id: UUID,
    payload: ProposeEditPayload,
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    propose: Annotated[ProposeCorrectionEdit, Depends(get_propose_correction_edit)],
) -> dict[str, Any]:
    """Draft an LLM edit of the report's correction; persist PENDING ops, apply nothing."""
    correction = await _resolve_correction(corrections, report_id)
    proposal = await propose.execute(
        correction, user_request=payload.user_request, patient_id=payload.patient_id
    )
    await session.commit()
    return proposal_to_dict(proposal)


@router.get("/reports/{report_id}/correction/editor/proposals", tags=["editor"])
async def get_proposal(
    report_id: UUID,
    corrections: CorrectionRepositoryDep,
    current: Annotated[GetCurrentProposal, Depends(get_current_proposal)],
) -> dict[str, Any]:
    """Return the active proposal (or the most recent one) with each op's diff and decision."""
    correction = await _resolve_correction(corrections, report_id)
    proposal = await current.execute(correction)
    return proposal_to_dict(proposal)


@router.post(
    "/reports/{report_id}/correction/editor/proposals/{proposal_id}/operations/{operation_id}/accept",
    tags=["editor"],
)
async def accept_operation(
    report_id: UUID,
    proposal_id: UUID,
    operation_id: UUID,
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    accept: Annotated[AcceptProposalOperation, Depends(get_accept_proposal_operation)],
) -> dict[str, Any]:
    """Accept one operation and apply it through the #8 correction use cases."""
    correction = await _resolve_correction(corrections, report_id)
    proposal = await accept.execute(correction, _decide(proposal_id, operation_id))
    await session.commit()
    return proposal_to_dict(proposal)


@router.post(
    "/reports/{report_id}/correction/editor/proposals/{proposal_id}/operations/{operation_id}/reject",
    tags=["editor"],
)
async def reject_operation(
    report_id: UUID,
    proposal_id: UUID,
    operation_id: UUID,
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    reject: Annotated[RejectProposalOperation, Depends(get_reject_proposal_operation)],
) -> dict[str, Any]:
    """Reject one operation, leaving the correction unchanged."""
    correction = await _resolve_correction(corrections, report_id)
    proposal = await reject.execute(correction, _decide(proposal_id, operation_id))
    await session.commit()
    return proposal_to_dict(proposal)


@router.get("/reports/{report_id}/correction/editor/metric", tags=["editor"])
async def get_editor_metric(
    report_id: UUID,
    corrections: CorrectionRepositoryDep,
    metric: Annotated[ComputeAcceptanceMetric, Depends(get_acceptance_metric)],
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Return the operation-level acceptance metric of the report's editor session."""
    correction = await _resolve_correction(corrections, report_id)
    result = await metric.execute(correction, AcceptanceMetricQuery(since=since, until=until))
    return acceptance_metric_to_dict(result)


def _decide(proposal_id: UUID, operation_id: UUID) -> DecideOperationCommand:
    return DecideOperationCommand(proposal_id=Id(proposal_id), operation_id=Id(operation_id))


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
    # The URL is always server-derived; validation against the catalog (and
    # canonicalisation of the title) happens in the correction use case.
    return IcdCoding(
        code=payload.code, name=payload.name, classifier_url=icd_classifier_url(payload.code)
    )


class PublicationPayload(BaseModel):
    patient_ref: str = Field(min_length=1)
    encounter_ref: str = Field(min_length=1)
    author_ref: str = Field(min_length=1)


@router.post("/reports/{report_id}/publication", tags=["publications"])
async def request_publication(
    report_id: UUID,
    payload: PublicationPayload,
    session: SessionDep,
    request: Annotated[RequestEhrPublication, Depends(get_request_ehr_publication)],
) -> JSONResponse:
    """Accept a verified correction for durable idempotent FHIR delivery."""
    delivery = await request.execute(
        RequestEhrPublicationCommand(
            report_id=Id(report_id),
            patient_ref=payload.patient_ref,
            encounter_ref=payload.encounter_ref,
            author_ref=payload.author_ref,
        )
    )
    await session.commit()
    status_code = 202 if delivery.publication.status is PublicationStatus.PENDING else 200
    return JSONResponse(publication_to_dict(delivery), status_code=status_code)


@router.get("/reports/{report_id}/publication", tags=["publications"])
async def get_publication(
    report_id: UUID,
    get: Annotated[GetEhrPublication, Depends(get_ehr_publication)],
) -> dict[str, Any]:
    """Return durable delivery status, attempts and the remote Bundle reference."""
    return publication_to_dict(await get.execute(Id(report_id)))
