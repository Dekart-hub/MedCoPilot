from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from dialogue import DialogueRepository
from ehr import (
    ApprovalConflictError,
    EhrGatewayError,
    EhrIntegrationDisabledError,
    InvalidEhrReferenceError,
    ReportNotApprovedError,
    ReportNotFoundError,
    ReportWorkflow,
    UnlinkedDialogueError,
)
from shared.value_objects import Id
from soap import ExtractScoredSoap

from di import (
    get_dialogue_repository,
    get_extract_scored_soap,
    get_report_workflow,
)

from .schemas import (
    ApproveReportRequest,
    CreateReportRequest,
    ReportResponse,
    ReportWorkflowResponse,
)

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
async def create_report(
    body: CreateReportRequest,
    use_case: ExtractScoredSoap = Depends(get_extract_scored_soap),
    repository: DialogueRepository = Depends(get_dialogue_repository),
    workflow: ReportWorkflow = Depends(get_report_workflow),
) -> ReportResponse:
    dialogue = await repository.get(Id.from_str(body.dialogue_id))
    if dialogue is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail="Dialogue not found"
        )
    view = await use_case.execute(dialogue)
    await workflow.store_generated_report(view, dialogue)
    return ReportResponse.from_domain(view)


@router.get("/{report_id}/workflow", response_model=ReportWorkflowResponse)
async def get_report_workflow_status(
    report_id: str,
    workflow: ReportWorkflow = Depends(get_report_workflow),
) -> ReportWorkflowResponse:
    try:
        record = await workflow.get_record(report_id)
    except ReportNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ReportWorkflowResponse.from_domain(record)


@router.post("/{report_id}/approve", response_model=ReportWorkflowResponse)
async def approve_report(
    report_id: str,
    body: ApproveReportRequest,
    workflow: ReportWorkflow = Depends(get_report_workflow),
) -> ReportWorkflowResponse:
    try:
        record = await workflow.approve(report_id, body.clinician_ref)
    except ReportNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except InvalidEhrReferenceError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except ApprovalConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return ReportWorkflowResponse.from_domain(record)


@router.post("/{report_id}/ehr-sync", response_model=ReportWorkflowResponse)
async def sync_report_to_ehr(
    report_id: str,
    workflow: ReportWorkflow = Depends(get_report_workflow),
) -> ReportWorkflowResponse:
    try:
        record = await workflow.sync(report_id)
    except ReportNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (ReportNotApprovedError, UnlinkedDialogueError) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except InvalidEhrReferenceError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except EhrIntegrationDisabledError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except EhrGatewayError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return ReportWorkflowResponse.from_domain(record)
