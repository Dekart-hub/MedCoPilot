from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from di import get_report_workflow
from ehr import (
    DialogueNotFoundError,
    EhrGatewayError,
    EhrIntegrationDisabledError,
    InvalidEhrReferenceError,
    ReportWorkflow,
    UnlinkedDialogueError,
)

from .schemas import PatientContextResponse

router = APIRouter(prefix="/ehr", tags=["mock-ehr"])


@router.get(
    "/dialogues/{dialogue_id}/context",
    response_model=PatientContextResponse,
)
async def get_dialogue_patient_context(
    dialogue_id: str,
    workflow: ReportWorkflow = Depends(get_report_workflow),
) -> PatientContextResponse:
    """Read the explicitly linked, pre-visit-only mock EHR context."""
    try:
        context = await workflow.get_patient_context(dialogue_id)
    except DialogueNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except UnlinkedDialogueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except InvalidEhrReferenceError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except EhrIntegrationDisabledError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except EhrGatewayError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return PatientContextResponse.from_domain(context)
