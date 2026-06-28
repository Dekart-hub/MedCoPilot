from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from dialogue import DialogueRepository
from shared.value_objects import Id
from soap import ExtractScoredSoap

from di import get_dialogue_repository, get_extract_scored_soap

from .schemas import CreateReportRequest, ReportResponse

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
async def create_report(
    body: CreateReportRequest,
    use_case: ExtractScoredSoap = Depends(get_extract_scored_soap),
    repository: DialogueRepository = Depends(get_dialogue_repository),
) -> ReportResponse:
    dialogue = await repository.get(Id.from_str(body.dialogue_id))
    if dialogue is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail="Dialogue not found"
        )
    view = await use_case.execute(dialogue)
    return ReportResponse.from_domain(view)
