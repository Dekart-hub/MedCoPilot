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
from pydantic import BaseModel

from dialogue.dialogue import DialogueId
from dialogue.use_cases import AddDialogue, AddDialogueCommand, TurnInput
from shared.value_objects import Id
from soap.repository import SoapReportRepository
from soap.serialization import report_to_dict
from soap.soap import SoapReportId
from soap.use_cases import (
    DialogueNotFoundError,
    ExtractSoapReport,
    ExtractSoapReportCommand,
)

from .dependencies import (
    SessionDep,
    get_add_dialogue,
    get_extract_soap_report,
    get_soap_report_repository,
)

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
