from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from dialogue import (
    CreateDialogue,
    CreateDialogueCommand,
    CreateDialogueFromText,
    DialogueRepository,
    DialogueTurnInput,
)
from shared.value_objects import Id
from di import (
    get_create_dialogue,
    get_create_dialogue_from_text,
    get_dialogue_repository,
)

from .schemas import (
    CreateDialogueFromTextRequest,
    CreateDialogueRequest,
    DialogueResponse,
)

router = APIRouter(prefix="/dialogues", tags=["dialogues"])


@router.get("", response_model=list[DialogueResponse])
async def list_dialogues(
    repository: DialogueRepository = Depends(get_dialogue_repository),
) -> list[DialogueResponse]:
    dialogues = await repository.list_all()
    return [DialogueResponse.from_domain(d) for d in dialogues]


@router.get("/{dialogue_id}", response_model=DialogueResponse)
async def get_dialogue(
    dialogue_id: str,
    repository: DialogueRepository = Depends(get_dialogue_repository),
) -> DialogueResponse:
    dialogue = await repository.get(Id.from_str(dialogue_id))
    if dialogue is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Dialogue not found")
    return DialogueResponse.from_domain(dialogue)


@router.post("", response_model=DialogueResponse, status_code=status.HTTP_201_CREATED)
async def create_dialogue(
    body: CreateDialogueRequest,
    use_case: CreateDialogue = Depends(get_create_dialogue),
) -> DialogueResponse:
    command = CreateDialogueCommand(
        turns=[DialogueTurnInput(role=t.role, content=t.content) for t in body.turns]
    )
    dialogue = await use_case.execute(command)
    return DialogueResponse.from_domain(dialogue)


@router.post(
    "/from-text",
    response_model=DialogueResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_dialogue_from_text(
    body: CreateDialogueFromTextRequest,
    use_case: CreateDialogueFromText = Depends(get_create_dialogue_from_text),
) -> DialogueResponse:
    dialogue = await use_case.execute(body.text)
    return DialogueResponse.from_domain(dialogue)
