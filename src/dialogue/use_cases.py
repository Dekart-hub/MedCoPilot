from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from shared.value_objects import Id

from .dialogue import Dialogue, DialogueTurn
from .repository import DialogueRepository


@dataclass(frozen=True, slots=True)
class DialogueTurnInput:
    """Одна реплика в структурированной команде создания диалога."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class CreateDialogueCommand:
    turns: list[DialogueTurnInput]
    patient_ref: str | None = None
    encounter_ref: str | None = None


class CreateDialogue:
    """Создаёт диалог из структурированной команды (готовые реплики)."""

    def __init__(self, repository: DialogueRepository) -> None:
        self._repository = repository

    async def execute(self, command: CreateDialogueCommand) -> Dialogue:
        now = datetime.now(timezone.utc)
        turns = [
            DialogueTurn(
                id=Id.new(),
                role=turn.role,
                content=turn.content,
                timestamp=now,
            )
            for turn in command.turns
        ]
        dialogue = Dialogue(
            id=Id.new(),
            turns=turns,
            created_at=now,
            patient_ref=command.patient_ref,
            encounter_ref=command.encounter_ref,
        )
        await self._repository.save(dialogue)
        return dialogue


class CreateDialogueFromText:
    """Создаёт диалог из «сырого» текста (см. :meth:`Dialogue.from_text`)."""

    def __init__(self, repository: DialogueRepository) -> None:
        self._repository = repository

    async def execute(
        self,
        text: str,
        *,
        patient_ref: str | None = None,
        encounter_ref: str | None = None,
    ) -> Dialogue:
        dialogue = Dialogue.from_text(
            text,
            patient_ref=patient_ref,
            encounter_ref=encounter_ref,
        )
        await self._repository.save(dialogue)
        return dialogue
