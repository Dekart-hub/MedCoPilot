"""Application use cases for dialogues."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .dialogue import Dialogue, DialogueId
from .repository import DialogueRepository


@dataclass(frozen=True, slots=True)
class TurnInput:
    """One turn in an :class:`AddDialogueCommand`."""

    speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class AddDialogueCommand:
    """Request to persist a new dialogue built from the given turns."""

    turns: Sequence[TurnInput]


class AddDialogue:
    """Builds a dialogue from a command, persists it and returns its id."""

    def __init__(self, repository: DialogueRepository) -> None:
        self._repository = repository

    async def execute(self, command: AddDialogueCommand) -> DialogueId:
        dialogue = Dialogue.start()
        for turn in command.turns:
            dialogue.add_turn(turn.speaker, turn.text)
        await self._repository.save(dialogue)
        return dialogue.id
