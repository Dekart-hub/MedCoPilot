"""Unit tests for the dialogue domain and the AddDialogue use case."""

from __future__ import annotations

import asyncio

from dialogue.dialogue import Dialogue
from dialogue.repository import DialogueRepository
from dialogue.use_cases import AddDialogue, AddDialogueCommand, TurnInput
from shared.value_objects import Id


class _FakeDialogueRepository(DialogueRepository):
    def __init__(self) -> None:
        self.saved: dict[Id[Dialogue], Dialogue] = {}

    async def save(self, dialogue: Dialogue) -> None:
        self.saved[dialogue.id] = dialogue

    async def get(self, dialogue_id: Id[Dialogue]) -> Dialogue | None:
        return self.saved.get(dialogue_id)


def test_entities_are_equal_by_identity_not_by_value() -> None:
    shared_id: Id[Dialogue] = Id.new()
    one = Dialogue(id=shared_id)
    one.add_turn("doctor", "How are you?")
    other = Dialogue(id=shared_id)

    assert one == other  # same identity, different turns
    assert one != Dialogue.start()  # fresh identity


def test_add_turn_keeps_order_and_assigns_distinct_ids() -> None:
    dialogue = Dialogue.start()

    first = dialogue.add_turn("doctor", "What brings you in?")
    second = dialogue.add_turn("patient", "A headache.")

    assert dialogue.turns == [first, second]
    assert first.id != second.id
    assert [turn.speaker for turn in dialogue.turns] == ["doctor", "patient"]


def test_add_dialogue_persists_turns_in_order_and_returns_id() -> None:
    repository = _FakeDialogueRepository()
    command = AddDialogueCommand(
        turns=[
            TurnInput(speaker="doctor", text="What brings you in?"),
            TurnInput(speaker="patient", text="A headache for three days."),
        ]
    )

    dialogue_id = asyncio.run(AddDialogue(repository).execute(command))

    stored = repository.saved[dialogue_id]
    assert [(turn.speaker, turn.text) for turn in stored.turns] == [
        ("doctor", "What brings you in?"),
        ("patient", "A headache for three days."),
    ]
