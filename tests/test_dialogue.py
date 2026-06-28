from __future__ import annotations

import asyncio

from dialogue import (
    SAMPLE_DIALOGUE_ID,
    CreateDialogue,
    CreateDialogueCommand,
    CreateDialogueFromText,
    Dialogue,
    DialogueTurnInput,
    InMemoryDialogueRepository,
    build_sample_dialogue,
)


def test_from_text_splits_lines_into_turns():
    dialogue = Dialogue.from_text(
        "person: I have a headache\n"
        "medic: Since when?\n"
        "person: Since yesterday\n"
    )

    assert [t.role for t in dialogue.turns] == ["person", "medic", "person"]
    assert dialogue.turns[0].content == "I have a headache"
    assert dialogue.turns[1].content == "Since when?"


def test_from_text_skips_blank_lines():
    dialogue = Dialogue.from_text("person hi\n\n   \nmedic hello\n")

    assert len(dialogue.turns) == 2
    assert dialogue.turns[0].role == "person"
    assert dialogue.turns[0].content == "hi"


def test_create_dialogue_persists_and_returns():
    repo = InMemoryDialogueRepository()
    use_case = CreateDialogue(repo)
    command = CreateDialogueCommand(
        turns=[
            DialogueTurnInput(role="person", content="hi"),
            DialogueTurnInput(role="medic", content="hello"),
        ]
    )

    dialogue = asyncio.run(use_case.execute(command))

    assert asyncio.run(repo.get(dialogue.id)) is dialogue
    assert [t.role for t in dialogue.turns] == ["person", "medic"]


def test_repository_seeds_initial_dialogues():
    repo = InMemoryDialogueRepository(initial=[build_sample_dialogue()])

    stored = asyncio.run(repo.get(SAMPLE_DIALOGUE_ID))

    assert stored is not None
    assert stored.id == SAMPLE_DIALOGUE_ID
    assert len(stored.turns) == 18


def test_create_dialogue_from_text_persists():
    repo = InMemoryDialogueRepository()
    use_case = CreateDialogueFromText(repo)

    dialogue = asyncio.run(use_case.execute("person hi\nmedic hello"))

    stored = asyncio.run(repo.get(dialogue.id))
    assert stored is dialogue
    assert len(dialogue.turns) == 2
