"""Integration test: a dialogue survives a save/get round-trip through Postgres.

Requires a reachable database via ``DATABASE_URL`` (skipped otherwise, as in
``tests/test_db.py``). Migrations run first, then the aggregate is saved in one
session and loaded in a fresh one to prove it truly hits the database.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from dialogue.dialogue import Dialogue
from dialogue.sqlalchemy_repository import SqlAlchemyDialogueRepository
from infra.db import dispose_engine, get_sessionmaker
from infra.migrations import run_migrations

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not configured; skipping DB integration test",
)


async def _save_then_get(dialogue: Dialogue) -> Dialogue | None:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        await SqlAlchemyDialogueRepository(session).save(dialogue)
        await session.commit()
    async with sessionmaker() as session:
        loaded = await SqlAlchemyDialogueRepository(session).get(dialogue.id)
    await dispose_engine()
    return loaded


def test_dialogue_round_trips_through_the_database() -> None:
    run_migrations()
    dialogue = Dialogue.start()
    dialogue.add_turn("doctor", "What brings you in today?")
    dialogue.add_turn("patient", "A headache for three days.")

    loaded = asyncio.run(_save_then_get(dialogue))

    assert loaded == dialogue
    assert loaded is not None
    assert [turn.id for turn in loaded.turns] == [turn.id for turn in dialogue.turns]
    assert [(turn.speaker, turn.text) for turn in loaded.turns] == [
        ("doctor", "What brings you in today?"),
        ("patient", "A headache for three days."),
    ]
