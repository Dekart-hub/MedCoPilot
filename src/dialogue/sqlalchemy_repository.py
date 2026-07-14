"""SQLAlchemy adapter for :class:`DialogueRepository`.

Maps the pure domain aggregate onto the ORM rows in :mod:`dialogue.orm` and
back. Transaction control (commit/rollback) is left to the caller so several
operations can share one unit of work.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from shared.value_objects import Id

from .dialogue import Dialogue, DialogueId, DialogueTurn
from .orm import DialogueRow, DialogueTurnRow
from .repository import DialogueRepository


class SqlAlchemyDialogueRepository(DialogueRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, dialogue: Dialogue) -> None:
        await self._session.merge(_to_row(dialogue))

    async def get(self, dialogue_id: DialogueId) -> Dialogue | None:
        statement = (
            select(DialogueRow)
            .where(DialogueRow.id == dialogue_id.value)
            .options(selectinload(DialogueRow.turns))
        )
        row = (await self._session.execute(statement)).scalar_one_or_none()
        return _to_domain(row) if row is not None else None


def _to_row(dialogue: Dialogue) -> DialogueRow:
    return DialogueRow(
        id=dialogue.id.value,
        turns=[
            DialogueTurnRow(
                id=turn.id.value,
                position=position,
                speaker=turn.speaker,
                text=turn.text,
            )
            for position, turn in enumerate(dialogue.turns)
        ],
    )


def _to_domain(row: DialogueRow) -> Dialogue:
    return Dialogue(
        id=Id(row.id),
        turns=[
            DialogueTurn(id=Id(turn.id), speaker=turn.speaker, text=turn.text) for turn in row.turns
        ],
    )
