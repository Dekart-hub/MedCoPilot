"""Composition helpers wiring the dialogue use cases to a database session.

Framework-agnostic for now (the REST API arrives in T13); each provider takes
an :class:`AsyncSession` and returns a ready-to-use collaborator.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from .repository import DialogueRepository
from .sqlalchemy_repository import SqlAlchemyDialogueRepository
from .use_cases import AddDialogue


def get_dialogue_repository(session: AsyncSession) -> DialogueRepository:
    return SqlAlchemyDialogueRepository(session)


def get_add_dialogue(session: AsyncSession) -> AddDialogue:
    return AddDialogue(get_dialogue_repository(session))
