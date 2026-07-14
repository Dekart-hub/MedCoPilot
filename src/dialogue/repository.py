"""Persistence port for the Dialogue aggregate.

The domain depends on this abstraction; concrete adapters (SQLAlchemy, fakes)
implement it. Keeping the port here inverts the dependency so the domain never
reaches out to infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .dialogue import Dialogue, DialogueId


class DialogueRepository(ABC):
    """Stores and retrieves :class:`~dialogue.dialogue.Dialogue` aggregates."""

    @abstractmethod
    async def save(self, dialogue: Dialogue) -> None:
        """Persist the aggregate and its turns."""

    @abstractmethod
    async def get(self, dialogue_id: DialogueId) -> Dialogue | None:
        """Return the aggregate by id, or ``None`` if it does not exist."""
