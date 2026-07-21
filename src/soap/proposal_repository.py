"""Persistence port for the LLM editor-session aggregate.

The domain depends on this abstraction; concrete adapters (SQLAlchemy, fakes)
implement it. A session is keyed to the correction it edits — at most one per
correction — so ``get_for_correction`` backs "resume this correction's LLM
editing history", the only history scope there is (never across reports).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .correction import CorrectionId
from .proposal import CorrectionEditorSession, SessionId


class CorrectionEditorSessionRepository(ABC):
    """Stores and retrieves :class:`~soap.proposal.CorrectionEditorSession` aggregates."""

    @abstractmethod
    async def save(self, session: CorrectionEditorSession) -> None:
        """Persist the session with its full proposal, operation and decision history."""

    @abstractmethod
    async def get(self, session_id: SessionId) -> CorrectionEditorSession | None:
        """Return the session by id, or ``None`` if it does not exist."""

    @abstractmethod
    async def get_for_correction(
        self, correction_id: CorrectionId
    ) -> CorrectionEditorSession | None:
        """Return the editing session of ``correction_id``, or ``None``."""
