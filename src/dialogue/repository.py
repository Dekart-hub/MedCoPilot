from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .dialogue import Dialogue, DialogueId


class DialogueRepository(ABC):
    """Хранилище диалогов."""

    @abstractmethod
    async def save(self, dialogue: Dialogue) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get(self, dialogue_id: DialogueId) -> Dialogue | None:
        raise NotImplementedError

    @abstractmethod
    async def list_all(self) -> list[Dialogue]:
        raise NotImplementedError


class InMemoryDialogueRepository(DialogueRepository):
    """Реализация в памяти — для разработки и тестов."""

    def __init__(self, initial: Iterable[Dialogue] | None = None) -> None:
        self._items: dict[DialogueId, Dialogue] = {
            dialogue.id: dialogue for dialogue in (initial or ())
        }

    async def save(self, dialogue: Dialogue) -> None:
        self._items[dialogue.id] = dialogue

    async def get(self, dialogue_id: DialogueId) -> Dialogue | None:
        return self._items.get(dialogue_id)

    async def list_all(self) -> list[Dialogue]:
        return list(self._items.values())
