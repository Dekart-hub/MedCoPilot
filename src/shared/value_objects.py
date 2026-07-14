"""Reusable domain value objects."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class Id[T]:
    """Typed identifier wrapping a UUID.

    ``T`` is a phantom type parameter: unused at runtime, it lets the type
    checker tell ``Id[Dialogue]`` apart from ``Id[DialogueTurn]``.
    """

    value: UUID

    @classmethod
    def new(cls) -> Id[T]:
        return cls(uuid4())

    @classmethod
    def from_str(cls, raw: str) -> Id[T]:
        return cls(UUID(raw))

    def __str__(self) -> str:
        return str(self.value)
