from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, NewType
from uuid import UUID, uuid4


# ============================================
# Семантические типы для ID
# ============================================

AnnotatorId = NewType('AnnotatorId', str)
ClaimId = NewType('ClaimId', str)


# ============================================
# Value Objects
# ============================================

@dataclass(frozen=True, slots=True)
class FloatRangedScore:

    MIN_SCORE: ClassVar[float] = 0.0
    MAX_SCORE: ClassVar[float] = 1.0
    score: float

    def __post_init__(self) -> None:
        if self.score < self.MIN_SCORE or self.score > self.MAX_SCORE:
            raise ValueError(
                f"Score must be between {self.MIN_SCORE} and {self.MAX_SCORE}"
            )


@dataclass(frozen=True, slots=True)
class Id[T]:
    """Типизированный идентификатор."""

    value: UUID

    @classmethod
    def new(cls) -> Id[T]:
        return cls(uuid4())

    @classmethod
    def from_str(cls, raw: str) -> Id[T]:
        return cls(UUID(raw))

    def __str__(self) -> str:
        return str(self.value)