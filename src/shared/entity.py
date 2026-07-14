"""Base class for domain entities: identity, not attribute values, defines them.

Two entities of the same type are equal when their ``id`` matches, regardless of
their other fields. ``TId`` is the identifier type (e.g. ``DialogueId``).

Subclasses declare themselves as ``@dataclass(eq=False, slots=True)``: the
dataclass supplies fields, ``__init__`` and ``__repr__`` while equality and
hashing come from here (``eq=False`` keeps the dataclass from overriding them).
"""

from __future__ import annotations


class Entity[TId]:
    """Domain entity identified by ``id`` rather than by its attribute values."""

    __slots__ = ()
    id: TId

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return type(self) is type(other) and self.id == other.id

    def __hash__(self) -> int:
        return hash((type(self), self.id))
