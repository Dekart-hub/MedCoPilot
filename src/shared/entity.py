from __future__ import annotations


class Entity[TId]:
    """Базовый класс сущности: идентичность по ``id``, а не по значению полей.

    Две сущности одного типа равны тогда и только тогда, когда совпадают их
    ``id``. ``TId`` — тип идентификатора (например, ``DialogueId``).

    Наследники объявляются как ``@dataclass(eq=False, slots=True)`` —
    dataclass отвечает за поля/``__init__``/``__repr__``, а равенство и хэш
    берутся отсюда (``eq=False`` не даёт dataclass их перетереть).
    """

    __slots__ = ()
    id: TId

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return type(self) is type(other) and self.id == other.id

    def __hash__(self) -> int:
        return hash((self.__class__, self.id))
