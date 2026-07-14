from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from shared.entity import Entity
from shared.value_objects import Id

type DialogueTurnId = Id[DialogueTurn]
type DialogueId = Id[Dialogue]


@dataclass(eq=False, slots=True)
class DialogueTurn(Entity[DialogueTurnId]):
    id: DialogueTurnId
    role: str
    content: str
    timestamp: datetime


@dataclass(eq=False, slots=True)
class Dialogue(Entity[DialogueId]):
    id: DialogueId
    turns: list[DialogueTurn]
    created_at: datetime
    patient_ref: str | None = None
    encounter_ref: str | None = None

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        patient_ref: str | None = None,
        encounter_ref: str | None = None,
    ) -> Dialogue:
        """Собирает диалог из «сырого» текста: одна строка — одна реплика.

        Метка участника — первое слово строки (например, ``person`` / ``medic``),
        остальное — содержимое реплики. Пустые строки пропускаются.

        Заготовка: позже парсинг ролей/таймстампов наверняка усложнится.
        """
        now = datetime.now(timezone.utc)
        turns: list[DialogueTurn] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            role, _, content = stripped.partition(" ")
            turns.append(
                DialogueTurn(
                    id=Id.new(),
                    role=role.rstrip(":"),
                    content=content.strip(),
                    timestamp=now,
                )
            )
        return cls(
            id=Id.new(),
            turns=turns,
            created_at=now,
            patient_ref=patient_ref,
            encounter_ref=encounter_ref,
        )
