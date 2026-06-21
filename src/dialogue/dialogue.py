from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

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
