"""The Dialogue aggregate: a raw, multi-party conversation of ordered turns.

Pure domain — no persistence concerns. A dialogue holds its turns in speaking
order; each turn carries its own identity so later work (SOAP claims in T6) can
reference the exact turn a statement came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.entity import Entity
from shared.value_objects import Id

type DialogueTurnId = Id[DialogueTurn]
type DialogueId = Id[Dialogue]


@dataclass(eq=False, slots=True)
class DialogueTurn(Entity[DialogueTurnId]):
    """A single utterance: who spoke (free-form label) and what was said."""

    id: DialogueTurnId
    speaker: str
    text: str


@dataclass(eq=False, slots=True)
class Dialogue(Entity[DialogueId]):
    """Aggregate root owning an ordered list of :class:`DialogueTurn`."""

    id: DialogueId
    turns: list[DialogueTurn] = field(default_factory=list)

    @classmethod
    def start(cls) -> Dialogue:
        """Begin a new, empty dialogue with a freshly generated identity."""
        return cls(id=Id.new())

    def add_turn(self, speaker: str, text: str) -> DialogueTurn:
        """Append a new turn to the end of the dialogue and return it."""
        turn = DialogueTurn(id=Id.new(), speaker=speaker, text=text)
        self.turns.append(turn)
        return turn
