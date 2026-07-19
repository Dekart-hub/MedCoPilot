"""JSON serialization for the Dialogue aggregate.

A boundary concern kept out of the domain: turning a :class:`Dialogue` into a
JSON-compatible ``dict``. Turn identities are rendered as strings and the turns
are emitted in speaking order so the result is directly ``json.dumps``-able.
"""

from __future__ import annotations

from typing import Any

from .dialogue import Dialogue, DialogueTurn


def dialogue_to_dict(dialogue: Dialogue) -> dict[str, Any]:
    """Serialize a dialogue and its ordered turns to a JSON-compatible dict."""
    return {
        "id": str(dialogue.id),
        "turns": [_turn_to_dict(turn) for turn in dialogue.turns],
    }


def _turn_to_dict(turn: DialogueTurn) -> dict[str, str]:
    return {"id": str(turn.id), "speaker": turn.speaker, "text": turn.text}
