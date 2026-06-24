from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from dialogue import Dialogue, DialogueTurn
from shared.value_objects import Id
from soap.soap import SoapClaim, SoapEvidence, SoapNote
from soap.score.scorer import LexicalGroundingScorer


def _turn(role: str, content: str) -> DialogueTurn:
    return DialogueTurn(
        id=Id.new(),
        role=role,
        content=content,
        timestamp=datetime.now(timezone.utc),
    )


def _claim(text: str, turn: DialogueTurn) -> SoapClaim:
    """Claim, цитата которого ссылается на конкретную реплику диалога."""
    return SoapClaim(
        id=Id.new(),
        claim=text,
        evidence=SoapEvidence(text=text, turn_id=turn.id),
    )


def _score(dialogue: Dialogue, note: SoapNote):
    """Синхронная обёртка: запускает async-скорер без pytest-asyncio."""
    return asyncio.run(LexicalGroundingScorer().score(dialogue, note))


@pytest.fixture
def dialogue_and_turns() -> tuple[Dialogue, dict[str, DialogueTurn]]:
    turns = {
        "s": _turn("patient", "I have a sharp headache since yesterday morning"),
        "o": _turn("doctor", "Blood pressure is 130 over 85, temperature normal"),
        "a": _turn("doctor", "Looks like a tension headache"),
        "p": _turn("doctor", "Take ibuprofen and rest, come back in a week"),
    }
    dialogue = Dialogue(
        id=Id.new(),
        turns=list(turns.values()),
        created_at=datetime.now(timezone.utc),
    )
    return dialogue, turns


def test_fully_grounded_note_scores_one(dialogue_and_turns):
    dialogue, t = dialogue_and_turns
    note = SoapNote(
        id=Id.new(),
        subjective=_claim("sharp headache since yesterday morning", t["s"]),
        objective=_claim("blood pressure 130 over 85", t["o"]),
        assessment=_claim("tension headache", t["a"]),
        plan=_claim("take ibuprofen and rest", t["p"]),
    )

    result = _score(dialogue, note)

    assert result.score.score == pytest.approx(1.0)
    assert result.soap_note_id == note.id


def test_hallucinated_plan_lowers_score(dialogue_and_turns):
    dialogue, t = dialogue_and_turns
    # Plan ссылается на реплику про ибупрофен, но цитирует выдуманные антибиотики.
    note = SoapNote(
        id=Id.new(),
        subjective=_claim("sharp headache since yesterday morning", t["s"]),
        objective=_claim("blood pressure 130 over 85", t["o"]),
        assessment=_claim("tension headache", t["a"]),
        plan=_claim("prescribe antibiotics for infection", t["p"]),
    )

    result = _score(dialogue, note)

    # Три секции идеальны, Plan = 0 -> (1 + 1 + 1 + 0) / 4.
    assert result.score.score == pytest.approx(0.75)


def test_partial_overlap_is_fraction(dialogue_and_turns):
    dialogue, t = dialogue_and_turns
    # "sharp" и "headache" есть в реплике, "migraine" и "evening" — нет: 2 из 4.
    note = SoapNote(
        id=Id.new(),
        subjective=_claim("sharp headache migraine evening", t["s"]),
        objective=_claim("blood pressure 130 over 85", t["o"]),
        assessment=_claim("tension headache", t["a"]),
        plan=_claim("take ibuprofen and rest", t["p"]),
    )

    result = _score(dialogue, note)

    # Subjective = 0.5, остальные = 1.0 -> (0.5 + 1 + 1 + 1) / 4 = 0.875.
    assert result.score.score == pytest.approx(0.875)


def test_missing_turn_reference_scores_zero(dialogue_and_turns):
    dialogue, t = dialogue_and_turns
    orphan_turn = _turn("doctor", "this turn is not part of the dialogue")
    note = SoapNote(
        id=Id.new(),
        subjective=_claim("sharp headache since yesterday morning", t["s"]),
        objective=_claim("blood pressure 130 over 85", t["o"]),
        assessment=_claim("tension headache", t["a"]),
        # Ссылка на реплику, которой нет в диалоге -> вклад 0.
        plan=_claim("take ibuprofen and rest", orphan_turn),
    )

    result = _score(dialogue, note)

    assert result.score.score == pytest.approx(0.75)
