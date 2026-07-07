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


def test_claim_scores_cover_all_sections_in_order(dialogue_and_turns):
    dialogue, t = dialogue_and_turns
    note = SoapNote(
        id=Id.new(),
        subjective=_claim("sharp headache", t["s"]),
        objective=_claim("blood pressure 130 over 85", t["o"]),
        assessment=_claim("tension headache", t["a"]),
        plan=_claim("take ibuprofen", t["p"]),
    )
    result = _score(dialogue, note)

    assert [cs.section for cs in result.claim_scores] == [
        "subjective",
        "objective",
        "assessment",
        "plan",
    ]
    assert [cs.claim_id for cs in result.claim_scores] == [
        note.subjective.id,
        note.objective.id,
        note.assessment.id,
        note.plan.id,
    ]


def test_ungrounded_claim_is_flagged(dialogue_and_turns):
    dialogue, t = dialogue_and_turns
    fabricated = SoapClaim(
        id=Id.new(),
        claim="fever of 39",
        evidence=SoapEvidence(text="temperature is 39 degrees", turn_id=t["o"].id),
    )
    note = SoapNote(
        id=Id.new(),
        subjective=_claim("sharp headache", t["s"]),
        objective=fabricated,
        assessment=_claim("tension headache", t["a"]),
        plan=_claim("take ibuprofen", t["p"]),
    )
    result = _score(dialogue, note)
    by_id = {cs.claim_id: cs for cs in result.claim_scores}

    assert by_id[fabricated.id].is_flagged is True
    assert by_id[note.subjective.id].is_flagged is False


def test_review_threshold_is_configurable(dialogue_and_turns):
    dialogue, t = dialogue_and_turns
    note = SoapNote(
        id=Id.new(),
        subjective=_claim("sharp headache since yesterday morning", t["s"]),
        objective=_claim("blood pressure 130 over 85", t["o"]),
        assessment=_claim("tension headache", t["a"]),
        plan=_claim("take ibuprofen and rest", t["p"]),
    )
    strict = asyncio.run(
        LexicalGroundingScorer(review_threshold=1.0).score(dialogue, note)
    )
    lax = asyncio.run(
        LexicalGroundingScorer(review_threshold=0.0).score(dialogue, note)
    )

    # Fully grounded quotes score exactly 1.0: `score < threshold` flags
    # nothing at threshold 1.0 and nothing at 0.0 either.
    assert all(not cs.is_flagged for cs in strict.claim_scores)
    assert all(not cs.is_flagged for cs in lax.claim_scores)

    partially = SoapNote(
        id=Id.new(),
        subjective=SoapClaim(
            id=Id.new(),
            claim="headache",
            evidence=SoapEvidence(
                text="sharp headache with nausea and vomiting", turn_id=t["s"].id
            ),
        ),
        objective=_claim("blood pressure 130 over 85", t["o"]),
        assessment=_claim("tension headache", t["a"]),
        plan=_claim("take ibuprofen and rest", t["p"]),
    )
    strict_partial = asyncio.run(
        LexicalGroundingScorer(review_threshold=1.0).score(dialogue, partially)
    )
    assert strict_partial.claim_scores[0].is_flagged is True


def test_note_score_is_mean_of_claim_scores(dialogue_and_turns):
    dialogue, t = dialogue_and_turns
    note = SoapNote(
        id=Id.new(),
        subjective=_claim("sharp headache", t["s"]),
        objective=_claim("blood pressure 130 over 85", t["o"]),
        assessment=_claim("tension headache", t["a"]),
        plan=_claim("take ibuprofen", t["p"]),
    )
    result = _score(dialogue, note)
    mean = sum(cs.score.score for cs in result.claim_scores) / 4

    assert result.score.score == pytest.approx(mean)
