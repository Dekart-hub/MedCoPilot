from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone

import pytest

from dialogue import Dialogue, DialogueTurn
from shared.value_objects import Id
from soap.soap import SoapClaim, SoapEvidence, SoapNote
from soap.score.scorer import NliGroundingScorer, _entailment_minus_contradiction


# --- чистая формула, без torch/transformers ------------------------------- #


def test_pure_entailment_scores_near_one():
    assert _entailment_minus_contradiction(p_entailment=0.95, p_contradiction=0.02) == pytest.approx(0.965)


def test_pure_contradiction_scores_near_zero():
    assert _entailment_minus_contradiction(p_entailment=0.02, p_contradiction=0.95) == pytest.approx(0.035)


def test_pure_neutral_scores_near_half():
    assert _entailment_minus_contradiction(p_entailment=0.1, p_contradiction=0.1) == pytest.approx(0.5)


def test_pure_result_is_clamped_to_unit_range():
    assert _entailment_minus_contradiction(p_entailment=1.0, p_contradiction=0.0) == pytest.approx(1.0)
    assert _entailment_minus_contradiction(p_entailment=0.0, p_contradiction=1.0) == pytest.approx(0.0)


# --- скорер целиком, с фейковыми tokenizer/model -------------------------- #
#
# transformers/torch не тянем в юнит-тесты: NliGroundingScorer принимает
# tokenizer/model напрямую через конструктор, поэтому реальная модель не
# загружается вовсе.

_LABEL_ORDER = ["entailment", "neutral", "contradiction"]


def _logits_for(entailment: float, neutral: float, contradiction: float) -> list[float]:
    """Обратный softmax: логиты, которые после softmax дадут заданные вероятности."""
    return [math.log(entailment), math.log(neutral), math.log(contradiction)]


class _FakeModelConfig:
    id2label = dict(enumerate(_LABEL_ORDER))


class _FakeOutput:
    def __init__(self, logits: list[float]) -> None:
        self.logits = [logits]  # logits[0] — то, что читает скорер


class _FakePairRef:
    """Заменяет тензор в inputs: несёт пару (premise, hypothesis) через вызов.

    Claim'ы считаются параллельно (``asyncio.gather`` + потоки), поэтому пара
    не может жить в общем изменяемом поле модели — иначе конкурентные вызовы
    затирают друг друга. Реальные тензоры transformers этой проблемы не имеют:
    пара приходит внутри самого объекта inputs, как и здесь.
    """

    def __init__(self, pair: tuple[str, str]) -> None:
        self.pair = pair

    def to(self, device: str) -> "_FakePairRef":
        return self


class _FakeModel:
    """Возвращает заранее заданные логиты вместо реального инференса."""

    def __init__(self, probs_by_pair: dict[tuple[str, str], dict[str, float]]) -> None:
        self.config = _FakeModelConfig()
        self._probs_by_pair = probs_by_pair

    def to(self, device: str) -> "_FakeModel":
        return self

    def eval(self) -> "_FakeModel":
        return self

    def __call__(self, **inputs):
        pair_ref: _FakePairRef = inputs["pair"]
        probs = self._probs_by_pair[pair_ref.pair]
        logits = _logits_for(probs["entailment"], probs["neutral"], probs["contradiction"])
        return _FakeOutput(logits)


class _FakeTokenizer:
    """Кладёт пару (premise, hypothesis) прямо в возвращаемые "inputs"."""

    def __call__(self, premise: str, hypothesis: str, **kwargs):
        return {"pair": _FakePairRef((premise, hypothesis))}


def _turn(content: str) -> DialogueTurn:
    return DialogueTurn(
        id=Id.new(), role="doctor", content=content, timestamp=datetime.now(timezone.utc)
    )


def _claim(text: str, turn: DialogueTurn) -> SoapClaim:
    return SoapClaim(id=Id.new(), claim=text, evidence=SoapEvidence(text=text, turn_id=turn.id))


def test_scorer_uses_injected_fake_model_without_transformers():
    t_ok = _turn("Patient denies chest pain, no shortness of breath.")
    t_bad = _turn("Patient reports severe chest pain radiating to the arm.")

    probs_by_pair = {
        (t_ok.content, "no chest pain reported"): {
            "entailment": 0.9, "neutral": 0.08, "contradiction": 0.02
        },
        (t_bad.content, "patient denies any chest pain"): {
            "entailment": 0.03, "neutral": 0.07, "contradiction": 0.9
        },
    }
    model = _FakeModel(probs_by_pair)
    tokenizer = _FakeTokenizer()
    scorer = NliGroundingScorer(tokenizer=tokenizer, model=model)

    dialogue = Dialogue(id=Id.new(), turns=[t_ok, t_bad], created_at=datetime.now(timezone.utc))
    note = SoapNote(
        id=Id.new(),
        subjective=_claim("no chest pain reported", t_ok),
        objective=_claim("no chest pain reported", t_ok),
        assessment=_claim("no chest pain reported", t_ok),
        # Галлюцинация-отрицание: реплика говорит о сильной боли, claim это отрицает.
        plan=_claim("patient denies any chest pain", t_bad),
    )

    result = asyncio.run(scorer.score(dialogue, note))

    # 3 секции с entailment (~0.94) + 1 секция с contradiction (~0.065) -> среднее в (0.6, 0.8).
    assert 0.6 < result.score.score < 0.8


def test_scorer_missing_turn_reference_scores_zero_without_model_call():
    model = _FakeModel({})
    tokenizer = _FakeTokenizer()
    scorer = NliGroundingScorer(tokenizer=tokenizer, model=model)

    orphan = _turn("this turn is not part of the dialogue")
    t1 = _turn("Patient reports headache since yesterday.")
    dialogue = Dialogue(id=Id.new(), turns=[t1], created_at=datetime.now(timezone.utc))
    note = SoapNote(
        id=Id.new(),
        subjective=_claim("headache since yesterday", orphan),
        objective=_claim("headache since yesterday", orphan),
        assessment=_claim("headache since yesterday", orphan),
        plan=_claim("headache since yesterday", orphan),
    )

    result = asyncio.run(scorer.score(dialogue, note))

    assert result.score.score == pytest.approx(0.0)
