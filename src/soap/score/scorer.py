from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter

from dialogue import Dialogue, DialogueTurn, DialogueTurnId

from shared.value_objects import FloatRangedScore, Id
from ..soap import SoapClaim, SoapNote
from .score import SoapNoteConfidenceScore


class ConfidenceScorer(ABC):
    @abstractmethod
    async def score(
        self, dialogue: Dialogue, soap_note: SoapNote
    ) -> SoapNoteConfidenceScore:
        raise NotImplementedError


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Простейшая токенизация: слова в нижнем регистре, без пунктуации."""
    return _TOKEN_RE.findall(text.lower())


def _clipped_unigram_precision(evidence: str, source: str) -> float:
    """Доля токенов цитаты, реально присутствующих в реплике-источнике.

    По сути BLEU-1 с клиппингом, но без brevity penalty: для коротких
    медицинских фраз полный BLEU-4 почти всегда вырождается в 0, поэтому
    как базовый сигнал groundedness он бесполезен. Здесь же мы измеряем,
    какая часть процитированного текста подтверждается репликой диалога,
    на которую ссылается claim.

    Возвращает значение в диапазоне [0, 1].
    """
    evidence_tokens = _tokenize(evidence)
    if not evidence_tokens:
        return 0.0

    evidence_counts = Counter(evidence_tokens)
    source_counts = Counter(_tokenize(source))

    matched = sum(
        min(count, source_counts[token]) for token, count in evidence_counts.items()
    )
    return matched / len(evidence_tokens)


class LexicalGroundingScorer(ConfidenceScorer):
    """Простейший baseline-скорер уверенности (Tier 1, лексический).

    Для каждого claim в SOAP-ноте берётся его цитата (``evidence.text``) и
    реплика диалога, на которую он ссылается (``evidence.turn_id``), после
    чего считается clipped unigram precision цитаты относительно этой реплики.
    Итоговый score ноты — среднее по четырём секциям (S / O / A / P).

    Это не заменяет NLI/LLM-проверку, а служит дешёвым детерминированным
    сигналом: «насколько процитированный текст вообще опирается на сказанное».
    """

    async def score(
        self, dialogue: Dialogue, soap_note: SoapNote
    ) -> SoapNoteConfidenceScore:
        turns_by_id: dict[DialogueTurnId, DialogueTurn] = {
            turn.id: turn for turn in dialogue.turns
        }

        claims: list[SoapClaim] = [
            soap_note.subjective,
            soap_note.objective,
            soap_note.assessment,
            soap_note.plan,
        ]

        per_claim = [self._score_claim(claim, turns_by_id) for claim in claims]
        mean_score = sum(per_claim) / len(per_claim)

        return SoapNoteConfidenceScore(
            id=Id.new(),
            score=FloatRangedScore(mean_score),
            soap_note_id=soap_note.id,
        )

    @staticmethod
    def _score_claim(
        claim: SoapClaim, turns_by_id: dict[DialogueTurnId, DialogueTurn]
    ) -> float:
        source_turn = turns_by_id.get(claim.evidence.turn_id)
        if source_turn is None:
            # Цитата ссылается на несуществующую реплику — нет опоры в диалоге.
            return 0.0
        return _clipped_unigram_precision(claim.evidence.text, source_turn.content)
