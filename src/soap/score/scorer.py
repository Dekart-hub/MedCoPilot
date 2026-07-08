from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter

from dialogue import Dialogue, DialogueTurn, DialogueTurnId

from shared.value_objects import FloatRangedScore, Id
from ..soap import SoapClaim, SoapNote
from .score import ClaimConfidenceScore, SoapNoteConfidenceScore


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
    """Lexical Tier 1 baseline: clipped unigram precision per claim.

    For each claim the cited quote is compared against the dialogue turn it
    references. A claim scoring below ``review_threshold`` is flagged for
    human review; the note-level score is the mean over the non-empty
    sections. Empty sections are skipped here (no score, no flag) and are
    surfaced by the Tier 0 gate alone, matching its empty-section rule.
    """

    def __init__(self, review_threshold: float = 0.6) -> None:
        self._review_threshold = review_threshold

    async def score(
        self, dialogue: Dialogue, soap_note: SoapNote
    ) -> SoapNoteConfidenceScore:
        turns_by_id: dict[DialogueTurnId, DialogueTurn] = {
            turn.id: turn for turn in dialogue.turns
        }

        claim_scores: list[ClaimConfidenceScore] = []
        for section, claim in soap_note.sections():
            # Skip empty sections to match the Tier 0 gate rule: they get no
            # score and no flag, and are surfaced by Tier 0's empty_sections.
            if not claim.claim.strip():
                continue
            value = self._score_claim(claim, turns_by_id)
            claim_scores.append(
                ClaimConfidenceScore(
                    claim_id=claim.id,
                    section=section,
                    score=FloatRangedScore(value),
                    is_flagged=value < self._review_threshold,
                )
            )

        # Guard the degenerate all-empty note against division by zero.
        mean_score = (
            sum(cs.score.score for cs in claim_scores) / len(claim_scores)
            if claim_scores
            else 0.0
        )
        return SoapNoteConfidenceScore(
            id=Id.new(),
            score=FloatRangedScore(mean_score),
            soap_note_id=soap_note.id,
            claim_scores=claim_scores,
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
