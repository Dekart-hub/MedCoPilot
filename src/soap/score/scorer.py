from __future__ import annotations

import asyncio
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

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


def _softmax(logits: list[float]) -> list[float]:
    """Softmax чистым Python — без зависимости от torch на этом шаге."""
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _entailment_minus_contradiction(p_entailment: float, p_contradiction: float) -> float:
    """Сворачивает NLI-вероятности в один score в [0, 1].

    ``P(entailment) - P(contradiction)`` лежит в [-1, 1]: 1 — цитата чисто
    подтверждается источником, -1 — источник ей прямо противоречит, 0 —
    источник нейтрален (ни подтверждает, ни опровергает). В отличие от cosine
    similarity эмбеддингов эта разница различает перефразировку (высокий
    entailment) от отрицания (высокий contradiction) — два текста вроде
    "жалуется на боль в груди" и "отрицает боль в груди" лексически и
    тематически близки, поэтому cosine их не разведёт, а NLI-модель обучена
    именно на этом различии.

    Результат линейно сдвигается в [0, 1], чтобы совпадать по шкале с
    остальными ``ConfidenceScorer``.
    """
    return max(0.0, min(1.0, (p_entailment - p_contradiction + 1.0) / 2.0))


class NliGroundingScorer(ConfidenceScorer):
    """Tier 2 (мультиязычный) скорер уверенности: NLI entailment вместо лексики.

    Для каждого claim берётся пара (реплика-источник = premise, цитата
    claim'а = hypothesis) и прогоняется через NLI-модель, предсказывающую
    P(entailment) / P(neutral) / P(contradiction). Итоговый score секции —
    :func:`_entailment_minus_contradiction`; score ноты — среднее по четырём
    секциям (S / O / A / P), как в :class:`LexicalGroundingScorer`.

    Модель по умолчанию — ``MoritzLaurer/mDeBERTa-v3-base-mnli-xnli``:
    дообучена на MNLI + XNLI, то есть одна и та же модель одинаково работает
    и на английском (бенчмарки), и на русском (прод) без переключения по
    конфигу. ``tokenizer``/``model`` можно передать напрямую (например, в
    тестах — фейком), тогда ``transformers`` не импортируется вовсе.

    ИЗВЕСТНОЕ ОГРАНИЧЕНИЕ: на ручном прогоне (``scripts/smoke_nli_scorer.py``)
    модель не ловит явное русское отрицание («головную боль отрицает» против
    claim «у пациента головная боль» дало score ~0.999 вместо низкого) — на
    английском тот же паттерн отрицания отрабатывает корректно. Пробовали
    ``cointegrated/rubert-base-cased-nli-threeway`` (русскоязычная, дообучена
    на NLI): на этом кейсе лучше (~0.589), но всё ещё выше 0.5 и не решает
    задачу полностью, плюс хуже на английском (не мультиязычная). Тонкие
    подмены деталей («левое» vs «правое» колено) обе модели ловят хорошо.
    Практический вывод: на пограничных score (~0.3-0.7) полагаться только на
    NLI не стоит — нужна эскалация на LLM-as-judge (Tier 3), см. паттерн
    отката в ``LlmRerankedDiagnosisNormalizer``.
    """

    _DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if tokenizer is None or model is None:
            # Ленивый импорт: transformers/torch — тяжёлые зависимости, нужны
            # только тем, кто реально использует этот (Tier 2) скорер.
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
            model = model or AutoModelForSequenceClassification.from_pretrained(
                model_name
            )

        self._tokenizer = tokenizer
        self._model = model.to(device).eval()
        self._device = device
        # Индексы классов узнаём из конфига модели, а не хардкодим — порядок
        # entailment/neutral/contradiction может отличаться между чекпоинтами.
        self._label_index: dict[str, int] = {
            label.lower(): idx for idx, label in self._model.config.id2label.items()
        }

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

        per_claim = await asyncio.gather(
            *(self._score_claim(claim, turns_by_id) for claim in claims)
        )
        mean_score = sum(per_claim) / len(per_claim)

        return SoapNoteConfidenceScore(
            id=Id.new(),
            score=FloatRangedScore(mean_score),
            soap_note_id=soap_note.id,
        )

    async def _score_claim(
        self, claim: SoapClaim, turns_by_id: dict[DialogueTurnId, DialogueTurn]
    ) -> float:
        source_turn = turns_by_id.get(claim.evidence.turn_id)
        if source_turn is None:
            # Цитата ссылается на несуществующую реплику — нет опоры в диалоге.
            return 0.0
        # Инференс — CPU-bound синхронный вызов; уводим в отдельный поток,
        # чтобы не блокировать event loop и чтобы claim'ы считались параллельно.
        return await asyncio.to_thread(
            self._entailment_score, source_turn.content, claim.evidence.text
        )

    def _entailment_score(self, premise: str, hypothesis: str) -> float:
        inputs = self._tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        logits = self._forward(inputs)
        probs = _softmax(logits)

        p_entailment = probs[self._label_index["entailment"]]
        p_contradiction = probs[self._label_index["contradiction"]]
        return _entailment_minus_contradiction(p_entailment, p_contradiction)

    def _forward(self, inputs: dict[str, Any]) -> list[float]:
        """Прогоняет модель, возвращая сырые логиты (3 класса) списком.

        ``torch.no_grad()`` применяется, только если torch реально доступен —
        в юнит-тестах модель подменяется фейком без torch, и градиенты там
        просто не считаются в принципе.
        """
        try:
            import torch
        except ImportError:
            logits = self._model(**inputs).logits[0]
        else:
            with torch.no_grad():
                logits = self._model(**inputs).logits[0]
        return logits.tolist() if hasattr(logits, "tolist") else list(logits)
