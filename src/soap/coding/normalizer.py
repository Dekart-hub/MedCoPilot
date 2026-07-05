from __future__ import annotations

from abc import ABC, abstractmethod

from shared.value_objects import FloatRangedScore, Id

from ..soap import SoapNote
from .coding import DiagnosisCoding, SoapNoteCoding
from .retrieval import MkbIndex


class DiagnosisNormalizer(ABC):
    """Порт нормализации диагноза в код классификатора.

    Зеркало ``ConfidenceScorer``: принимает ноту целиком (assessment — объект
    кодирования, S/O — контекст для будущей дизамбигуации) и возвращает
    side-car ``SoapNoteCoding``, привязанный к assessment-клейму. Доменную
    модель SOAP не мутирует.
    """

    @abstractmethod
    async def normalize(self, soap_note: SoapNote) -> SoapNoteCoding:
        raise NotImplementedError


class LexicalDiagnosisNormalizer(DiagnosisNormalizer):
    """Tier 1: лексический матчинг (BM25) текста ассессмента к Тому 3.

    Контекст S/O пока не используется — дизамбигуация и back-off к родителю
    появятся на следующих tier'ах (гибридный ретрив + LLM-реранк). Сейчас
    отдаём ранжированный список кодов-кандидатов как есть.
    """

    def __init__(self, index: MkbIndex, top_n: int = 5) -> None:
        self._index = index
        self._top_n = top_n

    async def normalize(self, soap_note: SoapNote) -> SoapNoteCoding:
        assessment = soap_note.assessment
        raw_candidates = self._index.search(assessment.claim, self._top_n)
        classifier = self._index.classifier
        candidates = [
            DiagnosisCoding(
                code=rc.code,
                title=self._index.name_of(rc.code) or rc.formulation,
                matched_formulation=rc.formulation,
                score=FloatRangedScore(min(1.0, max(0.0, rc.score))),
                classifier=classifier,
            )
            for rc in raw_candidates
        ]
        return SoapNoteCoding(
            id=Id.new(),
            soap_claim_id=assessment.id,
            candidates=candidates,
        )


class NullDiagnosisNormalizer(DiagnosisNormalizer):
    """Заглушка: всегда пустое кодирование.

    Нужна, чтобы контракт (use-case, view, API) работал, пока реальный
    индекс МКБ не собран/не докачан. Подменяется на ``LexicalDiagnosisNormalizer``
    в DI, когда данные готовы.
    """

    async def normalize(self, soap_note: SoapNote) -> SoapNoteCoding:
        return SoapNoteCoding(
            id=Id.new(),
            soap_claim_id=soap_note.assessment.id,
            candidates=[],
        )
