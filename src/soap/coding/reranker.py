"""Tier 2: LLM-реранк кандидатов кодирования.

Ретрив (Tier 1) отвечает за recall — затащить правильный код в топ-N.
Здесь LLM делает то, что BM25 не умеет: читает assessment вместе с S/O
(with/without, острота, локализация) и выбирает один код из предъявленных —
либо честно отказывается. Окрестность кандидата (родители + дети из Тома 1)
даёт модели уточнить 4-5-й знак или откатиться к рубрике.

Галлюцинации кодов отсекаются структурно: допустимое множество — только коды,
показанные в промпте (кандидаты + их окрестность). Невалидный ответ или отказ
LLM деградируют до лексического top-1 — хуже Tier 1 стать не можем.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError

from shared.prompts import PromptStore
from shared.value_objects import FloatRangedScore

from ..soap import SoapNote
from .coding import DiagnosisCoding, SoapNoteCoding
from .normalizer import DiagnosisNormalizer, LexicalDiagnosisNormalizer
from .retrieval import MkbIndex

logger = logging.getLogger(__name__)

RERANK_PROMPT_KEY = "coding.rerank"

DEFAULT_CODING_PROMPTS: dict[str, str] = {
    RERANK_PROMPT_KEY: (
        "Ты — медицинский кодировщик. Подбери код классификатора "
        "({{ classifier }}) для диагноза из SOAP-ноты.\n\n"
        "Диагноз (Assessment): {{ assessment }}\n"
        "Жалобы (Subjective): {{ subjective }}\n"
        "Осмотр (Objective): {{ objective }}\n\n"
        "Кандидаты (найдены лексическим поиском; у каждого может быть указана "
        "окрестность — родительские рубрики и уточняющие подкоды):\n"
        "{{ candidates }}\n\n"
        "Правила:\n"
        "1. Выбирай код ТОЛЬКО из показанных выше (кандидат, его родитель или "
        "подкод). Другие коды запрещены.\n"
        "2. Уточняй знаки по Subjective/Objective (with/without, острота, "
        "локализация). Если деталей нет — бери менее специфичный код "
        "(unspecified/родительскую рубрику), а не выдумывай уточнение.\n"
        "3. Если ни один код не подходит диагнозу по смыслу, верни code = null.\n"
        "4. rationale — одно-два предложения: почему этот код и что "
        "дизамбигуировало."
    ),
}


class RerankOut(BaseModel):
    """Структурированный ответ LLM-реранкера."""

    code: str | None = Field(
        description="Выбранный код из показанного списка, либо null при отказе"
    )
    rationale: str = Field(description="Краткое обоснование выбора")
    confidence: float = Field(ge=0.0, le=1.0, description="Уверенность в выборе")


class LlmRerankedDiagnosisNormalizer(DiagnosisNormalizer):
    """Tier 2 поверх Tier 1: лексический ретрив + LLM-выбор кода.

    Кандидаты ретрива сохраняются в ``candidates`` (аудит), выбор LLM — в
    ``selected``/``rationale``. При любом сбое (LLM недоступна, невалидный код,
    отказ) результат не хуже Tier 1: ``selected`` остаётся None, ``best``
    отдаёт лексический top-1.
    """

    def __init__(
        self,
        index: MkbIndex,
        model: BaseChatModel,
        prompts: PromptStore,
        top_n: int = 20,
    ) -> None:
        self._index = index
        self._lexical = LexicalDiagnosisNormalizer(index, top_n)
        self._llm = model.with_structured_output(RerankOut)
        self._prompts = prompts

    async def normalize(self, soap_note: SoapNote) -> SoapNoteCoding:
        coding = await self._lexical.normalize(soap_note)
        if not coding.candidates:
            return coding

        allowed = self._allowed_codes(coding.candidates)
        try:
            result = await self._rerank(soap_note, coding.candidates)
        except Exception:
            logger.exception("LLM-реранк упал — остаёмся на лексическом top-1")
            return coding

        if result.code is None:
            coding.rationale = result.rationale
            return coding
        if result.code not in allowed:
            logger.warning(
                "LLM выбрала код вне допустимого множества (%s) — игнорируем",
                result.code,
            )
            return coding

        coding.selected = self._to_coding(result)
        coding.rationale = result.rationale
        return coding

    # --- внутренности ------------------------------------------------------ #

    async def _rerank(
        self, soap_note: SoapNote, candidates: list[DiagnosisCoding]
    ) -> RerankOut:
        prompt = await self._prompts.get(
            RERANK_PROMPT_KEY,
            classifier=self._index.classifier.name,
            assessment=soap_note.assessment.claim,
            subjective=soap_note.subjective.claim or "—",
            objective=soap_note.objective.claim or "—",
            candidates=self._render_candidates(candidates),
        )
        try:
            return await self._llm.ainvoke(prompt)
        except ValidationError as e:
            # Некоторые модели (Qwen через OpenRouter) оборачивают объект
            # в одноэлементный список; разворачиваем вместо отказа.
            data = e.errors()[0].get("input")
            if isinstance(data, list) and data:
                return RerankOut.model_validate(data[0])
            raise

    def _allowed_codes(self, candidates: list[DiagnosisCoding]) -> set[str]:
        """Кандидаты + их окрестность — ровно то, что видит модель в промпте."""
        allowed: set[str] = set()
        for c in candidates:
            allowed.add(c.code)
            allowed.update(self._index.parent_chain(c.code))
            allowed.update(child.code for child in self._index.children_of(c.code))
        return allowed

    def _render_candidates(self, candidates: list[DiagnosisCoding]) -> str:
        lines: list[str] = []
        for c in candidates:
            lines.append(f"- {c.code}: {c.title} (найдено по: {c.matched_formulation})")
            parents = self._index.parent_chain(c.code)
            if parents:
                rendered = ", ".join(
                    f"{p} ({self._index.name_of(p) or '?'})" for p in parents
                )
                lines.append(f"    родители: {rendered}")
            children = self._index.children_of(c.code)
            if children:
                rendered = "; ".join(f"{ch.code} ({ch.name})" for ch in children)
                lines.append(f"    подкоды: {rendered}")
        return "\n".join(lines)

    def _to_coding(self, result: RerankOut) -> DiagnosisCoding:
        assert result.code is not None
        return DiagnosisCoding(
            code=result.code,
            title=self._index.name_of(result.code) or result.code,
            matched_formulation="LLM-реранк по кандидатам ретрива",
            score=FloatRangedScore(min(1.0, max(0.0, result.confidence))),
            classifier=self._index.classifier,
        )
