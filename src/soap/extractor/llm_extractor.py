from __future__ import annotations

import operator
import re
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from dialogue import Dialogue, DialogueTurnId
from shared.langgraph import LangGraphAgent
from shared.prompts import PromptStore
from shared.value_objects import Id

from ..soap import (
    SoapClaim,
    SoapEvidence,
    SoapNote,
    SoapReport,
)
from .base import SoapExtractor

# --------------------------------------------------------------------------- #
# Схемы структурированного вывода LLM.
# Это DTO «на проводе» между моделью и графом: модель цитирует реплики и
# ссылается на их id строкой, а доменные сущности собираются уже в экстракторе.
# --------------------------------------------------------------------------- #


class ClaimOut(BaseModel):
    """Один пункт SOAP-ноты с цитатой-обоснованием из диалога."""

    claim: str = Field(description="Сформулированное клиническое утверждение")
    evidence_text: str = Field(description="Дословная цитата из реплики диалога")
    turn_index: int = Field(
        description="Номер реплики ([N] в тексте), на которую опирается утверждение"
    )


class NoteOut(BaseModel):
    """SOAP-нота, извлечённая по одному сегменту диалога."""

    subjective: ClaimOut
    objective: ClaimOut
    assessment: ClaimOut
    plan: ClaimOut


class Segment(BaseModel):
    """Сегмент диалога — отдельная клиническая тема, дающая одну SOAP-ноту."""

    topic: str = Field(description="Краткое описание клинической темы")
    turn_indices: list[int] = Field(
        description="Номера реплик ([N] в тексте), относящихся к теме"
    )


class SegmentsOut(BaseModel):
    """Результат планировщика: общий список сегментов («записок»)."""

    segments: list[Segment]


# --------------------------------------------------------------------------- #
# Состояние графа.
# --------------------------------------------------------------------------- #


class TurnView(TypedDict):
    """Реплика диалога в «плоском» виде — то, что видит LLM."""

    id: str
    role: str
    content: str


class ExtractionState(TypedDict):
    """Состояние графа экстракции."""

    turns: list[TurnView]
    segments: list[Segment]
    # Параллельные ветки экстракции дописывают сюда свои ноты (reducer add).
    notes: Annotated[list[NoteOut], operator.add]


class ExtractTask(TypedDict):
    """Полезная нагрузка одной параллельной ветки экстракции."""

    turns: list[TurnView]
    segment: Segment


def _render_turns(turns: list[TurnView]) -> str:
    # Реплики нумеруем 1..N: модели проще сослаться на номер, чем на UUID.
    return "\n".join(
        f"[{i}] {t['role']}: {t['content']}" for i, t in enumerate(turns, start=1)
    )


# Технический префикс рендера, который модель часто копирует в цитату:
# "[12] medic: ..." или просто "[16][17]" / "[12], [6]".
_EVIDENCE_PREFIX_RE = re.compile(r"^\s*\[\d+\]\s*[^:\n]{0,30}:\s*")
_LEADING_REFS_RE = re.compile(r"^\s*(?:\[\d+\][,\s]*)+")


def _clean_evidence(text: str) -> str:
    """Срезает префикс ``[N] role:`` / голые ссылки ``[N]`` из начала цитаты."""
    cleaned = _EVIDENCE_PREFIX_RE.sub("", text)
    if cleaned == text:
        cleaned = _LEADING_REFS_RE.sub("", cleaned)
    return cleaned.strip()


# Ключи промтов в PromptStore.
PLAN_PROMPT_KEY = "soap.plan"
EXTRACT_PROMPT_KEY = "soap.extract"

# Промты по умолчанию — DI может собрать InMemoryPromptStore поверх них или
# подменить своим хранилищем (файлы, БД и т.п.).
DEFAULT_PROMPTS: dict[str, str] = {
    PLAN_PROMPT_KEY: (
        "Ты — медицинский ассистент. Определи отдельные клинические проблемы, "
        "обсуждаемые в диалоге. ВАЖНО: обычно весь приём — это ОДНА проблема "
        "(один эпизод/жалоба), тогда верни ровно одну тему. Создавай несколько "
        "тем только если в диалоге явно обсуждаются НЕСКОЛЬКО независимых "
        "клинических проблем. Не дроби одну проблему на части (жалоба, осмотр, "
        "план — это всё одна тема). Для каждой темы укажи краткое описание и "
        "номера относящихся реплик (число [N] в начале строки).\n\n"
        "{{ turns }}"
    ),
    EXTRACT_PROMPT_KEY: (
        "Извлеки SOAP-ноту (Subjective, Objective, Assessment, Plan) строго по "
        "теме «{{ topic }}». Для каждого пункта приведи дословную цитату и "
        "номер реплики-источника (число [N] в начале строки). Формулируй "
        "клеймы на языке диалога (английский диалог — английские клеймы); "
        "assessment — это конкретный диагноз, а не слово «диагноз».\n\n"
        "{{ turns }}"
    ),
}


def build_graph(model: BaseChatModel, prompts: PromptStore) -> StateGraph:
    """Собирает граф экстракции SOAP.

    Схема: одна нода-планировщик определяет общий список сегментов («записок»),
    затем по каждому сегменту параллельно (через ``Send`` / асинхронные задачи)
    запускается экстракция, а нода-сборщик собирает итог. Тексты промтов берутся
    из ``prompts`` по ключам :data:`PLAN_PROMPT_KEY` / :data:`EXTRACT_PROMPT_KEY`.

    Возвращает несобранный ``StateGraph`` — компиляцией занимается
    :class:`~shared.langgraph.LangGraphAgent`, которого поднимает DI.
    """
    planner = model.with_structured_output(SegmentsOut)
    extractor = model.with_structured_output(NoteOut)

    async def plan(state: ExtractionState) -> dict[str, Any]:
        prompt = await prompts.get(PLAN_PROMPT_KEY, turns=_render_turns(state["turns"]))
        result: SegmentsOut = await planner.ainvoke(prompt)
        return {"segments": result.segments}

    def fan_out(state: ExtractionState) -> list[Send]:
        return [
            Send("extract", ExtractTask(turns=state["turns"], segment=segment))
            for segment in state["segments"]
        ]

    async def extract(task: ExtractTask) -> dict[str, Any]:
        segment = task["segment"]
        prompt = await prompts.get(
            EXTRACT_PROMPT_KEY,
            topic=segment.topic,
            turns=_render_turns(task["turns"]),
        )
        note: NoteOut = await extractor.ainvoke(prompt)
        return {"notes": [note]}

    def assemble(state: ExtractionState) -> dict[str, Any]:
        # Точка сборки: все параллельные ветки уже дописали ноты в state["notes"].
        return {}

    graph: StateGraph = StateGraph(ExtractionState)
    graph.add_node("plan", plan)
    graph.add_node("extract", extract)
    graph.add_node("assemble", assemble)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", fan_out, ["extract"])
    graph.add_edge("extract", "assemble")
    graph.add_edge("assemble", END)

    return graph


class LlmSoapExtractor(SoapExtractor):
    """Экстрактор SOAP поверх графа LangGraph.

    Не знает деталей графа: переводит диалог во вход графа, прогоняет его через
    :class:`~shared.langgraph.LangGraphAgent` и собирает доменный
    :class:`~soap.soap.SoapReport` из итогового состояния.
    """

    def __init__(self, agent: LangGraphAgent) -> None:
        super().__init__()
        self._agent = agent

    async def extract(self, dialogue: Dialogue) -> SoapReport:
        state = await self._agent.run(self._to_input(dialogue))
        return self._to_report(dialogue, state["notes"])

    @staticmethod
    def _to_input(dialogue: Dialogue) -> ExtractionState:
        turns: list[TurnView] = [
            {"id": str(turn.id), "role": turn.role, "content": turn.content}
            for turn in dialogue.turns
        ]
        return ExtractionState(turns=turns, segments=[], notes=[])

    @classmethod
    def _to_report(cls, dialogue: Dialogue, notes: list[NoteOut]) -> SoapReport:
        now = datetime.now(timezone.utc)
        turn_ids = [turn.id for turn in dialogue.turns]
        return SoapReport(
            id=Id.new(),
            soap_notes=[cls._to_note(note, turn_ids) for note in notes],
            created_at=now,
            updated_at=now,
        )

    @classmethod
    def _to_note(cls, note: NoteOut, turn_ids: list[DialogueTurnId]) -> SoapNote:
        return SoapNote(
            id=Id.new(),
            subjective=cls._to_claim(note.subjective, turn_ids),
            objective=cls._to_claim(note.objective, turn_ids),
            assessment=cls._to_claim(note.assessment, turn_ids),
            plan=cls._to_claim(note.plan, turn_ids),
        )

    @staticmethod
    def _to_claim(claim: ClaimOut, turn_ids: list[DialogueTurnId]) -> SoapClaim:
        # Номера 1..N -> реальный id реплики. Если модель сослалась на
        # несуществующую реплику, ставим «висячий» id: claim считается
        # необоснованным (скорер даст 0), но мы не падаем.
        if 1 <= claim.turn_index <= len(turn_ids):
            turn_id = turn_ids[claim.turn_index - 1]
        else:
            turn_id = Id.new()
        return SoapClaim(
            id=Id.new(),
            claim=claim.claim,
            evidence=SoapEvidence(
                text=_clean_evidence(claim.evidence_text), turn_id=turn_id
            ),
        )
