from __future__ import annotations

import asyncio

from dialogue import DialogueTurnId
from shared.prompts import InMemoryPromptStore
from shared.value_objects import Id
from soap.coding import (
    DEFAULT_CODING_PROMPTS,
    LlmRerankedDiagnosisNormalizer,
    MkbIndex,
)
from soap.coding.preprocess_en import normalize as normalize_en
from soap.coding.reranker import RerankOut
from soap.soap import SoapClaim, SoapEvidence, SoapNote

# --- мини-каталог ICD-10-CM для юнит-тестов ------------------------------- #

_VOL1 = [
    {"ID": "E11", "MKB_CODE": "E11", "MKB_NAME": "Type 2 diabetes mellitus", "ID_PARENT": None, "ACTUAL": "1"},
    {"ID": "E11.9", "MKB_CODE": "E11.9", "MKB_NAME": "Type 2 diabetes mellitus without complications", "ID_PARENT": "E11", "ACTUAL": "1"},
    {"ID": "E11.21", "MKB_CODE": "E11.21", "MKB_NAME": "Type 2 diabetes mellitus with diabetic nephropathy", "ID_PARENT": "E11", "ACTUAL": "1"},
    {"ID": "J18.9", "MKB_CODE": "J18.9", "MKB_NAME": "Pneumonia, unspecified organism", "ID_PARENT": None, "ACTUAL": "1"},
]

_VOL3 = [
    {"S_NAME": "Diabetes type 2", "ICD-10": "E11"},
    {"S_NAME": "Pneumonia", "ICD-10": "J18.9"},
]


def _index() -> MkbIndex:
    return MkbIndex.from_records(_VOL1, _VOL3, tokenizer=normalize_en)


class StubModel:
    """Подменяет BaseChatModel: with_structured_output -> сам себя, ainvoke -> ответ."""

    def __init__(self, result: RerankOut | None = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.prompt: str | None = None

    def with_structured_output(self, schema):  # noqa: ANN001, ANN201
        return self

    async def ainvoke(self, prompt: str) -> RerankOut:
        self.prompt = prompt
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def _note(assessment: str, subjective: str = "", objective: str = "") -> SoapNote:
    def claim(text: str) -> SoapClaim:
        turn_id: DialogueTurnId = Id.new()
        return SoapClaim(
            id=Id.new(),
            claim=text,
            evidence=SoapEvidence(text=text, turn_id=turn_id),
        )

    return SoapNote(
        id=Id.new(),
        subjective=claim(subjective),
        objective=claim(objective),
        assessment=claim(assessment),
        plan=claim(""),
    )


def _normalize(model: StubModel, note: SoapNote):
    normalizer = LlmRerankedDiagnosisNormalizer(
        _index(), model, InMemoryPromptStore(DEFAULT_CODING_PROMPTS)
    )
    return asyncio.run(normalizer.normalize(note))


# --- выбор кода ------------------------------------------------------------ #


def test_llm_choice_lands_in_selected():
    model = StubModel(RerankOut(code="E11.9", rationale="без осложнений", confidence=0.9))
    coding = _normalize(model, _note("type 2 diabetes"))
    assert coding.selected is not None
    assert coding.selected.code == "E11.9"
    assert coding.best is coding.selected
    assert coding.rationale == "без осложнений"


def test_llm_can_refine_to_child_not_in_candidates():
    # Ретрив находит рубрику E11; дети (E11.9, E11.21) показаны как окрестность,
    # и LLM вправе уточнить до них по S/O.
    model = StubModel(RerankOut(code="E11.21", rationale="нефропатия в O", confidence=0.8))
    coding = _normalize(model, _note("type 2 diabetes", objective="diabetic nephropathy"))
    assert coding.selected is not None
    assert coding.selected.code == "E11.21"
    assert coding.selected.title == "Type 2 diabetes mellitus with diabetic nephropathy"


def test_candidates_survive_for_audit():
    model = StubModel(RerankOut(code="E11.9", rationale="ok", confidence=0.9))
    coding = _normalize(model, _note("type 2 diabetes"))
    assert coding.candidates  # выбор не стирает кандидатов ретрива


# --- деградация до Tier 1 --------------------------------------------------- #


def test_hallucinated_code_is_rejected():
    model = StubModel(RerankOut(code="Z99.9", rationale="выдумка", confidence=0.9))
    coding = _normalize(model, _note("type 2 diabetes"))
    assert coding.selected is None
    assert coding.best is not None  # лексический top-1 остаётся


def test_llm_refusal_keeps_lexical_top1_and_rationale():
    model = StubModel(RerankOut(code=None, rationale="ни один не подходит", confidence=0.2))
    coding = _normalize(model, _note("type 2 diabetes"))
    assert coding.selected is None
    assert coding.rationale == "ни один не подходит"
    assert coding.best is not None


def test_llm_failure_falls_back_to_lexical():
    model = StubModel(error=RuntimeError("api down"))
    coding = _normalize(model, _note("type 2 diabetes"))
    assert coding.selected is None
    assert coding.best is not None


def test_no_candidates_means_llm_is_not_called():
    model = StubModel(error=AssertionError("не должен вызываться"))
    coding = _normalize(model, _note("qqqq wwww"))
    assert coding.candidates == []
    assert model.prompt is None


def test_list_wrapped_llm_response_is_unwrapped():
    # Qwen через OpenRouter иногда отдаёт [{...}] вместо {...}: structured
    # output падает с ValidationError, а реранкер должен развернуть список.
    class ListWrappingModel(StubModel):
        async def ainvoke(self, prompt: str) -> RerankOut:
            self.prompt = prompt
            return RerankOut.model_validate(
                [{"code": "E11.9", "rationale": "ok", "confidence": 0.9}]
            )

    coding = _normalize(ListWrappingModel(), _note("type 2 diabetes"))
    assert coding.selected is not None
    assert coding.selected.code == "E11.9"


# --- промпт ----------------------------------------------------------------- #


def test_prompt_carries_soap_context_and_neighborhood():
    model = StubModel(RerankOut(code="E11.9", rationale="ok", confidence=0.9))
    _normalize(model, _note("type 2 diabetes", subjective="thirst", objective="hba1c 8.1"))
    assert model.prompt is not None
    assert "type 2 diabetes" in model.prompt
    assert "thirst" in model.prompt
    assert "hba1c 8.1" in model.prompt
    # Окрестность кандидата E11: дети с уточнением знаков.
    assert "E11.9" in model.prompt
    assert "E11.21" in model.prompt
