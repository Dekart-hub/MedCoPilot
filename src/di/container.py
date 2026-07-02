from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from config import CodingSettings, Settings, get_settings
from dialogue import (
    CreateDialogue,
    CreateDialogueFromText,
    DialogueRepository,
    InMemoryDialogueRepository,
)
from dialogue.samples import build_sample_dialogue
from infra import build_chat_model
from shared.langgraph import LangGraphAgent
from shared.prompts import InMemoryPromptStore, PromptStore
from soap import (
    ConfidenceScorer,
    DiagnosisNormalizer,
    ExtractScoredSoap,
    LexicalDiagnosisNormalizer,
    NullDiagnosisNormalizer,
    SoapExtractor,
)
from soap.coding import (
    DEFAULT_CODING_PROMPTS,
    LlmRerankedDiagnosisNormalizer,
    preprocess_en,
)
from soap.coding.coding import DEFAULT_ICD10CM_REF
from soap.coding.retrieval import MkbIndex
from soap.extractor import DEFAULT_PROMPTS, LlmSoapExtractor, build_graph
from soap.score.scorer import LexicalGroundingScorer

logger = logging.getLogger(__name__)


def build_normalizer(
    coding: CodingSettings,
    model: BaseChatModel | None = None,
    prompts: PromptStore | None = None,
) -> DiagnosisNormalizer:
    """Собирает нормализатор диагнозов по настройкам языка.

    ``en`` -> ICD-10-CM (tabular/index), ``ru`` -> МКБ-10 НСИ (Том 1/Том 3).
    Если файлов справочника нет — отдаём заглушку, чтобы приложение всё равно
    поднялось (данные качаются/парсятся отдельно, см. scripts/parse_icd10cm.py).

    При ``coding.llm_rerank`` и переданных ``model``/``prompts`` поверх
    лексического ретрива встаёт LLM-реранк (Tier 2); иначе — чистый Tier 1.
    """
    if coding.language == "en":
        vol1 = os.path.join(coding.data_dir, "tabular.jsonl")
        vol3 = os.path.join(coding.data_dir, "index.jsonl")
    else:
        vol1 = os.path.join(coding.data_dir, "mkb10_vol1.jsonl")
        vol3 = os.path.join(coding.data_dir, "mkb10_vol3_index.jsonl")

    if not (os.path.exists(vol1) and os.path.exists(vol3)):
        logger.warning(
            "Справочник кодирования не найден (%s, %s) — NullDiagnosisNormalizer",
            vol1,
            vol3,
        )
        return NullDiagnosisNormalizer()

    if coding.language == "en":
        index = MkbIndex.from_jsonl(
            vol1, vol3, base_ref=DEFAULT_ICD10CM_REF, tokenizer=preprocess_en.normalize
        )
    else:
        index = MkbIndex.from_jsonl(vol1, vol3)
    logger.info("Индекс кодирования собран (%s) из %s", coding.language, coding.data_dir)

    if coding.llm_rerank and model is not None and prompts is not None:
        return LlmRerankedDiagnosisNormalizer(
            index, model, prompts, top_n=coding.retrieval_top_n
        )
    if coding.llm_rerank:
        logger.warning("llm_rerank включён, но model/prompts не переданы — Tier 1")
    return LexicalDiagnosisNormalizer(index, top_n=coding.retrieval_top_n)


@dataclass(slots=True)
class Container:
    """Собранный граф зависимостей приложения."""

    settings: Settings
    model: BaseChatModel
    prompts: PromptStore
    agent: LangGraphAgent
    extractor: SoapExtractor
    scorer: ConfidenceScorer
    normalizer: DiagnosisNormalizer
    dialogue_repository: DialogueRepository
    create_dialogue: CreateDialogue
    create_dialogue_from_text: CreateDialogueFromText
    extract_scored_soap: ExtractScoredSoap


async def build_container() -> Container:
    """Поднимает все зависимости. Вызывается на старте приложения."""
    settings = get_settings()
    model = build_chat_model(settings)
    prompts: PromptStore = InMemoryPromptStore(
        {**DEFAULT_PROMPTS, **DEFAULT_CODING_PROMPTS}
    )
    agent = LangGraphAgent(build_graph(model, prompts))
    extractor: SoapExtractor = LlmSoapExtractor(agent)
    scorer: ConfidenceScorer = LexicalGroundingScorer()
    normalizer: DiagnosisNormalizer = build_normalizer(
        settings.coding, model=model, prompts=prompts
    )
    # Засеваем демо-диалог, чтобы /reports можно было дёрнуть без ручного создания.
    repository: DialogueRepository = InMemoryDialogueRepository(
        initial=[build_sample_dialogue()]
    )

    return Container(
        settings=settings,
        model=model,
        prompts=prompts,
        agent=agent,
        extractor=extractor,
        scorer=scorer,
        normalizer=normalizer,
        dialogue_repository=repository,
        create_dialogue=CreateDialogue(repository),
        create_dialogue_from_text=CreateDialogueFromText(repository),
        extract_scored_soap=ExtractScoredSoap(extractor, scorer, normalizer),
    )


async def teardown_container(container: Container) -> None:
    """Освобождает ресурсы. Вызывается при остановке приложения.

    Пока закрывать нечего — точка расширения для http-клиентов, пулов БД и т.п.
    """
    return None
