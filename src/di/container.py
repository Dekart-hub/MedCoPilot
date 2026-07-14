from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from config import EhrSettings, Settings, get_settings
from dialogue import (
    CreateDialogue,
    CreateDialogueFromText,
    DialogueRepository,
    InMemoryDialogueRepository,
)
from dialogue.samples import build_sample_dialogue
from ehr import (
    DisabledEhrGateway,
    EhrGateway,
    InMemoryReportRepository,
    ReportRepository,
    ReportWorkflow,
)
from infra import build_chat_model
from infra.fhir import FhirR4EhrGateway
from shared.langgraph import LangGraphAgent
from shared.prompts import InMemoryPromptStore, PromptStore
from soap import (
    ConfidenceScorer,
    DiagnosisNormalizer,
    ExtractScoredSoap,
    NullDiagnosisNormalizer,
    SoapExtractor,
)
from soap.extractor import DEFAULT_PROMPTS, LlmSoapExtractor, build_graph
from soap.score.scorer import LexicalGroundingScorer


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
    report_repository: ReportRepository
    ehr_gateway: EhrGateway
    report_workflow: ReportWorkflow
    create_dialogue: CreateDialogue
    create_dialogue_from_text: CreateDialogueFromText
    extract_scored_soap: ExtractScoredSoap


async def build_container() -> Container:
    """Поднимает все зависимости. Вызывается на старте приложения."""
    settings = get_settings()
    model = build_chat_model(settings)
    prompts: PromptStore = InMemoryPromptStore(DEFAULT_PROMPTS)
    agent = LangGraphAgent(build_graph(model, prompts))
    extractor: SoapExtractor = LlmSoapExtractor(agent)
    scorer: ConfidenceScorer = LexicalGroundingScorer(
        review_threshold=settings.scoring.review_threshold
    )
    # Заглушка, пока индекс МКБ не собран/не докачан; подменяется на
    # LexicalDiagnosisNormalizer(MkbIndex.from_jsonl(...)), когда данные готовы.
    normalizer: DiagnosisNormalizer = NullDiagnosisNormalizer()
    # Засеваем демо-диалог, чтобы /reports можно было дёрнуть без ручного создания.
    repository: DialogueRepository = InMemoryDialogueRepository(
        initial=[build_sample_dialogue()]
    )
    report_repository: ReportRepository = InMemoryReportRepository()
    ehr_gateway = _build_ehr_gateway(settings.ehr)
    report_workflow = ReportWorkflow(
        repository,
        report_repository,
        ehr_gateway,
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
        report_repository=report_repository,
        ehr_gateway=ehr_gateway,
        report_workflow=report_workflow,
        create_dialogue=CreateDialogue(repository),
        create_dialogue_from_text=CreateDialogueFromText(repository),
        extract_scored_soap=ExtractScoredSoap(extractor, scorer, normalizer),
    )


def _build_ehr_gateway(settings: EhrSettings) -> EhrGateway:
    if settings.enabled:
        return FhirR4EhrGateway(settings)
    return DisabledEhrGateway()


async def teardown_container(container: Container) -> None:
    """Освобождает ресурсы. Вызывается при остановке приложения.

    Closes the external mock-EHR HTTP client when it is enabled.
    """
    await container.ehr_gateway.aclose()
