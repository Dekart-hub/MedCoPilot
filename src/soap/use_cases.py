import asyncio

from dialogue import Dialogue
from shared.value_objects import Id

from .coding.coding import SoapCodingReport
from .coding.normalizer import DiagnosisNormalizer
from .extractor import SoapExtractor
from .score.score import SoapConfidenceReport, SoapNoteConfidenceScore
from .score.scorer import ConfidenceScorer
from .coding.coding import SoapNoteCoding
from .soap import SoapReport
from .view import ReportView, to_view


class ExtractScoredSoap:
    """Извлечение SOAP + два независимых обогащения, собранных в один view.

    Поток: ``extract`` (барьер — нужен обоим) -> ``score`` ∥ ``normalize``
    (независимы, гоняются параллельно) -> ``to_view`` (джойн в дерево).
    """

    def __init__(
        self,
        extractor: SoapExtractor,
        scorer: ConfidenceScorer,
        normalizer: DiagnosisNormalizer,
    ) -> None:
        self.extractor = extractor
        self.scorer = scorer
        self.normalizer = normalizer

    async def execute(self, dialogue: Dialogue) -> ReportView:
        report = await self.extractor.extract(dialogue)
        scores, codings = await asyncio.gather(
            self._score_all(dialogue, report),
            self._normalize_all(report),
        )
        confidence = SoapConfidenceReport(
            id=Id.new(),
            soap_report_id=report.id,
            confidence_scores=scores,
        )
        coding = SoapCodingReport(
            id=Id.new(),
            soap_report_id=report.id,
            codings=codings,
        )
        return to_view(report, confidence, coding)

    async def _score_all(
        self, dialogue: Dialogue, report: SoapReport
    ) -> list[SoapNoteConfidenceScore]:
        return list(
            await asyncio.gather(
                *(self.scorer.score(dialogue, note) for note in report.soap_notes)
            )
        )

    async def _normalize_all(self, report: SoapReport) -> list[SoapNoteCoding]:
        return list(
            await asyncio.gather(
                *(self.normalizer.normalize(note) for note in report.soap_notes)
            )
        )
