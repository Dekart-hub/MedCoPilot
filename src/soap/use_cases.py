import asyncio
from dataclasses import dataclass

from dialogue import Dialogue
from shared.value_objects import Id

from .extractor import SoapExtractor
from .score.score import SoapConfidenceReport
from .score.scorer import ConfidenceScorer
from .soap import SoapReport


@dataclass(frozen=True, slots=True)
class ScoredReport:
    """Результат use case: извлечённый отчёт вместе с оценками уверенности."""

    report: SoapReport
    confidence_report: SoapConfidenceReport


class ExtractScoredSoap:
    def __init__(self, extractor: SoapExtractor, scorer: ConfidenceScorer) -> None:
        self.extractor = extractor
        self.scorer = scorer

    async def execute(self, dialogue: Dialogue) -> ScoredReport:
        report = await self.extractor.extract(dialogue)
        note_scores = await asyncio.gather(
            *(self.scorer.score(dialogue, note) for note in report.soap_notes)
        )
        scores = SoapConfidenceReport(
            id=Id.new(),
            soap_report_id=report.id,
            confidence_scores=list(note_scores),
        )
        return ScoredReport(report=report, confidence_report=scores)