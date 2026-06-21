from __future__ import annotations

from dataclasses import dataclass

from shared.entity import Entity
from shared.value_objects import Id, FloatRangedScore
from ..soap import SoapNoteId, SoapReportId

type SoapNoteConfidenceScoreId = Id[SoapNoteConfidenceScore]
type SoapConfidenceReportId = Id[SoapConfidenceReport]


@dataclass(eq=False, slots=True)
class SoapNoteConfidenceScore(Entity[SoapNoteConfidenceScoreId]):
    id: SoapNoteConfidenceScoreId
    score: FloatRangedScore
    soap_note_id: SoapNoteId


@dataclass(eq=False, slots=True)
class SoapConfidenceReport(Entity[SoapConfidenceReportId]):
    id: SoapConfidenceReportId
    soap_report_id: SoapReportId
    confidence_scores: list[SoapNoteConfidenceScore]
