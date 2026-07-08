from __future__ import annotations

from dataclasses import dataclass, field

from shared.entity import Entity
from shared.value_objects import Id, FloatRangedScore
from ..soap import SoapClaimId, SoapNoteId, SoapReportId

type SoapNoteConfidenceScoreId = Id[SoapNoteConfidenceScore]
type SoapConfidenceReportId = Id[SoapConfidenceReport]


@dataclass(frozen=True, slots=True)
class ClaimConfidenceScore:
    """Per-claim groundedness with a threshold-based review flag (Tier 1)."""

    claim_id: SoapClaimId
    section: str
    score: FloatRangedScore
    is_flagged: bool


@dataclass(eq=False, slots=True)
class SoapNoteConfidenceScore(Entity[SoapNoteConfidenceScoreId]):
    id: SoapNoteConfidenceScoreId
    score: FloatRangedScore
    soap_note_id: SoapNoteId
    claim_scores: list[ClaimConfidenceScore] = field(default_factory=list)


@dataclass(eq=False, slots=True)
class SoapConfidenceReport(Entity[SoapConfidenceReportId]):
    id: SoapConfidenceReportId
    soap_report_id: SoapReportId
    confidence_scores: list[SoapNoteConfidenceScore]
