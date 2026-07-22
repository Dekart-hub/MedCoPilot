"""The SOAP domain: the structured result of extracting a dialogue.

Pure domain — no persistence, no LLM, no serialization concerns. A
:class:`SoapReport` is a list of :class:`SoapNote`\\ s for one encounter; each
note carries the four SOAP sections (Subjective / Objective / Assessment /
Plan), and every section holds :class:`SoapClaim`\\ s. Each claim is traceable:
it must cite at least one source dialogue turn (a verbatim quote is optional).

The Assessment section holds :class:`AssessmentClaim`\\ s, which reserve a slot
for an ICD coding (populated in T10). Note-level ``confidence`` is reserved for
T12. Both are left empty here — this task only fixes the shape.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from dialogue.dialogue import DialogueTurnId
from shared.entity import Entity
from shared.value_objects import Id

type SoapClaimId = Id[SoapClaim]
type SoapNoteId = Id[SoapNote]
type SoapReportId = Id[SoapReport]


class SoapSection(StrEnum):
    """The four sections of a SOAP note, in canonical order."""

    SUBJECTIVE = "subjective"
    OBJECTIVE = "objective"
    ASSESSMENT = "assessment"
    PLAN = "plan"


@dataclass(frozen=True, slots=True)
class TurnCitation:
    """Traceability link from a claim back to a source dialogue turn.

    The ``turn_id`` is the load-bearing evidence (FR-6); ``quote`` is an
    optional verbatim span and may be omitted.
    """

    turn_id: DialogueTurnId
    quote: str | None = None


@dataclass(frozen=True, slots=True)
class IcdCoding:
    """An ICD code assigned to an assessment, with a link to its classifier.

    Reserved shape for T10, which populates instances during coding.
    """

    code: str
    name: str
    classifier_url: str


class IcdResolutionStatus(StrEnum):
    """Outcome of resolving an assessment text against the ICD catalog (T29).

    ``AMBIGUOUS`` is reserved: phase 1 always selects the top candidate, the
    score/margin gates that produce it arrive with phase 2. The member ships
    now so the wire and storage values are frozen before the behaviour lands.
    """

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class IcdCandidate:
    """One ranked entry from the resolver's candidate pool.

    ``bm25_score`` is a raw lexical ranking signal, not a probability — it is
    only meaningful relative to the other candidates of the same resolution.
    """

    code: str
    name: str
    rank: int
    bm25_score: float


@dataclass(frozen=True, slots=True)
class IcdResolution:
    """The full outcome of ICD resolution for one assessment claim.

    Keeps the audit trail next to the decision: the ordered candidate pool the
    ``selected`` coding was chosen from, and the classifier version it was
    chosen against.
    """

    status: IcdResolutionStatus
    selected: IcdCoding | None
    candidates: tuple[IcdCandidate, ...]
    classifier_version: str


@dataclass(eq=False, slots=True)
class SoapClaim(Entity[SoapClaimId]):
    """A single extracted statement, traceable to its source turn(s).

    A claim without any citation is invalid by construction: extractions must
    stay grounded in the dialogue.
    """

    id: SoapClaimId
    text: str
    citations: list[TurnCitation]

    def __post_init__(self) -> None:
        if not self.citations:
            raise ValueError("a SoapClaim must cite at least one dialogue turn")


@dataclass(eq=False, slots=True)
class AssessmentClaim(SoapClaim):
    """An Assessment-section claim that may carry an ICD coding (set in T10).

    ``icd_resolution`` (T29) records how the coding was chosen: status, the
    ranked candidate pool and the classifier version. ``icd`` stays the single
    source of the chosen code — a manual correction may replace it while the
    resolution keeps describing the original automated pass.
    """

    icd: IcdCoding | None = None
    icd_resolution: IcdResolution | None = None


@dataclass(eq=False, slots=True)
class SoapNote(Entity[SoapNoteId]):
    """One SOAP note: the four sections, each a list of claims.

    Section membership is unambiguous — a claim's section is the field it lives
    in — and only the Assessment section accepts ICD-bearing claims.
    """

    id: SoapNoteId
    subjective: list[SoapClaim] = field(default_factory=list)
    objective: list[SoapClaim] = field(default_factory=list)
    assessment: list[AssessmentClaim] = field(default_factory=list)
    plan: list[SoapClaim] = field(default_factory=list)
    confidence: float | None = None

    def sections(self) -> list[tuple[SoapSection, Sequence[SoapClaim]]]:
        """Ordered ``(section, claims)`` pairs — the single source of ordering."""
        return [
            (SoapSection.SUBJECTIVE, self.subjective),
            (SoapSection.OBJECTIVE, self.objective),
            (SoapSection.ASSESSMENT, self.assessment),
            (SoapSection.PLAN, self.plan),
        ]


@dataclass(eq=False, slots=True)
class SoapReport(Entity[SoapReportId]):
    """The SOAP notes extracted from one encounter."""

    id: SoapReportId
    notes: list[SoapNote] = field(default_factory=list)
