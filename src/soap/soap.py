from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from dialogue import DialogueTurnId
from shared.entity import Entity
from shared.value_objects import Id

type SoapClaimId = Id[SoapClaim]
type SoapNoteId = Id[SoapNote]
type SoapReportId = Id[SoapReport]


@dataclass(frozen=True, slots=True)
class SoapEvidence:
    text: str
    turn_id: DialogueTurnId


@dataclass(eq=False, slots=True)
class SoapClaim(Entity[SoapClaimId]):
    id: SoapClaimId
    claim: str
    evidence: SoapEvidence


@dataclass(eq=False, slots=True)
class SoapNote(Entity[SoapNoteId]):
    id: SoapNoteId
    subjective: SoapClaim
    objective: SoapClaim
    assessment: SoapClaim
    plan: SoapClaim

    def sections(self) -> list[tuple[str, SoapClaim]]:
        """Ordered (section name, claim) pairs — single source of S/O/A/P ordering."""
        return [
            ("subjective", self.subjective),
            ("objective", self.objective),
            ("assessment", self.assessment),
            ("plan", self.plan),
        ]


@dataclass(eq=False, slots=True)
class SoapReport(Entity[SoapReportId]):
    id: SoapReportId
    soap_notes: list[SoapNote]
    created_at: datetime
    updated_at: datetime
