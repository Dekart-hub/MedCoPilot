"""Immutable EHR publication snapshots and their durable delivery event."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from dialogue.dialogue import Dialogue
from shared.entity import Entity
from shared.value_objects import Id
from soap.correction import (
    CorrectedNote,
    CorrectionId,
    CorrectionStatus,
    SoapReportCorrection,
)
from soap.soap import AssessmentClaim, SoapClaim, SoapReportId

SNAPSHOT_SCHEMA_VERSION = 1
_FHIR_ID = re.compile(r"^[A-Za-z0-9.-]{1,64}$")

type PublicationId = Id[EhrPublication]
type PublicationOutboxId = Id[PublicationOutbox]


class PublicationStatus(StrEnum):
    PENDING = "pending"
    DELIVERED = "delivered"


class PublicationError(Exception):
    """Base class for publication domain failures."""


class PublicationRequiresVerifiedCorrection(PublicationError):
    """Raised when a draft or already locked correction is published."""


class InvalidFhirReference(PublicationError):
    """Raised when a Patient, Encounter or Practitioner reference is invalid."""


class PublicationInvalidTransition(PublicationError):
    """Raised when a delivered publication is changed again."""


class SnapshotIntegrityError(PublicationError):
    """Raised when persisted immutable content no longer matches its hash."""


@dataclass(frozen=True, slots=True)
class SnapshotCitation:
    turn_id: str
    quote: str | None


@dataclass(frozen=True, slots=True)
class SnapshotIcd:
    code: str
    name: str
    classifier_url: str


@dataclass(frozen=True, slots=True)
class SnapshotClaim:
    id: str
    text: str
    citations: tuple[SnapshotCitation, ...]
    icd: SnapshotIcd | None = None


@dataclass(frozen=True, slots=True)
class SnapshotNote:
    id: str
    source_note_id: str | None
    subjective: tuple[SnapshotClaim, ...]
    objective: tuple[SnapshotClaim, ...]
    assessment: tuple[SnapshotClaim, ...]
    plan: tuple[SnapshotClaim, ...]


@dataclass(frozen=True, slots=True)
class SnapshotDialogueTurn:
    id: str
    speaker: str
    text: str


@dataclass(frozen=True, slots=True)
class PublicationSnapshot:
    schema_version: int
    dialogue_id: str
    dialogue_turns: tuple[SnapshotDialogueTurn, ...]
    source_report_id: str
    correction_id: str
    verified_by: str
    verified_at: datetime
    notes: tuple[SnapshotNote, ...]

    @classmethod
    def capture(cls, dialogue: Dialogue, correction: SoapReportCorrection) -> PublicationSnapshot:
        if (
            correction.status is not CorrectionStatus.VERIFIED
            or correction.verified_by is None
            or correction.verified_at is None
        ):
            raise PublicationRequiresVerifiedCorrection(
                "only a verified correction can be published"
            )
        return cls(
            schema_version=SNAPSHOT_SCHEMA_VERSION,
            dialogue_id=str(dialogue.id),
            dialogue_turns=tuple(
                SnapshotDialogueTurn(id=str(turn.id), speaker=turn.speaker, text=turn.text)
                for turn in dialogue.turns
            ),
            source_report_id=str(correction.source_report_id),
            correction_id=str(correction.id),
            verified_by=correction.verified_by,
            verified_at=correction.verified_at,
            notes=tuple(_snapshot_note(note) for note in correction.notes),
        )


@dataclass(eq=False, slots=True)
class EhrPublication(Entity[PublicationId]):
    id: PublicationId
    source_report_id: SoapReportId
    correction_id: CorrectionId
    patient_ref: str
    encounter_ref: str
    author_ref: str
    snapshot: PublicationSnapshot
    snapshot_hash: str
    snapshot_schema_version: int
    status: PublicationStatus
    created_at: datetime
    updated_at: datetime
    remote_reference: str | None = None
    remote_version: str | None = None
    delivered_at: datetime | None = None

    def __post_init__(self) -> None:
        self.verify_snapshot()

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"snapshot", "snapshot_hash", "snapshot_schema_version"} and hasattr(self, name):
            raise AttributeError(f"{name} is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def request(
        cls,
        correction: SoapReportCorrection,
        dialogue: Dialogue,
        *,
        patient_ref: str,
        encounter_ref: str,
        author_ref: str,
        at: datetime,
    ) -> EhrPublication:
        validate_reference(patient_ref, "Patient")
        validate_reference(encounter_ref, "Encounter")
        validate_reference(author_ref, "Practitioner")
        snapshot = PublicationSnapshot.capture(dialogue, correction)
        return cls(
            id=Id.new(),
            source_report_id=correction.source_report_id,
            correction_id=correction.id,
            patient_ref=patient_ref,
            encounter_ref=encounter_ref,
            author_ref=author_ref,
            snapshot=snapshot,
            snapshot_hash=snapshot_hash(snapshot),
            snapshot_schema_version=snapshot.schema_version,
            status=PublicationStatus.PENDING,
            created_at=at,
            updated_at=at,
        )

    def verify_snapshot(self) -> None:
        if self.snapshot_schema_version != self.snapshot.schema_version:
            raise SnapshotIntegrityError("publication snapshot schema version changed")
        if self.snapshot_hash != snapshot_hash(self.snapshot):
            raise SnapshotIntegrityError("publication snapshot hash mismatch")

    def mark_delivered(
        self,
        *,
        remote_reference: str,
        remote_version: str | None,
        at: datetime,
    ) -> None:
        if self.status is not PublicationStatus.PENDING:
            raise PublicationInvalidTransition("a delivered publication is terminal")
        self.status = PublicationStatus.DELIVERED
        self.remote_reference = remote_reference
        self.remote_version = remote_version
        self.delivered_at = at
        self.updated_at = at


@dataclass(eq=False, slots=True)
class PublicationOutbox(Entity[PublicationOutboxId]):
    id: PublicationOutboxId
    publication_id: PublicationId
    payload: str
    payload_hash: str
    attempt_count: int
    next_attempt_at: datetime
    created_at: datetime
    last_error: str | None = None
    delivered_at: datetime | None = None

    def __post_init__(self) -> None:
        self.verify_payload()

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"publication_id", "payload", "payload_hash"} and hasattr(self, name):
            raise AttributeError(f"{name} is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def for_publication(cls, publication: EhrPublication, *, at: datetime) -> PublicationOutbox:
        payload = canonical_json(
            {
                "event_type": "ehr.publication.requested",
                "publication_id": str(publication.id),
                "patient_ref": publication.patient_ref,
                "encounter_ref": publication.encounter_ref,
                "author_ref": publication.author_ref,
                "snapshot_hash": publication.snapshot_hash,
                "snapshot": snapshot_to_dict(publication.snapshot),
            }
        )
        return cls(
            id=Id.new(),
            publication_id=publication.id,
            payload=payload,
            payload_hash=_sha256(payload.encode()),
            attempt_count=0,
            next_attempt_at=at,
            created_at=at,
        )

    def verify_payload(self) -> None:
        if self.payload_hash != _sha256(self.payload.encode()):
            raise SnapshotIntegrityError("publication outbox payload hash mismatch")

    def record_failure(self, error: str, *, next_attempt_at: datetime) -> None:
        if self.delivered_at is not None:
            raise PublicationInvalidTransition("a delivered outbox event is terminal")
        self.attempt_count += 1
        self.last_error = error
        self.next_attempt_at = next_attempt_at

    def mark_delivered(self, *, at: datetime) -> None:
        if self.delivered_at is not None:
            raise PublicationInvalidTransition("a delivered outbox event is terminal")
        self.attempt_count += 1
        self.last_error = None
        self.delivered_at = at


def validate_reference(reference: str, resource_type: str) -> str:
    prefix = f"{resource_type}/"
    if not reference.startswith(prefix):
        raise InvalidFhirReference(f"expected {resource_type}/{{id}}, got {reference!r}")
    resource_id = reference.removeprefix(prefix)
    if not _FHIR_ID.fullmatch(resource_id):
        raise InvalidFhirReference(f"invalid FHIR reference: {reference!r}")
    return resource_id


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def snapshot_hash(snapshot: PublicationSnapshot) -> str:
    return _sha256(canonical_json(snapshot_to_dict(snapshot)).encode())


def snapshot_to_dict(snapshot: PublicationSnapshot) -> dict[str, Any]:
    return {
        "schema_version": snapshot.schema_version,
        "dialogue": {
            "id": snapshot.dialogue_id,
            "turns": [
                {"id": turn.id, "speaker": turn.speaker, "text": turn.text}
                for turn in snapshot.dialogue_turns
            ],
        },
        "correction": {
            "id": snapshot.correction_id,
            "source_report_id": snapshot.source_report_id,
            "verified_by": snapshot.verified_by,
            "verified_at": snapshot.verified_at.isoformat(),
            "notes": [_snapshot_note_to_dict(note) for note in snapshot.notes],
        },
    }


def snapshot_from_dict(data: dict[str, Any]) -> PublicationSnapshot:
    dialogue = data["dialogue"]
    correction = data["correction"]
    return PublicationSnapshot(
        schema_version=int(data["schema_version"]),
        dialogue_id=str(dialogue["id"]),
        dialogue_turns=tuple(
            SnapshotDialogueTurn(
                id=str(turn["id"]),
                speaker=str(turn["speaker"]),
                text=str(turn["text"]),
            )
            for turn in dialogue["turns"]
        ),
        source_report_id=str(correction["source_report_id"]),
        correction_id=str(correction["id"]),
        verified_by=str(correction["verified_by"]),
        verified_at=datetime.fromisoformat(str(correction["verified_at"])),
        notes=tuple(_snapshot_note_from_dict(note) for note in correction["notes"]),
    )


def _snapshot_note(note: CorrectedNote) -> SnapshotNote:
    return SnapshotNote(
        id=str(note.id),
        source_note_id=str(note.source_note_id) if note.source_note_id is not None else None,
        subjective=tuple(_snapshot_claim(claim) for claim in note.subjective),
        objective=tuple(_snapshot_claim(claim) for claim in note.objective),
        assessment=tuple(_snapshot_claim(claim) for claim in note.assessment),
        plan=tuple(_snapshot_claim(claim) for claim in note.plan),
    )


def _snapshot_claim(claim: SoapClaim) -> SnapshotClaim:
    icd = claim.icd if isinstance(claim, AssessmentClaim) else None
    return SnapshotClaim(
        id=str(claim.id),
        text=claim.text,
        citations=tuple(
            SnapshotCitation(turn_id=str(citation.turn_id), quote=citation.quote)
            for citation in claim.citations
        ),
        icd=(
            SnapshotIcd(
                code=icd.code,
                name=icd.name,
                classifier_url=icd.classifier_url,
            )
            if icd is not None
            else None
        ),
    )


def _snapshot_note_to_dict(note: SnapshotNote) -> dict[str, Any]:
    return {
        "id": note.id,
        "source_note_id": note.source_note_id,
        "sections": {
            "subjective": [_snapshot_claim_to_dict(claim) for claim in note.subjective],
            "objective": [_snapshot_claim_to_dict(claim) for claim in note.objective],
            "assessment": [_snapshot_claim_to_dict(claim) for claim in note.assessment],
            "plan": [_snapshot_claim_to_dict(claim) for claim in note.plan],
        },
    }


def _snapshot_claim_to_dict(claim: SnapshotClaim) -> dict[str, Any]:
    return {
        "id": claim.id,
        "text": claim.text,
        "citations": [
            {"turn_id": citation.turn_id, "quote": citation.quote} for citation in claim.citations
        ],
        "icd": (
            {
                "code": claim.icd.code,
                "name": claim.icd.name,
                "classifier_url": claim.icd.classifier_url,
            }
            if claim.icd is not None
            else None
        ),
    }


def _snapshot_note_from_dict(data: dict[str, Any]) -> SnapshotNote:
    sections = data["sections"]
    return SnapshotNote(
        id=str(data["id"]),
        source_note_id=(
            str(data["source_note_id"]) if data.get("source_note_id") is not None else None
        ),
        subjective=tuple(_snapshot_claim_from_dict(c) for c in sections["subjective"]),
        objective=tuple(_snapshot_claim_from_dict(c) for c in sections["objective"]),
        assessment=tuple(_snapshot_claim_from_dict(c) for c in sections["assessment"]),
        plan=tuple(_snapshot_claim_from_dict(c) for c in sections["plan"]),
    )


def _snapshot_claim_from_dict(data: dict[str, Any]) -> SnapshotClaim:
    icd = data.get("icd")
    return SnapshotClaim(
        id=str(data["id"]),
        text=str(data["text"]),
        citations=tuple(
            SnapshotCitation(
                turn_id=str(citation["turn_id"]),
                quote=(str(citation["quote"]) if citation.get("quote") is not None else None),
            )
            for citation in data["citations"]
        ),
        icd=(
            SnapshotIcd(
                code=str(icd["code"]),
                name=str(icd["name"]),
                classifier_url=str(icd["classifier_url"]),
            )
            if icd is not None
            else None
        ),
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
