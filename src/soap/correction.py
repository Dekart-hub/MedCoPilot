"""Doctor correction of a generated SOAP report — the pure domain model.

A generated :class:`~soap.soap.SoapReport` is an immutable *original*. On top of
it lives one working corrected version — the doctor's — which starts as a
``DRAFT`` and becomes ``VERIFIED`` once the doctor has checked it. Draft and
verified are the two states of that single doctor version; together with the
untouched original they are the two meaningful versions of an encounter.

Each corrected note keeps a ``source_note_id`` back to the original note it was
copied from, so downstream online-quality work (#10) can reconstruct the
lineage. A doctor-added note has ``source_note_id = None``; an original note
missing from the corrected version was deleted.

Pure domain — no persistence, no IO, no wall clock. Timestamps are injected by
the application layer so the model stays deterministic and side-effect free.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from shared.entity import Entity
from shared.value_objects import Id

from .soap import (
    AssessmentClaim,
    SoapClaim,
    SoapNote,
    SoapNoteId,
    SoapReport,
    SoapReportId,
    SoapSection,
)

type CorrectionId = Id[SoapReportCorrection]


class CorrectionStatus(StrEnum):
    """The lifecycle state of a doctor's corrected version."""

    DRAFT = "draft"
    VERIFIED = "verified"


class CorrectionError(Exception):
    """Base class for SOAP-correction domain errors."""


class CorrectionNotEditable(CorrectionError):
    """Raised when a note is added, updated or deleted on a VERIFIED correction."""


class NoteNotInCorrection(CorrectionError):
    """Raised when a note id does not belong to the correction."""


class EmptyDoctorId(CorrectionError):
    """Raised when a correction is verified without a doctor identifier."""


class DuplicateSourceNote(CorrectionError):
    """Raised when two corrected notes claim the same source note."""


@dataclass(eq=False, slots=True)
class CorrectedNote(Entity[SoapNoteId]):
    """A note in the doctor's version, linked to the original it was copied from.

    ``source_note_id`` is the lineage anchor: the id of the original
    :class:`~soap.soap.SoapNote`, or ``None`` for a note the doctor added.
    """

    id: SoapNoteId
    source_note_id: SoapNoteId | None = None
    subjective: list[SoapClaim] = field(default_factory=list)
    objective: list[SoapClaim] = field(default_factory=list)
    assessment: list[AssessmentClaim] = field(default_factory=list)
    plan: list[SoapClaim] = field(default_factory=list)

    def sections(self) -> list[tuple[SoapSection, Sequence[SoapClaim]]]:
        """Ordered ``(section, claims)`` pairs — the single source of ordering."""
        return [
            (SoapSection.SUBJECTIVE, self.subjective),
            (SoapSection.OBJECTIVE, self.objective),
            (SoapSection.ASSESSMENT, self.assessment),
            (SoapSection.PLAN, self.plan),
        ]


@dataclass(eq=False, slots=True)
class SoapReportCorrection(Entity[CorrectionId]):
    """The doctor's editable version of a source report, with lineage preserved."""

    id: CorrectionId
    source_report_id: SoapReportId
    status: CorrectionStatus
    created_at: datetime
    updated_at: datetime
    notes: list[CorrectedNote] = field(default_factory=list)
    verified_by: str | None = None
    verified_at: datetime | None = None

    def __post_init__(self) -> None:
        self._guard_unique_sources()

    @classmethod
    def start(cls, source: SoapReport, *, created_at: datetime) -> SoapReportCorrection:
        """Open a DRAFT correction as a deep copy of ``source``'s notes and their claims."""
        return cls(
            id=Id.new(),
            source_report_id=source.id,
            status=CorrectionStatus.DRAFT,
            created_at=created_at,
            updated_at=created_at,
            notes=[_copy_of(note) for note in source.notes],
        )

    def add_note(
        self,
        *,
        at: datetime,
        subjective: list[SoapClaim] | None = None,
        objective: list[SoapClaim] | None = None,
        assessment: list[AssessmentClaim] | None = None,
        plan: list[SoapClaim] | None = None,
    ) -> CorrectedNote:
        """Add a brand-new note (``source_note_id = None``); allowed in DRAFT only."""
        self.ensure_editable()
        note = CorrectedNote(
            id=Id.new(),
            source_note_id=None,
            subjective=list(subjective or []),
            objective=list(objective or []),
            assessment=list(assessment or []),
            plan=list(plan or []),
        )
        self.notes.append(note)
        self.updated_at = at
        return note

    def update_note(
        self,
        note_id: SoapNoteId,
        *,
        at: datetime,
        subjective: list[SoapClaim] | None = None,
        objective: list[SoapClaim] | None = None,
        assessment: list[AssessmentClaim] | None = None,
        plan: list[SoapClaim] | None = None,
    ) -> CorrectedNote:
        """Replace a note's four sections (ICD rides on the assessment claims); DRAFT only."""
        self.ensure_editable()
        note = self._note(note_id)
        note.subjective = list(subjective or [])
        note.objective = list(objective or [])
        note.assessment = list(assessment or [])
        note.plan = list(plan or [])
        self.updated_at = at
        return note

    def delete_note(self, note_id: SoapNoteId, *, at: datetime) -> None:
        """Drop a note from the doctor's version; allowed in DRAFT only."""
        self.ensure_editable()
        self.notes.remove(self._note(note_id))
        self.updated_at = at

    def verify(self, doctor_id: str, *, at: datetime) -> None:
        """Move DRAFT → VERIFIED, stamping the checking doctor and the time."""
        if not doctor_id.strip():
            raise EmptyDoctorId("verifying a correction requires a doctor id")
        self.status = CorrectionStatus.VERIFIED
        self.verified_by = doctor_id
        self.verified_at = at
        self.updated_at = at

    def reopen(self, *, at: datetime) -> None:
        """Move VERIFIED → DRAFT, dropping the now-stale verification stamp."""
        self.status = CorrectionStatus.DRAFT
        self.verified_by = None
        self.verified_at = None
        self.updated_at = at

    def ensure_editable(self) -> None:
        """Guard: a verified correction rejects any edit until it is reopened."""
        if self.status is CorrectionStatus.VERIFIED:
            raise CorrectionNotEditable("a verified correction must be reopened before editing")

    def _note(self, note_id: SoapNoteId) -> CorrectedNote:
        for note in self.notes:
            if note.id == note_id:
                return note
        raise NoteNotInCorrection("note does not belong to this correction")

    def _guard_unique_sources(self) -> None:
        sources = [note.source_note_id for note in self.notes if note.source_note_id is not None]
        if len(sources) != len(set(sources)):
            raise DuplicateSourceNote("a source note may appear at most once in a correction")


def _copy_of(note: SoapNote) -> CorrectedNote:
    """Deep-copy an original note into an independent corrected note carrying its lineage."""
    return CorrectedNote(
        id=Id.new(),
        source_note_id=note.id,
        subjective=deepcopy(note.subjective),
        objective=deepcopy(note.objective),
        assessment=deepcopy(note.assessment),
        plan=deepcopy(note.plan),
    )
