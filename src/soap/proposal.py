"""LLM SOAP-editor proposals — the pure domain model (story #12).

The LLM never edits a :class:`~soap.correction.SoapReportCorrection` directly. It
forms a **proposal**: an ordered list of operations addressing the doctor's notes
by their stable ids. Nothing is applied until the doctor decides each operation
individually. A :class:`CorrectionEditorSession` — one per correction — keeps the
whole trace: every proposal, every operation and every per-operation decision,
with before-snapshots and target fingerprints, so the online acceptance metric
(#12/FR-6) and offline label mining (#11/T35) can read the full history later.

Operations come in three shapes, each carrying its own decision state:

* ``ADD_NOTE`` — brand-new note content (no target, no snapshot);
* ``UPDATE_NOTE`` — replace an existing note's content by its id;
* ``DELETE_NOTE`` — drop an existing note by its id.

Staleness is per-note: an ``UPDATE``/``DELETE`` snapshots its target and records
a fingerprint of it, and accepting the operation is refused if that note has
since changed (or vanished). A doctor's edit to *another* note leaves the
operation decidable. **ICD never travels through a proposal**: the proposed
content type simply has no channel for it, so an ``UPDATE`` can only carry the
S/O/A/P text — the existing note's ICD is preserved by the applier (T33), never
set here.

Pure domain — no persistence, no LLM, no wall clock. Timestamps are injected by
the application layer, exactly as in :mod:`soap.correction`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from shared.entity import Entity
from shared.value_objects import Id

from .correction import CorrectedNote, CorrectionId, CorrectionStatus, SoapReportCorrection
from .soap import AssessmentClaim, SoapClaim, SoapNoteId, TurnCitation

type SessionId = Id[CorrectionEditorSession]
type ProposalId = Id[CorrectionProposal]
type OperationId = Id[ProposalOperation]


class OperationType(StrEnum):
    """What an operation does to the doctor's notes."""

    ADD_NOTE = "add_note"
    UPDATE_NOTE = "update_note"
    DELETE_NOTE = "delete_note"


class OperationDecision(StrEnum):
    """The doctor's per-operation verdict."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ProposalStatus(StrEnum):
    """A proposal's outcome, derived from its operations' decisions."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    MIXED = "mixed"


class ProposalError(Exception):
    """Base class for LLM-proposal domain errors."""


class CorrectionNotProposable(ProposalError):
    """Raised when a proposal is formed against a non-draft correction."""


class ActiveProposalExists(ProposalError):
    """Raised when a new proposal is opened while one still has pending operations."""


class WrongCorrection(ProposalError):
    """Raised when a proposal is formed against a correction the session does not own."""


class EmptyProposal(ProposalError):
    """Raised when a proposal carries no operations."""


class UnknownProposalTarget(ProposalError):
    """Raised when an operation targets a note absent from the correction."""


class DuplicateOperationTarget(ProposalError):
    """Raised when two operations in one proposal address the same note."""


class OperationNotInProposal(ProposalError):
    """Raised when an operation id does not belong to the proposal."""


class ConflictingDecision(ProposalError):
    """Raised when an already-decided operation is flipped to the opposite verdict."""


class StaleOperationTarget(ProposalError):
    """Raised when accepting an operation whose target note has since changed."""


@dataclass(frozen=True, slots=True)
class ProposedClaim:
    """A single claim the LLM proposes: text and citations — deliberately no ICD."""

    text: str
    citations: list[TurnCitation]


@dataclass(frozen=True, slots=True)
class ProposedNote:
    """Note content proposed by the LLM. Has no channel for ICD by construction."""

    subjective: list[ProposedClaim] = field(default_factory=list)
    objective: list[ProposedClaim] = field(default_factory=list)
    assessment: list[ProposedClaim] = field(default_factory=list)
    plan: list[ProposedClaim] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AddNoteOperation:
    """Spec for an ``ADD_NOTE`` operation: the full content of a new note."""

    content: ProposedNote


@dataclass(frozen=True, slots=True)
class UpdateNoteOperation:
    """Spec for an ``UPDATE_NOTE`` operation: replace note ``target_note_id``."""

    target_note_id: SoapNoteId
    content: ProposedNote


@dataclass(frozen=True, slots=True)
class DeleteNoteOperation:
    """Spec for a ``DELETE_NOTE`` operation: drop note ``target_note_id``."""

    target_note_id: SoapNoteId


type OperationSpec = AddNoteOperation | UpdateNoteOperation | DeleteNoteOperation


@dataclass(eq=False, slots=True)
class ProposalOperation(Entity[OperationId]):
    """One LLM edit, decided on its own.

    ``before`` is a frozen snapshot of the target note as it stood when the
    proposal was formed (``None`` for an add); ``target_fingerprint`` is that
    note's content fingerprint, replayed on accept to catch a stale target.
    ``proposed`` is the new content (``None`` for a delete).
    """

    id: OperationId
    type: OperationType
    target_note_id: SoapNoteId | None
    proposed: ProposedNote | None
    before: CorrectedNote | None
    target_fingerprint: str | None
    decision: OperationDecision = OperationDecision.PENDING
    decided_at: datetime | None = None
    decision_reason: str | None = None

    def accept(self, *, current: CorrectedNote | None, at: datetime) -> None:
        """Accept the operation; re-decide is idempotent, the opposite verdict conflicts."""
        if self.decision is OperationDecision.ACCEPTED:
            return
        if self.decision is OperationDecision.REJECTED:
            raise ConflictingDecision("a rejected operation cannot be accepted")
        self._guard_fresh_target(current)
        self.decision = OperationDecision.ACCEPTED
        self.decided_at = at

    def reject(self, *, at: datetime, reason: str | None = None) -> None:
        """Reject the operation; re-decide is idempotent, the opposite verdict conflicts."""
        if self.decision is OperationDecision.REJECTED:
            return
        if self.decision is OperationDecision.ACCEPTED:
            raise ConflictingDecision("an accepted operation cannot be rejected")
        self.decision = OperationDecision.REJECTED
        self.decided_at = at
        self.decision_reason = reason

    def is_pending(self) -> bool:
        return self.decision is OperationDecision.PENDING

    def _guard_fresh_target(self, current: CorrectedNote | None) -> None:
        if self.type is OperationType.ADD_NOTE:
            return
        if current is None or fingerprint_of(current) != self.target_fingerprint:
            raise StaleOperationTarget("the target note changed since the proposal was formed")


@dataclass(eq=False, slots=True)
class CorrectionProposal(Entity[ProposalId]):
    """One LLM editing turn: a request, its provenance and the ordered operations."""

    id: ProposalId
    user_request: str
    base_correction_revision: int
    model_id: str
    prompt_version: str
    created_at: datetime
    updated_at: datetime
    operations: list[ProposalOperation] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        correction: SoapReportCorrection,
        user_request: str,
        model_id: str,
        prompt_version: str,
        operations: Sequence[OperationSpec],
        at: datetime,
    ) -> CorrectionProposal:
        """Build a proposal, snapshotting each operation's target against ``correction``."""
        built = [_build_operation(spec, correction) for spec in operations]
        _guard_non_empty(built)
        _guard_unique_targets(built)
        return cls(
            id=Id.new(),
            user_request=user_request,
            base_correction_revision=correction.revision,
            model_id=model_id,
            prompt_version=prompt_version,
            created_at=at,
            updated_at=at,
            operations=built,
        )

    def accept_operation(
        self, operation_id: OperationId, correction: SoapReportCorrection, *, at: datetime
    ) -> ProposalOperation:
        """Accept one operation, checking its target against the live correction."""
        operation = self._operation(operation_id)
        current = _live_target(operation, correction)
        operation.accept(current=current, at=at)
        self.updated_at = at
        return operation

    def reject_operation(
        self, operation_id: OperationId, *, at: datetime, reason: str | None = None
    ) -> ProposalOperation:
        """Reject one operation."""
        operation = self._operation(operation_id)
        operation.reject(at=at, reason=reason)
        self.updated_at = at
        return operation

    def reject_pending(self, *, at: datetime, reason: str) -> None:
        """Reject every still-pending operation, recording ``reason`` on each."""
        for operation in self.operations:
            if operation.is_pending():
                operation.reject(at=at, reason=reason)
        self.updated_at = at

    def has_pending(self) -> bool:
        return any(operation.is_pending() for operation in self.operations)

    def status(self) -> ProposalStatus:
        """Derive the proposal outcome from its operations' decisions."""
        decisions = {operation.decision for operation in self.operations}
        if OperationDecision.PENDING in decisions:
            return ProposalStatus.PENDING
        if decisions == {OperationDecision.ACCEPTED}:
            return ProposalStatus.ACCEPTED
        if decisions == {OperationDecision.REJECTED}:
            return ProposalStatus.REJECTED
        return ProposalStatus.MIXED

    def _operation(self, operation_id: OperationId) -> ProposalOperation:
        for operation in self.operations:
            if operation.id == operation_id:
                return operation
        raise OperationNotInProposal("operation does not belong to this proposal")


@dataclass(eq=False, slots=True)
class CorrectionEditorSession(Entity[SessionId]):
    """The LLM-editing session of a single correction — its full proposal history.

    History lives here and nowhere else: proposals of a correction are never
    shared with another report. At most one proposal may hold pending operations
    at a time; superseded proposals keep all their operations and decisions.
    """

    id: SessionId
    correction_id: CorrectionId
    created_at: datetime
    updated_at: datetime
    proposals: list[CorrectionProposal] = field(default_factory=list)

    @classmethod
    def start(cls, correction_id: CorrectionId, *, created_at: datetime) -> CorrectionEditorSession:
        """Open an empty editing session for ``correction_id``."""
        return cls(
            id=Id.new(),
            correction_id=correction_id,
            created_at=created_at,
            updated_at=created_at,
        )

    def propose(
        self,
        correction: SoapReportCorrection,
        *,
        user_request: str,
        model_id: str,
        prompt_version: str,
        operations: Sequence[OperationSpec],
        at: datetime,
    ) -> CorrectionProposal:
        """Form a new proposal against ``correction`` without modifying it."""
        self._guard_owns(correction)
        self._guard_proposable(correction)
        self._guard_no_active_proposal()
        proposal = CorrectionProposal.create(
            correction=correction,
            user_request=user_request,
            model_id=model_id,
            prompt_version=prompt_version,
            operations=operations,
            at=at,
        )
        self.proposals.append(proposal)
        self.updated_at = at
        return proposal

    def active_proposal(self) -> CorrectionProposal | None:
        """The one proposal still awaiting decisions, or ``None``."""
        return next((p for p in self.proposals if p.has_pending()), None)

    def reject_pending(self, *, at: datetime, reason: str) -> CorrectionProposal | None:
        """Reject the active proposal's pending operations (a doctor manual edit does this)."""
        active = self.active_proposal()
        if active is not None:
            active.reject_pending(at=at, reason=reason)
            self.updated_at = at
        return active

    def _guard_owns(self, correction: SoapReportCorrection) -> None:
        if correction.id != self.correction_id:
            raise WrongCorrection("proposal correction does not match the session")

    def _guard_proposable(self, correction: SoapReportCorrection) -> None:
        if correction.status is not CorrectionStatus.DRAFT:
            raise CorrectionNotProposable(
                f"a correction in {correction.status.value} state does not accept proposals"
            )

    def _guard_no_active_proposal(self) -> None:
        if self.active_proposal() is not None:
            raise ActiveProposalExists("a proposal with pending operations already exists")


def fingerprint_of(note: CorrectedNote) -> str:
    """A content fingerprint of a corrected note; changes iff the note's content does.

    Claim identities are excluded — only text, citations, section order and ICD
    matter — so a load/save round-trip is stable while any real edit shifts it.
    """
    payload = {
        "source_note_id": _opt_str(note.source_note_id),
        "sections": {
            section.value: [_claim_fingerprint(claim) for claim in claims]
            for section, claims in note.sections()
        },
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _claim_fingerprint(claim: SoapClaim) -> dict[str, object]:
    icd = claim.icd if isinstance(claim, AssessmentClaim) else None
    return {
        "text": claim.text,
        "citations": [[str(c.turn_id), c.quote] for c in claim.citations],
        "icd": None if icd is None else [icd.code, icd.name, icd.classifier_url],
    }


def _opt_str(value: object | None) -> str | None:
    return None if value is None else str(value)


def _build_operation(spec: OperationSpec, correction: SoapReportCorrection) -> ProposalOperation:
    if isinstance(spec, AddNoteOperation):
        return ProposalOperation(
            id=Id.new(),
            type=OperationType.ADD_NOTE,
            target_note_id=None,
            proposed=spec.content,
            before=None,
            target_fingerprint=None,
        )
    target = _require_target(spec.target_note_id, correction)
    if isinstance(spec, UpdateNoteOperation):
        return ProposalOperation(
            id=Id.new(),
            type=OperationType.UPDATE_NOTE,
            target_note_id=spec.target_note_id,
            proposed=spec.content,
            before=deepcopy(target),
            target_fingerprint=fingerprint_of(target),
        )
    return ProposalOperation(
        id=Id.new(),
        type=OperationType.DELETE_NOTE,
        target_note_id=spec.target_note_id,
        proposed=None,
        before=deepcopy(target),
        target_fingerprint=fingerprint_of(target),
    )


def _require_target(note_id: SoapNoteId, correction: SoapReportCorrection) -> CorrectedNote:
    note = correction.find_note(note_id)
    if note is None:
        raise UnknownProposalTarget("operation targets a note absent from the correction")
    return note


def _live_target(
    operation: ProposalOperation, correction: SoapReportCorrection
) -> CorrectedNote | None:
    if operation.target_note_id is None:
        return None
    return correction.find_note(operation.target_note_id)


def _guard_non_empty(operations: Sequence[ProposalOperation]) -> None:
    if not operations:
        raise EmptyProposal("a proposal must carry at least one operation")


def _guard_unique_targets(operations: Sequence[ProposalOperation]) -> None:
    targets = [op.target_note_id for op in operations if op.target_note_id is not None]
    if len(targets) != len(set(targets)):
        raise DuplicateOperationTarget("a note may be addressed by at most one operation")
