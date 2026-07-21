"""Application use cases for the LLM SOAP editor workflow (story #12, task #81).

The LLM never edits a correction directly. A doctor's request becomes a
:class:`~soap.proposal.CorrectionProposal` — an ordered list of note operations,
each decided on its own. These use cases orchestrate that workflow over the
durable editor session (T31) and the #8 correction domain:

* :class:`ProposeCorrectionEdit` asks the :class:`~soap.llm_editor.SoapEditAgent`
  for a validated draft and records it as PENDING operations; it applies nothing.
* :class:`AcceptProposalOperation` marks one operation accepted and applies it
  **through the existing #8 use cases** (ADD → add note, UPDATE → replace note,
  DELETE → delete note), so the correction's ``revision`` bumps exactly as a
  doctor edit would. An UPDATE preserves the target note's ICD codings, which
  never travel through a proposal.
* :class:`RejectProposalOperation` records a rejection and leaves the correction
  untouched.
* :class:`RejectPendingProposals` auto-rejects the active proposal's pending
  operations when the doctor edits the correction by hand — a recorded side
  effect that stays in the durable log for the acceptance metric and offline
  mining.
* :class:`EnsureNoPendingProposal` blocks ``verify`` while an operation is
  undecided.
* :class:`ComputeAcceptanceMetric` reads the durable log and returns the
  operation-level acceptance counts and rate.

Every decision — including auto-rejects — is persisted with its before-snapshot,
proposed content, model/prompt version and timestamps. The caller (the route)
owns the commit; these use cases only ``save`` and ``flush``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from dialogue.repository import DialogueRepository
from shared.value_objects import Id

from .correction import CorrectedNote, CorrectionId, CorrectionStatus, SoapReportCorrection
from .correction_use_cases import (
    AddCorrectedNote,
    AddCorrectedNoteCommand,
    DeleteCorrectedNote,
    DeleteCorrectedNoteCommand,
    SourceReportNotFoundError,
    UpdateCorrectedNote,
    UpdateCorrectedNoteCommand,
)
from .llm_editor import EditContext, SoapEditAgent
from .proposal import (
    ActiveProposalExists,
    CorrectionEditorSession,
    CorrectionNotProposable,
    CorrectionProposal,
    OperationDecision,
    OperationId,
    OperationType,
    ProposalId,
    ProposalOperation,
    ProposedClaim,
)
from .proposal_repository import CorrectionEditorSessionRepository
from .repository import SoapReportRepository
from .soap import AssessmentClaim, IcdCoding, SoapClaim
from .use_cases import DialogueNotFoundError


class ProposalNotFoundError(Exception):
    """Raised when a correction has no editor session or the proposal id is unknown."""

    def __init__(self, proposal_id: ProposalId | None = None) -> None:
        super().__init__(
            f"proposal {proposal_id} not found" if proposal_id else "no proposal found"
        )
        self.proposal_id = proposal_id


class PendingOperationsBlockVerify(Exception):
    """Raised when a correction is verified while its active proposal has pending ops."""

    def __init__(self, correction_id: CorrectionId) -> None:
        super().__init__(f"correction {correction_id} has undecided proposal operations")
        self.correction_id = correction_id


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class DecideOperationCommand:
    """Request to accept or reject one operation of a proposal."""

    proposal_id: ProposalId
    operation_id: OperationId


class ProposeCorrectionEdit:
    """Generates a proposal via the edit agent and records its PENDING operations."""

    def __init__(
        self,
        session: AsyncSession,
        sessions: CorrectionEditorSessionRepository,
        reports: SoapReportRepository,
        dialogues: DialogueRepository,
        agent: SoapEditAgent,
    ) -> None:
        self._session = session
        self._sessions = sessions
        self._reports = reports
        self._dialogues = dialogues
        self._agent = agent

    async def execute(
        self, correction: SoapReportCorrection, *, user_request: str, patient_id: str
    ) -> CorrectionProposal:
        editor = await self._sessions.get_for_correction(
            correction.id
        ) or CorrectionEditorSession.start(correction.id, created_at=_now())
        self._guard_proposable(correction, editor)
        context = await self._build_context(correction, editor, patient_id)
        draft = await self._agent.propose(context, user_request=user_request)
        proposal = editor.propose(
            correction,
            user_request=draft.user_request,
            model_id=draft.model_id,
            prompt_version=draft.prompt_version,
            operations=draft.operations,
            at=_now(),
        )
        await self._sessions.save(editor)
        await self._session.flush()
        return proposal

    def _guard_proposable(
        self, correction: SoapReportCorrection, editor: CorrectionEditorSession
    ) -> None:
        if correction.status is CorrectionStatus.VERIFIED:
            raise CorrectionNotProposable("a verified correction does not accept proposals")
        if editor.active_proposal() is not None:
            raise ActiveProposalExists("a proposal with pending operations already exists")

    async def _build_context(
        self, correction: SoapReportCorrection, editor: CorrectionEditorSession, patient_id: str
    ) -> EditContext:
        report = await self._reports.get(correction.source_report_id)
        dialogue_id = await self._reports.get_dialogue_id(correction.source_report_id)
        if report is None or dialogue_id is None:
            raise SourceReportNotFoundError(correction.source_report_id)
        dialogue = await self._dialogues.get(dialogue_id)
        if dialogue is None:
            raise DialogueNotFoundError(dialogue_id)
        return EditContext(
            dialogue=dialogue,
            report=report,
            correction=correction,
            session=editor,
            patient_id=patient_id,
        )


class AcceptProposalOperation:
    """Accepts one operation and applies it through the #8 correction use cases."""

    def __init__(
        self,
        session: AsyncSession,
        sessions: CorrectionEditorSessionRepository,
        add: AddCorrectedNote,
        update: UpdateCorrectedNote,
        delete: DeleteCorrectedNote,
    ) -> None:
        self._session = session
        self._sessions = sessions
        self._add = add
        self._update = update
        self._delete = delete

    async def execute(
        self, correction: SoapReportCorrection, command: DecideOperationCommand
    ) -> CorrectionProposal:
        editor = await self._require_session(correction.id)
        proposal = _proposal_of(editor, command.proposal_id)
        pending = _pending_operation(proposal, command.operation_id)
        operation = proposal.accept_operation(command.operation_id, correction, at=_now())
        if pending is not None:
            await self._apply(correction, operation)
        await self._sessions.save(editor)
        await self._session.flush()
        return proposal

    async def _require_session(self, correction_id: CorrectionId) -> CorrectionEditorSession:
        editor = await self._sessions.get_for_correction(correction_id)
        if editor is None:
            raise ProposalNotFoundError()
        return editor

    async def _apply(self, correction: SoapReportCorrection, operation: ProposalOperation) -> None:
        if operation.type is OperationType.ADD_NOTE:
            await self._add.execute(
                AddCorrectedNoteCommand(correction_id=correction.id, **_content(operation, None))
            )
            return
        note_id = operation.target_note_id
        if note_id is None:
            raise RuntimeError("an update/delete operation must carry a target note")
        if operation.type is OperationType.UPDATE_NOTE:
            current = correction.find_note(note_id)
            await self._update.execute(
                UpdateCorrectedNoteCommand(
                    correction_id=correction.id, note_id=note_id, **_content(operation, current)
                )
            )
            return
        await self._delete.execute(
            DeleteCorrectedNoteCommand(correction_id=correction.id, note_id=note_id)
        )


class RejectProposalOperation:
    """Rejects one operation, leaving the correction unchanged."""

    def __init__(self, session: AsyncSession, sessions: CorrectionEditorSessionRepository) -> None:
        self._session = session
        self._sessions = sessions

    async def execute(
        self, correction: SoapReportCorrection, command: DecideOperationCommand
    ) -> CorrectionProposal:
        editor = await self._sessions.get_for_correction(correction.id)
        if editor is None:
            raise ProposalNotFoundError(command.proposal_id)
        proposal = _proposal_of(editor, command.proposal_id)
        proposal.reject_operation(command.operation_id, at=_now())
        await self._sessions.save(editor)
        await self._session.flush()
        return proposal


class RejectPendingProposals:
    """Auto-rejects the active proposal's pending operations after a doctor edit."""

    def __init__(self, session: AsyncSession, sessions: CorrectionEditorSessionRepository) -> None:
        self._session = session
        self._sessions = sessions

    async def execute(self, correction: SoapReportCorrection, *, reason: str) -> None:
        editor = await self._sessions.get_for_correction(correction.id)
        if editor is None:
            return
        if editor.reject_pending(at=_now(), reason=reason) is None:
            return
        await self._sessions.save(editor)
        await self._session.flush()


class EnsureNoPendingProposal:
    """Blocks ``verify`` while the active proposal still has pending operations."""

    def __init__(self, sessions: CorrectionEditorSessionRepository) -> None:
        self._sessions = sessions

    async def execute(self, correction: SoapReportCorrection) -> None:
        editor = await self._sessions.get_for_correction(correction.id)
        if editor is not None and editor.active_proposal() is not None:
            raise PendingOperationsBlockVerify(correction.id)


class GetCurrentProposal:
    """Returns the correction's active proposal, or the most recent one."""

    def __init__(self, sessions: CorrectionEditorSessionRepository) -> None:
        self._sessions = sessions

    async def execute(self, correction: SoapReportCorrection) -> CorrectionProposal:
        editor = await self._sessions.get_for_correction(correction.id)
        proposal = _current_proposal(editor)
        if proposal is None:
            raise ProposalNotFoundError()
        return proposal


@dataclass(frozen=True, slots=True)
class AcceptanceMetricQuery:
    """Optional time window (on proposal creation) for the acceptance metric."""

    since: datetime | None = None
    until: datetime | None = None


@dataclass(frozen=True, slots=True)
class OperationCounts:
    """Operation-level tallies; the acceptance rate's denominator is decided ops only."""

    proposed: int
    accepted: int
    rejected: int
    pending: int

    @property
    def acceptance_rate(self) -> float | None:
        decided = self.accepted + self.rejected
        return self.accepted / decided if decided else None


@dataclass(frozen=True, slots=True)
class MetricBreakdown:
    """Acceptance counts for one (model_id, prompt_version) slice of the log."""

    model_id: str
    prompt_version: str
    counts: OperationCounts


@dataclass(frozen=True, slots=True)
class AcceptanceMetric:
    """The online acceptance metric of a correction's editor session."""

    correction_id: CorrectionId
    since: datetime | None
    until: datetime | None
    counts: OperationCounts
    breakdown: list[MetricBreakdown]


class ComputeAcceptanceMetric:
    """Computes operation-level acceptance from the durable proposal/operation log."""

    def __init__(self, sessions: CorrectionEditorSessionRepository) -> None:
        self._sessions = sessions

    async def execute(
        self, correction: SoapReportCorrection, query: AcceptanceMetricQuery
    ) -> AcceptanceMetric:
        since, until = _aware(query.since), _aware(query.until)
        editor = await self._sessions.get_for_correction(correction.id)
        proposals = [
            proposal
            for proposal in (editor.proposals if editor is not None else [])
            if _within(proposal.created_at, since, until)
        ]
        operations = [operation for proposal in proposals for operation in proposal.operations]
        return AcceptanceMetric(
            correction_id=correction.id,
            since=since,
            until=until,
            counts=_counts(operations),
            breakdown=_breakdown(proposals),
        )


def _proposal_of(editor: CorrectionEditorSession, proposal_id: ProposalId) -> CorrectionProposal:
    for proposal in editor.proposals:
        if proposal.id == proposal_id:
            return proposal
    raise ProposalNotFoundError(proposal_id)


def _pending_operation(
    proposal: CorrectionProposal, operation_id: OperationId
) -> ProposalOperation | None:
    operation = next((op for op in proposal.operations if op.id == operation_id), None)
    return operation if operation is not None and operation.is_pending() else None


def _current_proposal(editor: CorrectionEditorSession | None) -> CorrectionProposal | None:
    if editor is None or not editor.proposals:
        return None
    return editor.active_proposal() or editor.proposals[-1]


def _content(operation: ProposalOperation, current: CorrectedNote | None) -> dict[str, Any]:
    proposed = operation.proposed
    if proposed is None:
        raise RuntimeError("an add/update operation must carry proposed content")
    icds = [claim.icd for claim in current.assessment] if current is not None else []
    return {
        "subjective": [_to_claim(claim) for claim in proposed.subjective],
        "objective": [_to_claim(claim) for claim in proposed.objective],
        "assessment": [
            _to_assessment(claim, icds[index] if index < len(icds) else None)
            for index, claim in enumerate(proposed.assessment)
        ],
        "plan": [_to_claim(claim) for claim in proposed.plan],
    }


def _to_claim(claim: ProposedClaim) -> SoapClaim:
    return SoapClaim(id=Id.new(), text=claim.text, citations=list(claim.citations))


def _to_assessment(claim: ProposedClaim, icd: IcdCoding | None) -> AssessmentClaim:
    return AssessmentClaim(id=Id.new(), text=claim.text, citations=list(claim.citations), icd=icd)


def _counts(operations: Sequence[ProposalOperation]) -> OperationCounts:
    return OperationCounts(
        proposed=len(operations),
        accepted=_tally(operations, OperationDecision.ACCEPTED),
        rejected=_tally(operations, OperationDecision.REJECTED),
        pending=_tally(operations, OperationDecision.PENDING),
    )


def _tally(operations: Iterable[ProposalOperation], decision: OperationDecision) -> int:
    return sum(1 for operation in operations if operation.decision is decision)


def _breakdown(proposals: Sequence[CorrectionProposal]) -> list[MetricBreakdown]:
    grouped: dict[tuple[str, str], list[ProposalOperation]] = {}
    for proposal in proposals:
        key = (proposal.model_id, proposal.prompt_version)
        grouped.setdefault(key, []).extend(proposal.operations)
    return [
        MetricBreakdown(
            model_id=model_id, prompt_version=prompt_version, counts=_counts(operations)
        )
        for (model_id, prompt_version), operations in sorted(grouped.items())
    ]


def _within(moment: datetime, since: datetime | None, until: datetime | None) -> bool:
    if since is not None and moment < since:
        return False
    return not (until is not None and moment > until)


def _aware(moment: datetime | None) -> datetime | None:
    if moment is None:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
