"""Application use cases for the SOAP-correction workflow (story #8).

A doctor works one editable version on top of an immutable source report. The
:class:`~soap.correction.SoapReportCorrection` aggregate owns the rules — DRAFT
vs VERIFIED, lineage, editability — while these use cases orchestrate loading,
citation grounding and persistence around it.

``StartSoapCorrection`` is idempotent, exactly like
:class:`~soap.use_cases.ExtractSoapReport`: one correction per source report,
enforced by the unique constraint on ``source_report_id``. A repeat call
short-circuits to the stored correction; a concurrent create that races past the
pre-check trips the constraint on flush, is rolled back and resolves to the
winner.

Every added or updated note must stay grounded in the source dialogue: each
citation's ``turn_id`` is checked against the real turns of the dialogue the
source report was extracted from, and a stray citation is rejected (→ 422).

Each successful mutation persists the intermediate state (``save`` + ``flush``);
the caller (T21's API) owns the commit. Timestamps come from the application
clock and are injected into the pure domain, which never reads the wall clock.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from dialogue.dialogue import DialogueTurnId
from dialogue.repository import DialogueRepository

from .correction import CorrectedNote, CorrectionId, SoapReportCorrection
from .correction_repository import SoapReportCorrectionRepository
from .repository import SoapReportRepository
from .soap import AssessmentClaim, SoapClaim, SoapNoteId, SoapReportId
from .use_cases import DialogueNotFoundError


class SourceReportNotFoundError(Exception):
    """Raised when the source report to correct does not exist."""

    def __init__(self, report_id: SoapReportId) -> None:
        super().__init__(f"soap report {report_id} not found")
        self.report_id = report_id


class CorrectionNotFoundError(Exception):
    """Raised when the correction to edit does not exist."""

    def __init__(self, correction_id: CorrectionId) -> None:
        super().__init__(f"soap correction {correction_id} not found")
        self.correction_id = correction_id


class CitationNotInSourceDialogue(Exception):
    """Raised when a note cites a turn absent from the source report's dialogue."""

    def __init__(self, turn_id: DialogueTurnId) -> None:
        super().__init__(f"citation turn {turn_id} is not part of the source dialogue")
        self.turn_id = turn_id


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class StartSoapCorrectionCommand:
    """Request to open, or resume, the doctor's correction of a source report."""

    report_id: SoapReportId


@dataclass(frozen=True, slots=True)
class AddCorrectedNoteCommand:
    """Request to add a doctor-authored note (no source lineage) to a correction."""

    correction_id: CorrectionId
    subjective: list[SoapClaim] = field(default_factory=list)
    objective: list[SoapClaim] = field(default_factory=list)
    assessment: list[AssessmentClaim] = field(default_factory=list)
    plan: list[SoapClaim] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class UpdateCorrectedNoteCommand:
    """Request to replace a corrected note's four sections, citations and ICD."""

    correction_id: CorrectionId
    note_id: SoapNoteId
    subjective: list[SoapClaim] = field(default_factory=list)
    objective: list[SoapClaim] = field(default_factory=list)
    assessment: list[AssessmentClaim] = field(default_factory=list)
    plan: list[SoapClaim] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DeleteCorrectedNoteCommand:
    """Request to drop a note from the doctor's version."""

    correction_id: CorrectionId
    note_id: SoapNoteId


@dataclass(frozen=True, slots=True)
class VerifySoapCorrectionCommand:
    """Request to move a correction DRAFT → VERIFIED under a doctor's name."""

    correction_id: CorrectionId
    doctor_id: str


@dataclass(frozen=True, slots=True)
class ReopenSoapCorrectionCommand:
    """Request to move a correction VERIFIED → DRAFT for further edits."""

    correction_id: CorrectionId


class _CorrectionUseCase:
    """Shared plumbing: load an existing correction, persist an intermediate state."""

    def __init__(self, session: AsyncSession, corrections: SoapReportCorrectionRepository) -> None:
        self._session = session
        self._corrections = corrections

    async def _load(self, correction_id: CorrectionId) -> SoapReportCorrection:
        correction = await self._corrections.get(correction_id)
        if correction is None:
            raise CorrectionNotFoundError(correction_id)
        return correction

    async def _persist(self, correction: SoapReportCorrection) -> None:
        await self._corrections.save(correction)
        await self._session.flush()


class StartSoapCorrection(_CorrectionUseCase):
    """Opens the doctor's DRAFT correction of a report, or resumes the existing one."""

    def __init__(
        self,
        session: AsyncSession,
        corrections: SoapReportCorrectionRepository,
        reports: SoapReportRepository,
    ) -> None:
        super().__init__(session, corrections)
        self._reports = reports

    async def execute(self, command: StartSoapCorrectionCommand) -> SoapReportCorrection:
        existing = await self._corrections.get_by_source_report_id(command.report_id)
        if existing is not None:
            return existing
        report = await self._reports.get(command.report_id)
        if report is None:
            raise SourceReportNotFoundError(command.report_id)
        correction = SoapReportCorrection.start(report, created_at=_now())
        return await self._create(correction)

    async def _create(self, correction: SoapReportCorrection) -> SoapReportCorrection:
        # Flush now so a lost race trips the unique constraint here, inside the
        # guard, rather than later in the caller's commit; the caller still owns
        # the commit.
        try:
            await self._corrections.save(correction)
            await self._session.flush()
        except IntegrityError:
            await self._session.rollback()
            winner = await self._corrections.get_by_source_report_id(correction.source_report_id)
            if winner is None:
                raise
            return winner
        return correction


class _EditingUseCase(_CorrectionUseCase):
    """Base for edits that must keep every citation grounded in the source dialogue."""

    def __init__(
        self,
        session: AsyncSession,
        corrections: SoapReportCorrectionRepository,
        reports: SoapReportRepository,
        dialogues: DialogueRepository,
    ) -> None:
        super().__init__(session, corrections)
        self._reports = reports
        self._dialogues = dialogues

    async def _reject_ungrounded(
        self, correction: SoapReportCorrection, *sections: Sequence[SoapClaim]
    ) -> None:
        grounded = await self._source_turn_ids(correction)
        for claims in sections:
            for claim in claims:
                for citation in claim.citations:
                    if citation.turn_id not in grounded:
                        raise CitationNotInSourceDialogue(citation.turn_id)

    async def _source_turn_ids(self, correction: SoapReportCorrection) -> set[DialogueTurnId]:
        dialogue_id = await self._reports.get_dialogue_id(correction.source_report_id)
        if dialogue_id is None:
            raise SourceReportNotFoundError(correction.source_report_id)
        dialogue = await self._dialogues.get(dialogue_id)
        if dialogue is None:
            raise DialogueNotFoundError(dialogue_id)
        return {turn.id for turn in dialogue.turns}


class AddCorrectedNote(_EditingUseCase):
    """Adds a doctor-authored note to a DRAFT correction."""

    async def execute(self, command: AddCorrectedNoteCommand) -> CorrectedNote:
        correction = await self._load(command.correction_id)
        await self._reject_ungrounded(
            correction, command.subjective, command.objective, command.assessment, command.plan
        )
        note = correction.add_note(
            at=_now(),
            subjective=command.subjective,
            objective=command.objective,
            assessment=command.assessment,
            plan=command.plan,
        )
        await self._persist(correction)
        return note


class UpdateCorrectedNote(_EditingUseCase):
    """Replaces a corrected note's sections, citations and Assessment ICD."""

    async def execute(self, command: UpdateCorrectedNoteCommand) -> CorrectedNote:
        correction = await self._load(command.correction_id)
        await self._reject_ungrounded(
            correction, command.subjective, command.objective, command.assessment, command.plan
        )
        note = correction.update_note(
            command.note_id,
            at=_now(),
            subjective=command.subjective,
            objective=command.objective,
            assessment=command.assessment,
            plan=command.plan,
        )
        await self._persist(correction)
        return note


class DeleteCorrectedNote(_CorrectionUseCase):
    """Removes a note from the doctor's version of a DRAFT correction."""

    async def execute(self, command: DeleteCorrectedNoteCommand) -> None:
        correction = await self._load(command.correction_id)
        correction.delete_note(command.note_id, at=_now())
        await self._persist(correction)


class VerifySoapCorrection(_CorrectionUseCase):
    """Moves a correction DRAFT → VERIFIED, stamping the checking doctor."""

    async def execute(self, command: VerifySoapCorrectionCommand) -> SoapReportCorrection:
        correction = await self._load(command.correction_id)
        correction.verify(command.doctor_id, at=_now())
        await self._persist(correction)
        return correction


class ReopenSoapCorrection(_CorrectionUseCase):
    """Moves a correction VERIFIED → DRAFT so the doctor can edit again."""

    async def execute(self, command: ReopenSoapCorrectionCommand) -> SoapReportCorrection:
        correction = await self._load(command.correction_id)
        correction.reopen(at=_now())
        await self._persist(correction)
        return correction
