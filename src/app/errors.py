"""HTTP error mapping for SOAP correction and online-quality workflows.

Each domain and use-case error is translated into a stable JSON body
``{"code": ..., "detail": ...}`` with a machine-readable ``code`` and the right
status. The #7 endpoints are untouched: they raise ``HTTPException`` directly and
keep FastAPI's default error shape.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

from ehr.publication import (
    InvalidFhirReference,
    PublicationInvalidTransition,
    PublicationRequiresVerifiedCorrection,
    SnapshotIntegrityError,
)
from ehr.publication_use_cases import (
    PublicationCorrectionNotFoundError,
    PublicationNotFoundError,
    PublicationOutboxNotFoundError,
    PublicationSourceDialogueNotFoundError,
    PublicationSourceReportNotFoundError,
)
from icd.resolver import IcdTitleMismatch, InactiveIcdCode, UnknownIcdCode
from soap.correction import (
    CorrectionInvalidTransition,
    CorrectionNotEditable,
    DuplicateSourceNote,
    EmptyDoctorId,
    NoteNotInCorrection,
)
from soap.correction_use_cases import (
    CitationNotInSourceDialogue,
    CorrectionNotFoundError,
    SourceReportNotFoundError,
)
from soap.editor_use_cases import PendingOperationsBlockVerify, ProposalNotFoundError
from soap.llm_editor import InvalidProposalError, SoapEditError
from soap.proposal import (
    ActiveProposalExists,
    ConflictingDecision,
    CorrectionNotProposable,
    DuplicateOperationTarget,
    EmptyProposal,
    OperationNotInProposal,
    StaleOperationTarget,
    UnknownProposalTarget,
)
from soap.quality import CorrectionNotVerifiedError
from soap.quality_use_cases import (
    DialogueReportNotFoundError,
    QualityCorrectionNotFoundError,
)

Handler = Callable[[Request, Exception], Awaitable[JSONResponse]]


class CorrectionForReportNotFound(Exception):
    """Raised when a report has no correction yet — resolving one by report id fails."""

    def __init__(self, report_id: object) -> None:
        super().__init__(f"report {report_id} has no correction")
        self.report_id = report_id


_ERROR_MAP: list[tuple[type[Exception], int, str]] = [
    (DialogueReportNotFoundError, 404, "report_not_found"),
    (QualityCorrectionNotFoundError, 404, "correction_not_found"),
    (CorrectionNotVerifiedError, 409, "REPORT_NOT_VERIFIED"),
    (SourceReportNotFoundError, 404, "report_not_found"),
    (CorrectionForReportNotFound, 404, "correction_not_found"),
    (CorrectionNotFoundError, 404, "correction_not_found"),
    (NoteNotInCorrection, 404, "note_not_found"),
    (CorrectionNotEditable, 409, "correction_not_editable"),
    (CitationNotInSourceDialogue, 422, "citation_not_in_source_dialogue"),
    (EmptyDoctorId, 422, "empty_doctor_id"),
    (DuplicateSourceNote, 422, "duplicate_source_note"),
    (UnknownIcdCode, 422, "unknown_icd_code"),
    (InactiveIcdCode, 422, "inactive_icd_code"),
    (IcdTitleMismatch, 422, "icd_title_mismatch"),
    (ProposalNotFoundError, 404, "proposal_not_found"),
    (OperationNotInProposal, 404, "operation_not_found"),
    (ActiveProposalExists, 409, "active_proposal_exists"),
    (CorrectionNotProposable, 409, "correction_not_proposable"),
    (StaleOperationTarget, 409, "stale_operation_target"),
    (ConflictingDecision, 409, "conflicting_decision"),
    (PendingOperationsBlockVerify, 409, "pending_operations_block_verify"),
    (InvalidProposalError, 422, "invalid_generated_content"),
    (EmptyProposal, 422, "invalid_generated_content"),
    (UnknownProposalTarget, 422, "invalid_generated_content"),
    (DuplicateOperationTarget, 422, "invalid_generated_content"),
    # Base editor error = a genuine model/transport failure (its InvalidProposalError
    # subclass is matched above as 422); surface it as an upstream outage, not a 500.
    (SoapEditError, 503, "editor_unavailable"),
    (PublicationNotFoundError, 404, "publication_not_found"),
    (PublicationCorrectionNotFoundError, 404, "correction_not_found"),
    (PublicationSourceReportNotFoundError, 404, "report_not_found"),
    (PublicationSourceDialogueNotFoundError, 404, "dialogue_not_found"),
    (PublicationRequiresVerifiedCorrection, 409, "report_not_verified"),
    (CorrectionInvalidTransition, 409, "invalid_correction_transition"),
    (PublicationInvalidTransition, 409, "invalid_publication_transition"),
    (InvalidFhirReference, 422, "invalid_fhir_reference"),
    (PublicationOutboxNotFoundError, 500, "publication_outbox_missing"),
    (SnapshotIntegrityError, 500, "publication_snapshot_invalid"),
]


def _handler(status_code: int, code: str) -> Handler:
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"code": code, "detail": str(exc)}, status_code=status_code)

    return handle


def register_exception_handlers(app: FastAPI) -> None:
    """Map workflow error types to their statuses and stable machine codes."""
    for exc_type, status_code, code in _ERROR_MAP:
        app.add_exception_handler(exc_type, _handler(status_code, code))
