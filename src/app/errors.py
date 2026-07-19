"""HTTP error mapping for the SOAP-correction workflow.

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

from soap.correction import (
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

Handler = Callable[[Request, Exception], Awaitable[JSONResponse]]


class CorrectionForReportNotFound(Exception):
    """Raised when a report has no correction yet — resolving one by report id fails."""

    def __init__(self, report_id: object) -> None:
        super().__init__(f"report {report_id} has no correction")
        self.report_id = report_id


_ERROR_MAP: list[tuple[type[Exception], int, str]] = [
    (SourceReportNotFoundError, 404, "report_not_found"),
    (CorrectionForReportNotFound, 404, "correction_not_found"),
    (CorrectionNotFoundError, 404, "correction_not_found"),
    (NoteNotInCorrection, 404, "note_not_found"),
    (CorrectionNotEditable, 409, "correction_not_editable"),
    (CitationNotInSourceDialogue, 422, "citation_not_in_source_dialogue"),
    (EmptyDoctorId, 422, "empty_doctor_id"),
    (DuplicateSourceNote, 422, "duplicate_source_note"),
]


def _handler(status_code: int, code: str) -> Handler:
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"code": code, "detail": str(exc)}, status_code=status_code)

    return handle


def register_exception_handlers(app: FastAPI) -> None:
    """Map every correction error type to its status and stable ``code``."""
    for exc_type, status_code, code in _ERROR_MAP:
        app.add_exception_handler(exc_type, _handler(status_code, code))
