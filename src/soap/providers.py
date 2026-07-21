"""Composition helpers wiring the SoapReport repository to a database session.

Framework-agnostic: the provider takes an :class:`AsyncSession` and returns a
ready-to-use adapter. The FastAPI dependency layer (:mod:`app.dependencies`)
adapts this into a request-scoped dependency and composes the use case around it.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from .correction_repository import SoapReportCorrectionRepository
from .correction_sqlalchemy_repository import SqlAlchemySoapReportCorrectionRepository
from .proposal_repository import CorrectionEditorSessionRepository
from .proposal_sqlalchemy_repository import SqlAlchemyCorrectionEditorSessionRepository
from .repository import SoapReportRepository
from .sqlalchemy_repository import SqlAlchemySoapReportRepository


def get_soap_report_repository(session: AsyncSession) -> SoapReportRepository:
    return SqlAlchemySoapReportRepository(session)


def get_soap_report_correction_repository(
    session: AsyncSession,
) -> SoapReportCorrectionRepository:
    return SqlAlchemySoapReportCorrectionRepository(session)


def get_correction_editor_session_repository(
    session: AsyncSession,
) -> CorrectionEditorSessionRepository:
    return SqlAlchemyCorrectionEditorSessionRepository(session)
