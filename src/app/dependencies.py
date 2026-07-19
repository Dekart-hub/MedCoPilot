"""FastAPI dependency providers wiring use cases to a request-scoped session.

Each collaborator is exposed as its own dependency so tests can override the
pieces in isolation — above all the SOAP extractor, which must never reach a
live LLM in CI. The heavy, read-only singletons (the LLM extractor and the mock
EHR client) are built once and cached; the repositories and use cases are cheap
and built per request around the caller's session.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from dialogue.providers import get_dialogue_repository as _build_dialogue_repository
from dialogue.repository import DialogueRepository
from dialogue.use_cases import AddDialogue
from ehr.client import EhrClient
from icd.bm25_coder import Bm25IcdCoder
from infra.db import get_session
from infra.ehr import build_ehr_client
from infra.llm import build_llm_extractor
from soap.correction_repository import SoapReportCorrectionRepository
from soap.correction_use_cases import (
    AddCorrectedNote,
    DeleteCorrectedNote,
    ReopenSoapCorrection,
    StartSoapCorrection,
    UpdateCorrectedNote,
    VerifySoapCorrection,
)
from soap.extractor import SoapExtractor
from soap.providers import (
    get_soap_report_correction_repository as _build_correction_repository,
)
from soap.providers import get_soap_report_repository as _build_report_repository
from soap.repository import SoapReportRepository
from soap.use_cases import ExtractSoapReport

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_dialogue_repository(session: SessionDep) -> DialogueRepository:
    return _build_dialogue_repository(session)


def get_add_dialogue(
    repository: Annotated[DialogueRepository, Depends(get_dialogue_repository)],
) -> AddDialogue:
    return AddDialogue(repository)


def get_soap_report_repository(session: SessionDep) -> SoapReportRepository:
    return _build_report_repository(session)


@lru_cache
def _extractor() -> SoapExtractor:
    return build_llm_extractor(get_settings(), coder=Bm25IcdCoder.from_bundled())


def get_soap_extractor() -> SoapExtractor:
    return _extractor()


@lru_cache
def _ehr_client() -> EhrClient:
    return build_ehr_client(get_settings())


def get_ehr_client() -> EhrClient:
    return _ehr_client()


def get_extract_soap_report(
    session: SessionDep,
    dialogues: Annotated[DialogueRepository, Depends(get_dialogue_repository)],
    reports: Annotated[SoapReportRepository, Depends(get_soap_report_repository)],
    extractor: Annotated[SoapExtractor, Depends(get_soap_extractor)],
    ehr: Annotated[EhrClient, Depends(get_ehr_client)],
) -> ExtractSoapReport:
    return ExtractSoapReport(session, dialogues, reports, extractor, ehr)


def get_correction_repository(session: SessionDep) -> SoapReportCorrectionRepository:
    return _build_correction_repository(session)


CorrectionRepositoryDep = Annotated[
    SoapReportCorrectionRepository, Depends(get_correction_repository)
]
ReportRepositoryDep = Annotated[SoapReportRepository, Depends(get_soap_report_repository)]
DialogueRepositoryDep = Annotated[DialogueRepository, Depends(get_dialogue_repository)]


def get_start_soap_correction(
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    reports: ReportRepositoryDep,
) -> StartSoapCorrection:
    return StartSoapCorrection(session, corrections, reports)


def get_add_corrected_note(
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    reports: ReportRepositoryDep,
    dialogues: DialogueRepositoryDep,
) -> AddCorrectedNote:
    return AddCorrectedNote(session, corrections, reports, dialogues)


def get_update_corrected_note(
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
    reports: ReportRepositoryDep,
    dialogues: DialogueRepositoryDep,
) -> UpdateCorrectedNote:
    return UpdateCorrectedNote(session, corrections, reports, dialogues)


def get_delete_corrected_note(
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
) -> DeleteCorrectedNote:
    return DeleteCorrectedNote(session, corrections)


def get_verify_soap_correction(
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
) -> VerifySoapCorrection:
    return VerifySoapCorrection(session, corrections)


def get_reopen_soap_correction(
    session: SessionDep,
    corrections: CorrectionRepositoryDep,
) -> ReopenSoapCorrection:
    return ReopenSoapCorrection(session, corrections)
