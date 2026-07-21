"""Composition helpers for EHR publication persistence adapters."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from .publication_repository import (
    EhrPublicationRepository,
    PublicationOutboxRepository,
)
from .publication_sqlalchemy_repository import (
    SqlAlchemyEhrPublicationRepository,
    SqlAlchemyPublicationOutboxRepository,
)


def get_ehr_publication_repository(
    session: AsyncSession,
) -> EhrPublicationRepository:
    return SqlAlchemyEhrPublicationRepository(session)


def get_publication_outbox_repository(
    session: AsyncSession,
) -> PublicationOutboxRepository:
    return SqlAlchemyPublicationOutboxRepository(session)
