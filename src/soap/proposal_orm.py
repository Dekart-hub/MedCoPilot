"""SQLAlchemy ORM tables for the LLM editor-session aggregate.

These rows register the ``soap_correction_editor_session``,
``soap_correction_proposal`` and ``soap_correction_proposal_operation`` tables on
``Base.metadata`` (so Alembic autogenerate sees them) and are mapped to/from the
pure :mod:`soap.proposal` aggregate by the repository adapter. The domain classes
stay free of any ORM concern.

A session is unique per ``correction_id`` and cascade-deleted with its
correction: one LLM-editing session per correction, gone when the correction is.
Proposals are ordered children (``position``) and their operations are ordered
grandchildren, each generation cascade-deleted with its parent so a superseded
proposal keeps every operation, decision and snapshot. Within one proposal a
non-``NULL`` ``target_note_id`` is unique (partial unique index), mirroring the
domain's one-operation-per-note invariant. ``proposed_content`` and
``before_snapshot`` are JSON blobs; ``target_fingerprint`` is the content
fingerprint replayed to detect a stale target.
"""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from infra.db import Base


class SoapCorrectionEditorSessionRow(Base):
    __tablename__ = "soap_correction_editor_session"
    __table_args__ = (
        UniqueConstraint("correction_id", name="uq_soap_editor_session_correction_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    correction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("soap_correction.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    proposals: Mapped[list[SoapCorrectionProposalRow]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="SoapCorrectionProposalRow.position",
    )


class SoapCorrectionProposalRow(Base):
    __tablename__ = "soap_correction_proposal"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("soap_correction_editor_session.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer())
    user_request: Mapped[str] = mapped_column(Text())
    base_correction_revision: Mapped[int] = mapped_column(Integer())
    model_id: Mapped[str] = mapped_column(String(255))
    prompt_version: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    session: Mapped[SoapCorrectionEditorSessionRow] = relationship(back_populates="proposals")
    operations: Mapped[list[SoapCorrectionProposalOperationRow]] = relationship(
        back_populates="proposal",
        cascade="all, delete-orphan",
        order_by="SoapCorrectionProposalOperationRow.position",
    )


class SoapCorrectionProposalOperationRow(Base):
    __tablename__ = "soap_correction_proposal_operation"
    __table_args__ = (
        Index(
            "uq_soap_proposal_operation_target",
            "proposal_id",
            "target_note_id",
            unique=True,
            postgresql_where=text("target_note_id IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    proposal_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("soap_correction_proposal.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer())
    type: Mapped[str] = mapped_column(String(32))
    decision: Mapped[str] = mapped_column(String(32))
    target_note_id: Mapped[uuid.UUID | None] = mapped_column()
    proposed_content: Mapped[dict[str, object] | None] = mapped_column(JSON())
    before_snapshot: Mapped[dict[str, object] | None] = mapped_column(JSON())
    target_fingerprint: Mapped[str | None] = mapped_column(String(64))
    decided_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    decision_reason: Mapped[str | None] = mapped_column(Text())
    proposal: Mapped[SoapCorrectionProposalRow] = relationship(back_populates="operations")
