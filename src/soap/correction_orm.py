"""SQLAlchemy ORM tables for the SoapReportCorrection aggregate.

These rows live in the infrastructure layer: they register the
``soap_correction``, ``soap_corrected_note`` and ``soap_corrected_claim`` tables
on ``Base.metadata`` (so Alembic autogenerate sees them) and are mapped to/from
the pure domain aggregate by the repository adapter. The domain classes stay
free of any ORM concern. The original ``soap_report`` tables are never touched.

``soap_correction.source_report_id`` is unique and cascades from the source
report: at most one doctor version per report, gone when the report is. A
corrected note keeps ``source_note_id`` — the id of the original note it was
copied from, ``NULL`` for a doctor-added note. That column is a plain nullable
uuid, deliberately *not* a foreign key to ``soap_note``: lineage must survive a
correction whose source note was later removed, so no DB-level FK constrains it.
A non-``NULL`` ``source_note_id`` is unique within a correction (partial unique
index), matching the domain's one-source-note-per-correction invariant. Claim
ordering within a note is captured by ``position`` and ``section`` so the four
S/O/A/P lists rebuild in order, exactly as in ``soap.orm``.
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


class SoapCorrectionRow(Base):
    __tablename__ = "soap_correction"
    __table_args__ = (
        UniqueConstraint("source_report_id", name="uq_soap_correction_source_report_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    source_report_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("soap_report.id", ondelete="CASCADE")
    )
    status: Mapped[str] = mapped_column(String(32))
    verified_by: Mapped[str | None] = mapped_column(String(255))
    verified_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    revision: Mapped[int] = mapped_column(Integer())
    notes: Mapped[list[SoapCorrectedNoteRow]] = relationship(
        back_populates="correction",
        cascade="all, delete-orphan",
        order_by="SoapCorrectedNoteRow.position",
    )


class SoapCorrectedNoteRow(Base):
    __tablename__ = "soap_corrected_note"
    __table_args__ = (
        Index(
            "uq_soap_corrected_note_source",
            "correction_id",
            "source_note_id",
            unique=True,
            postgresql_where=text("source_note_id IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    correction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("soap_correction.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column()
    source_note_id: Mapped[uuid.UUID | None] = mapped_column()
    correction: Mapped[SoapCorrectionRow] = relationship(back_populates="notes")
    claims: Mapped[list[SoapCorrectedClaimRow]] = relationship(
        back_populates="note",
        cascade="all, delete-orphan",
        order_by="SoapCorrectedClaimRow.position",
    )


class SoapCorrectedClaimRow(Base):
    __tablename__ = "soap_corrected_claim"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    note_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("soap_corrected_note.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column()
    section: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(Text())
    citations: Mapped[list[dict[str, str | None]]] = mapped_column(JSON())
    icd_code: Mapped[str | None] = mapped_column(String(32))
    icd_name: Mapped[str | None] = mapped_column(Text())
    icd_classifier_url: Mapped[str | None] = mapped_column(Text())
    note: Mapped[SoapCorrectedNoteRow] = relationship(back_populates="claims")
