"""SQLAlchemy ORM tables for the SoapReport aggregate.

These rows live in the infrastructure layer: they register the ``soap_report``,
``soap_note`` and ``soap_claim`` tables on ``Base.metadata`` (so Alembic
autogenerate sees them) and are mapped to/from the pure domain aggregate by the
repository adapter. The domain classes stay free of any ORM concern.

``soap_report.dialogue_id`` is unique: it ties a report to the dialogue it was
extracted from and is the idempotency key one report per dialogue rests on.
Claim ordering within a note is captured by ``position`` (assigned in canonical
S/O/A/P section order), and each claim records its ``section`` so the note's
four lists can be rebuilt on load.
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    JSON,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from infra.db import Base


class SoapReportRow(Base):
    __tablename__ = "soap_report"
    __table_args__ = (UniqueConstraint("dialogue_id", name="uq_soap_report_dialogue_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    dialogue_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dialogue.id", ondelete="CASCADE"))
    notes: Mapped[list[SoapNoteRow]] = relationship(
        back_populates="report",
        cascade="all, delete-orphan",
        order_by="SoapNoteRow.position",
    )


class SoapNoteRow(Base):
    __tablename__ = "soap_note"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    report_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("soap_report.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column()
    confidence: Mapped[float | None] = mapped_column()
    report: Mapped[SoapReportRow] = relationship(back_populates="notes")
    claims: Mapped[list[SoapClaimRow]] = relationship(
        back_populates="note",
        cascade="all, delete-orphan",
        order_by="SoapClaimRow.position",
    )


class SoapClaimRow(Base):
    __tablename__ = "soap_claim"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    note_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("soap_note.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column()
    section: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(Text())
    citations: Mapped[list[dict[str, str | None]]] = mapped_column(JSON())
    icd_code: Mapped[str | None] = mapped_column(String(32))
    icd_name: Mapped[str | None] = mapped_column(Text())
    icd_classifier_url: Mapped[str | None] = mapped_column(Text())
    note: Mapped[SoapNoteRow] = relationship(back_populates="claims")
