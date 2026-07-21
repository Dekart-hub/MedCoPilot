"""SQLAlchemy rows for immutable EHR publications and their durable outbox."""

from __future__ import annotations

import datetime
import uuid
from typing import Any

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
from sqlalchemy.orm import Mapped, mapped_column

from infra.db import Base


class EhrPublicationRow(Base):
    __tablename__ = "ehr_publication"
    __table_args__ = (
        UniqueConstraint("correction_id", name="uq_ehr_publication_correction_id"),
        UniqueConstraint("source_report_id", name="uq_ehr_publication_source_report_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    source_report_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("soap_report.id", ondelete="RESTRICT")
    )
    correction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("soap_correction.id", ondelete="RESTRICT")
    )
    patient_ref: Mapped[str] = mapped_column(String(255))
    encounter_ref: Mapped[str] = mapped_column(String(255))
    author_ref: Mapped[str] = mapped_column(String(255))
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSON())
    snapshot_hash: Mapped[str] = mapped_column(String(64))
    snapshot_schema_version: Mapped[int] = mapped_column(Integer())
    status: Mapped[str] = mapped_column(String(32))
    remote_reference: Mapped[str | None] = mapped_column(String(255))
    remote_version: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))


class PublicationOutboxRow(Base):
    __tablename__ = "ehr_publication_outbox"
    __table_args__ = (
        UniqueConstraint("publication_id", name="uq_ehr_publication_outbox_publication_id"),
        Index(
            "ix_ehr_publication_outbox_due",
            "next_attempt_at",
            postgresql_where=text("delivered_at IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    publication_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("ehr_publication.id", ondelete="RESTRICT")
    )
    payload: Mapped[str] = mapped_column(Text())
    payload_hash: Mapped[str] = mapped_column(String(64))
    attempt_count: Mapped[int] = mapped_column(Integer())
    next_attempt_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text())
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
