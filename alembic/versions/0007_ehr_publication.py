"""EHR publication snapshot and durable outbox

Revision ID: 0007_ehr_publication
Revises: 0006_llm_correction_proposal
Create Date: 2026-07-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_ehr_publication"
down_revision: str | Sequence[str] | None = "0006_llm_correction_proposal"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ehr_publication",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_report_id", sa.Uuid(), nullable=False),
        sa.Column("correction_id", sa.Uuid(), nullable=False),
        sa.Column("patient_ref", sa.String(length=255), nullable=False),
        sa.Column("encounter_ref", sa.String(length=255), nullable=False),
        sa.Column("author_ref", sa.String(length=255), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("snapshot_hash", sa.String(length=64), nullable=False),
        sa.Column("snapshot_schema_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("remote_reference", sa.String(length=255), nullable=True),
        sa.Column("remote_version", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["correction_id"], ["soap_correction.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["source_report_id"], ["soap_report.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "correction_id", name="uq_ehr_publication_correction_id"
        ),
        sa.UniqueConstraint(
            "source_report_id", name="uq_ehr_publication_source_report_id"
        ),
    )
    op.create_table(
        "ehr_publication_outbox",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("publication_id", sa.Uuid(), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["publication_id"], ["ehr_publication.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "publication_id", name="uq_ehr_publication_outbox_publication_id"
        ),
    )
    op.create_index(
        "ix_ehr_publication_outbox_due",
        "ehr_publication_outbox",
        ["next_attempt_at"],
        unique=False,
        postgresql_where=sa.text("delivered_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ehr_publication_outbox_due",
        table_name="ehr_publication_outbox",
    )
    op.drop_table("ehr_publication_outbox")
    op.drop_table("ehr_publication")
