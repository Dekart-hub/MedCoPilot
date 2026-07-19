"""soap report correction aggregate

Revision ID: 0004_soap_correction
Revises: 0003_soap_report
Create Date: 2026-07-19

Creates the doctor-correction aggregate tables on top of the untouched
``soap_report`` tables: the ``soap_correction`` root (unique per
``source_report_id`` and cascade-deleted with its source report), its ordered
``soap_corrected_note`` children and their ``soap_corrected_claim``
grandchildren, each generation cascade-deleted with its parent. A corrected
note keeps a nullable ``source_note_id`` (a plain uuid, not an FK, so lineage
survives even if the source note is later removed); a non-``NULL``
``source_note_id`` is unique within a correction via a partial unique index.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_soap_correction"
down_revision: str | Sequence[str] | None = "0003_soap_report"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "soap_correction",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_report_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("verified_by", sa.String(length=255), nullable=True),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_report_id"], ["soap_report.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_report_id", name="uq_soap_correction_source_report_id"),
    )
    op.create_table(
        "soap_corrected_note",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("correction_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("source_note_id", sa.Uuid(), nullable=True),
        sa.ForeignKeyConstraint(["correction_id"], ["soap_correction.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_soap_corrected_note_correction_id"),
        "soap_corrected_note",
        ["correction_id"],
        unique=False,
    )
    op.create_index(
        "uq_soap_corrected_note_source",
        "soap_corrected_note",
        ["correction_id", "source_note_id"],
        unique=True,
        postgresql_where=sa.text("source_note_id IS NOT NULL"),
    )
    op.create_table(
        "soap_corrected_claim",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("note_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(length=32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("icd_code", sa.String(length=32), nullable=True),
        sa.Column("icd_name", sa.Text(), nullable=True),
        sa.Column("icd_classifier_url", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["note_id"], ["soap_corrected_note.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_soap_corrected_claim_note_id"),
        "soap_corrected_claim",
        ["note_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_soap_corrected_claim_note_id"), table_name="soap_corrected_claim")
    op.drop_table("soap_corrected_claim")
    op.drop_index("uq_soap_corrected_note_source", table_name="soap_corrected_note")
    op.drop_index(op.f("ix_soap_corrected_note_correction_id"), table_name="soap_corrected_note")
    op.drop_table("soap_corrected_note")
    op.drop_table("soap_correction")
