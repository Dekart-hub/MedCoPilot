"""soap report aggregate

Revision ID: 0003_soap_report
Revises: 0002_dialogue
Create Date: 2026-07-15

Creates the ``soap_report`` aggregate tables: the report root (unique per
``dialogue_id`` — the idempotency key), its ordered ``soap_note`` children and
their ``soap_claim`` grandchildren, each generation cascade-deleted with its
parent. A claim records its ``section`` and ``position`` so the note's four
S/O/A/P lists rebuild in order, its citations as JSON, and an optional ICD
coding for assessment claims.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_soap_report"
down_revision: str | Sequence[str] | None = "0002_dialogue"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "soap_report",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("dialogue_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["dialogue_id"], ["dialogue.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dialogue_id", name="uq_soap_report_dialogue_id"),
    )
    op.create_table(
        "soap_note",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("report_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["report_id"], ["soap_report.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_soap_note_report_id"), "soap_note", ["report_id"], unique=False)
    op.create_table(
        "soap_claim",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("note_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(length=32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("icd_code", sa.String(length=32), nullable=True),
        sa.Column("icd_name", sa.Text(), nullable=True),
        sa.Column("icd_classifier_url", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["note_id"], ["soap_note.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_soap_claim_note_id"), "soap_claim", ["note_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_soap_claim_note_id"), table_name="soap_claim")
    op.drop_table("soap_claim")
    op.drop_index(op.f("ix_soap_note_report_id"), table_name="soap_note")
    op.drop_table("soap_note")
    op.drop_table("soap_report")
