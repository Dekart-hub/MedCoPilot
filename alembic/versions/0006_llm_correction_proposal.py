"""llm correction proposal aggregate

Revision ID: 0006_llm_correction_proposal
Revises: 0005_soap_report_created_at
Create Date: 2026-07-21

Adds the LLM editor-session aggregate (#12) and the correction revision it hangs
off. ``soap_correction.revision`` is a monotonic content counter; it is added
with a ``1`` server default so existing rows backfill cleanly, then the default
is dropped so the column matches the ORM (application-supplied), exactly as
``0005`` handled ``created_at``. On top of the untouched correction tables come
``soap_correction_editor_session`` (unique per correction, cascade-deleted with
it), its ordered ``soap_correction_proposal`` children and their ordered
``soap_correction_proposal_operation`` grandchildren, each generation
cascade-deleted with its parent. Within a proposal a non-``NULL``
``target_note_id`` is unique via a partial unique index, mirroring the
one-operation-per-note invariant.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_llm_correction_proposal"
down_revision: str | Sequence[str] | None = "0005_soap_report_created_at"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "soap_correction",
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
    )
    op.alter_column("soap_correction", "revision", server_default=None)

    op.create_table(
        "soap_correction_editor_session",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("correction_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["correction_id"], ["soap_correction.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("correction_id", name="uq_soap_editor_session_correction_id"),
    )
    op.create_table(
        "soap_correction_proposal",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("user_request", sa.Text(), nullable=False),
        sa.Column("base_correction_revision", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=False),
        sa.Column("prompt_version", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["soap_correction_editor_session.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_soap_correction_proposal_session_id"),
        "soap_correction_proposal",
        ["session_id"],
        unique=False,
    )
    op.create_table(
        "soap_correction_proposal_operation",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("proposal_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("target_note_id", sa.Uuid(), nullable=True),
        sa.Column("proposed_content", sa.JSON(), nullable=True),
        sa.Column("before_snapshot", sa.JSON(), nullable=True),
        sa.Column("target_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decision_reason", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["proposal_id"], ["soap_correction_proposal.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_soap_correction_proposal_operation_proposal_id"),
        "soap_correction_proposal_operation",
        ["proposal_id"],
        unique=False,
    )
    op.create_index(
        "uq_soap_proposal_operation_target",
        "soap_correction_proposal_operation",
        ["proposal_id", "target_note_id"],
        unique=True,
        postgresql_where=sa.text("target_note_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_soap_proposal_operation_target",
        table_name="soap_correction_proposal_operation",
    )
    op.drop_index(
        op.f("ix_soap_correction_proposal_operation_proposal_id"),
        table_name="soap_correction_proposal_operation",
    )
    op.drop_table("soap_correction_proposal_operation")
    op.drop_index(
        op.f("ix_soap_correction_proposal_session_id"),
        table_name="soap_correction_proposal",
    )
    op.drop_table("soap_correction_proposal")
    op.drop_table("soap_correction_editor_session")
    op.drop_column("soap_correction", "revision")
