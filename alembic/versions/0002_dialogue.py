"""dialogue aggregate

Revision ID: 0002_dialogue
Revises: 0001_initial
Create Date: 2026-07-15

Creates the ``dialogue`` aggregate tables: ``dialogue`` (the root) and its
ordered ``dialogue_turn`` children, cascade-deleted with their parent.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_dialogue"
down_revision: str | Sequence[str] | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dialogue",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "dialogue_turn",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("dialogue_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("speaker", sa.String(length=255), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["dialogue_id"], ["dialogue.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_dialogue_turn_dialogue_id"),
        "dialogue_turn",
        ["dialogue_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_dialogue_turn_dialogue_id"), table_name="dialogue_turn")
    op.drop_table("dialogue_turn")
    op.drop_table("dialogue")
