"""initial baseline

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-14

Establishes the ``alembic_version`` table so a fresh database has a migration
head to track. Domain tables arrive in T5.
"""

from __future__ import annotations

from collections.abc import Sequence

revision: str = "0001_initial"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
