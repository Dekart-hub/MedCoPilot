"""soap report created_at

Revision ID: 0005_soap_report_created_at
Revises: 0004_soap_correction
Create Date: 2026-07-20

Adds ``soap_report.created_at`` — the moment a dialogue's report was first
extracted, stamped once by the application and left untouched on re-extraction
[#88/FR-1]. The column is added with a ``now()`` server default so it backfills
cleanly on a populated table (existing rows get a timestamp); the default is then
dropped so the column matches the ORM — application-supplied, no DB default —
exactly as the correction tables' timestamps are handled.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_soap_report_created_at"
down_revision: str | Sequence[str] | None = "0004_soap_correction"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "soap_report",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.alter_column("soap_report", "created_at", server_default=None)


def downgrade() -> None:
    op.drop_column("soap_report", "created_at")
