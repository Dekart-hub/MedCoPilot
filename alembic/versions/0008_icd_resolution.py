"""ICD resolution audit trail on claims (T29)

Adds the resolution status, ranked candidate pool (JSON) and classifier
version to both the extracted and the corrected claim tables. All columns are
nullable: pre-T29 rows and manually entered codings carry no resolution.

Revision ID: 0008_icd_resolution
Revises: 0007_ehr_publication
Create Date: 2026-07-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_icd_resolution"
down_revision: str | Sequence[str] | None = "0007_ehr_publication"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("soap_claim", "soap_corrected_claim")


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(table, sa.Column("icd_status", sa.String(length=16), nullable=True))
        op.add_column(
            table, sa.Column("icd_classifier_version", sa.String(length=64), nullable=True)
        )
        op.add_column(table, sa.Column("icd_candidates", sa.JSON(), nullable=True))


def downgrade() -> None:
    for table in _TABLES:
        op.drop_column(table, "icd_candidates")
        op.drop_column(table, "icd_classifier_version")
        op.drop_column(table, "icd_status")
