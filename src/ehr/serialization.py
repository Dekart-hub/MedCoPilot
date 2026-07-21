"""JSON serialization for the EHR publication workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .publication_use_cases import PublicationDelivery


def publication_to_dict(delivery: PublicationDelivery) -> dict[str, Any]:
    publication = delivery.publication
    outbox = delivery.outbox
    return {
        "id": str(publication.id),
        "source_report_id": str(publication.source_report_id),
        "correction_id": str(publication.correction_id),
        "status": publication.status.value,
        "patient_ref": publication.patient_ref,
        "encounter_ref": publication.encounter_ref,
        "author_ref": publication.author_ref,
        "snapshot_schema_version": publication.snapshot_schema_version,
        "snapshot_hash": publication.snapshot_hash,
        "attempts": outbox.attempt_count,
        "next_attempt_at": (
            None if outbox.delivered_at is not None else _iso(outbox.next_attempt_at)
        ),
        "last_error": outbox.last_error,
        "remote_reference": publication.remote_reference,
        "remote_version": publication.remote_version,
        "created_at": _iso(publication.created_at),
        "delivered_at": _iso(publication.delivered_at),
    }


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment is not None else None
