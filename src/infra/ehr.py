"""Read-only mock EHR backing the SOAP extractor's patient context (T9).

Adapts a bundled ``patient_id -> context`` JSON map to the
:class:`~ehr.client.EhrClient` port, so the extractor can be grounded with a
patient's history without a real EHR/FHIR integration. An unknown id resolves to
an empty string — the extractor renders that as "(none)" — so a missing record
degrades gracefully instead of raising. ``build_ehr_client`` is the composition
point that wires the settings-driven data source (bundled fixture by default).
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from config.settings import Settings
from ehr.client import EhrClient

_LOG = structlog.get_logger(__name__)

BUNDLED_EHR_PATH = Path(__file__).parent / "ehr_patients.json"


class MockEhrClient(EhrClient):
    """Serves patient context from an in-memory ``patient_id -> text`` map."""

    def __init__(self, contexts: dict[str, str]) -> None:
        self._contexts = contexts

    @classmethod
    def from_path(cls, path: Path) -> MockEhrClient:
        """Load the ``patient_id -> context`` map from a JSON object file."""
        records: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
        return cls(records)

    async def get_patient_context(self, patient_id: str) -> str:
        context = self._contexts.get(patient_id, "")
        if not context:
            _LOG.info("ehr.patient_context_missing", patient_id=patient_id)
        return context


def build_ehr_client(settings: Settings) -> MockEhrClient:
    """Construct a :class:`MockEhrClient`, backed by the bundled fixture by default."""
    return MockEhrClient.from_path(settings.ehr_mock_path or BUNDLED_EHR_PATH)
