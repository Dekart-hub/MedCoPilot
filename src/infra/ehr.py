from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from config import Settings
from ehr import EhrClient


class MockEhrClient(EhrClient):
    """In-memory ``EhrClient`` backed by a ``patient_id -> text`` mapping.

    An unknown ``patient_id`` yields ``default`` (empty by default), so callers
    always get a string they can drop into the extractor prompt.
    """

    def __init__(self, contexts: Mapping[str, str], *, default: str = "") -> None:
        self._contexts = dict(contexts)
        self._default = default

    async def get_patient_context(self, patient_id: str) -> str:
        return self._contexts.get(patient_id, self._default)

    @classmethod
    def from_file(cls, path: Path, *, default: str = "") -> MockEhrClient:
        contexts: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
        return cls(contexts, default=default)


def build_ehr_client(settings: Settings) -> EhrClient:
    """Собирает мок EHR-клиента из настроек."""
    return MockEhrClient.from_file(settings.ehr_mock_path)
