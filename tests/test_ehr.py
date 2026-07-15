"""Unit tests for the mock EHR client (only the bundled fixture touches disk)."""

from __future__ import annotations

import asyncio

from config.settings import Settings
from infra.ehr import MockEhrClient, build_ehr_client


def _context(client: MockEhrClient, patient_id: str) -> str:
    return asyncio.run(client.get_patient_context(patient_id))


def test_known_patient_resolves_to_its_context() -> None:
    client = MockEhrClient({"P001": "History: asthma.\nMedications: salbutamol."})
    assert _context(client, "P001") == "History: asthma.\nMedications: salbutamol."


def test_unknown_patient_resolves_to_empty_context() -> None:
    client = MockEhrClient({"P001": "History: asthma."})
    assert _context(client, "P999") == ""


def test_build_ehr_client_serves_a_usable_context_from_the_bundled_fixture() -> None:
    client = build_ehr_client(Settings())
    context = _context(client, "P001")
    assert context.strip()
