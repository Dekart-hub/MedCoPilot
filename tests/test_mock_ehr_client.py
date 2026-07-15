from __future__ import annotations

import asyncio

from config.settings import DEFAULT_EHR_MOCK_PATH, Settings
from infra import MockEhrClient, build_ehr_client

KNOWN_PATIENT_ID = "mock-patient-001"


def test_known_patient_id_resolves_to_its_context():
    client = MockEhrClient({KNOWN_PATIENT_ID: "penicillin allergy"})
    assert asyncio.run(client.get_patient_context(KNOWN_PATIENT_ID)) == "penicillin allergy"


def test_unknown_patient_id_falls_back_to_default():
    client = MockEhrClient({KNOWN_PATIENT_ID: "penicillin allergy"})
    assert asyncio.run(client.get_patient_context("nobody")) == ""


def test_build_ehr_client_serves_bundled_patient_context(monkeypatch):
    monkeypatch.setenv("OPENAI__API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.ehr_mock_path == DEFAULT_EHR_MOCK_PATH

    client = build_ehr_client(settings)
    context = asyncio.run(client.get_patient_context(KNOWN_PATIENT_ID))
    assert "penicillin" in context.lower()
    assert asyncio.run(client.get_patient_context("nobody")) == ""
