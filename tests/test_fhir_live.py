from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import create_app
from config import EhrSettings
from config.settings import get_settings
from ehr import ApprovalStatus, ReportRecord
from infra.fhir import FhirR4EhrGateway
from shared.value_objects import Id
from soap import ReportView

BASE_URL = os.environ.get("MOCK_EHR_BASE_URL")

pytestmark = pytest.mark.skipif(
    not BASE_URL,
    reason="Set MOCK_EHR_BASE_URL to run the opt-in live HAPI FHIR smoke test",
)


def test_live_hapi_context_boundary_and_conditional_create():
    now = datetime.now(timezone.utc)
    view = ReportView(
        id=Id.from_str("55555555-5555-5555-5555-555555555555"),
        notes=[],
        created_at=now,
        updated_at=now,
    )
    record = ReportRecord(
        report_id=str(view.id),
        dialogue_id="11111111-1111-1111-1111-111111111111",
        report=view,
        patient_ref="Patient/mock-patient-001",
        encounter_ref="Encounter/mock-encounter-001",
        approval_status=ApprovalStatus.APPROVED,
        approved_by="Practitioner/mock-gp-001",
        approved_at=now,
    )

    async def exercise():
        gateway = FhirR4EhrGateway(
            EhrSettings(enabled=True, base_url=BASE_URL or "")
        )
        try:
            context = await gateway.get_patient_context(
                patient_ref="Patient/mock-patient-001",
                encounter_ref="Encounter/mock-encounter-001",
            )
            first = await gateway.sync_report(record)
            second = await gateway.sync_report(record)
            return context, first, second
        finally:
            await gateway.aclose()

    context, first, second = asyncio.run(exercise())

    condition_refs = {condition.reference for condition in context.conditions}
    assert "Condition/mock-condition-covid-001" in condition_refs
    assert "Condition/mock-condition-tension-headache-001" not in condition_refs
    assert first.remote_reference == second.remote_reference


def test_live_fastapi_context_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI__API_KEY", "sk-test")
    monkeypatch.setenv("EHR__ENABLED", "true")
    monkeypatch.setenv("EHR__BASE_URL", BASE_URL or "")
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            response = client.get(
                "/api/v1/ehr/dialogues/"
                "11111111-1111-1111-1111-111111111111/context"
            )
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    condition_refs = {
        condition["reference"] for condition in response.json()["conditions"]
    }
    assert "Condition/mock-condition-covid-001" in condition_refs
    assert "Condition/mock-condition-tension-headache-001" not in condition_refs
