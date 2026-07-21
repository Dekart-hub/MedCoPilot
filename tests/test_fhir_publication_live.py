from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from dialogue.dialogue import Dialogue
from ehr.publication import EhrPublication
from infra.fhir import FhirR4PublicationGateway
from shared.value_objects import Id
from soap.correction import SoapReportCorrection
from soap.soap import (
    AssessmentClaim,
    IcdCoding,
    SoapNote,
    SoapReport,
    TurnCitation,
)

BASE_URL = os.environ.get("MOCK_EHR_BASE_URL")
FIXTURE = Path(__file__).resolve().parents[1] / "mock_ehr" / "fixtures" / "publication-context.json"

pytestmark = pytest.mark.skipif(
    not BASE_URL,
    reason="Set MOCK_EHR_BASE_URL to run the opt-in HAPI FHIR contract",
)


def _publication() -> EhrPublication:
    at = datetime(2026, 7, 21, 10, 30, tzinfo=UTC)
    dialogue = Dialogue.start()
    turn = dialogue.add_turn("patient", "Pressure-like headache for three days")
    report = SoapReport(
        id=Id.new(),
        notes=[
            SoapNote(
                id=Id.new(),
                assessment=[
                    AssessmentClaim(
                        id=Id.new(),
                        text="Tension-type headache",
                        citations=[TurnCitation(turn.id, "Pressure-like headache")],
                        icd=IcdCoding(
                            code="G44.2",
                            name="Tension-type headache",
                            classifier_url="https://icd.who.int/browse10/2019/en#/G44.2",
                        ),
                    )
                ],
            )
        ],
    )
    correction = SoapReportCorrection.start(report, created_at=at)
    correction.verify("doctor-1", at=at)
    return EhrPublication.request(
        correction,
        dialogue,
        patient_ref="Patient/mock-patient-001",
        encounter_ref="Encounter/mock-encounter-001",
        author_ref="Practitioner/mock-practitioner-001",
        at=at,
    )


def test_live_hapi_validates_document_and_keeps_delivery_idempotent() -> None:
    async def exercise() -> tuple[str, str, list[dict]]:
        async with httpx.AsyncClient(
            base_url=f"{(BASE_URL or '').rstrip('/')}/", timeout=30
        ) as client:
            seed = await client.post(
                "",
                json=json.loads(FIXTURE.read_text(encoding="utf-8")),
                headers={"Content-Type": "application/fhir+json"},
            )
            seed.raise_for_status()
            gateway = FhirR4PublicationGateway(
                base_url=BASE_URL or "",
                identifier_system="urn:medcopilot:ehr-publication",
                timeout_seconds=30,
                client=client,
            )
            publication = _publication()
            first = await gateway.deliver(publication)
            second = await gateway.deliver(publication)
            document = await client.get(first.remote_reference)
            document.raise_for_status()
            validation = await client.post(
                "Bundle/$validate",
                json=document.json(),
                headers={"Content-Type": "application/fhir+json"},
            )
            validation.raise_for_status()
            return (
                first.remote_reference,
                second.remote_reference,
                validation.json().get("issue", []),
            )

    first, second, issues = asyncio.run(exercise())

    assert first == second
    assert not [issue for issue in issues if issue.get("severity") in {"fatal", "error"}]
