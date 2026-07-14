from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timezone

import httpx

from config import EhrSettings
from ehr import (
    FIXTURE_PHASE_SYSTEM,
    POST_VISIT_PHASE,
    PRE_VISIT_PHASE,
    WHO_ICD10_SYSTEM,
    ApprovalStatus,
    ReportRecord,
)
from infra.fhir import FhirR4EhrGateway
from shared.value_objects import Id
from soap import ContextStatus, ReportView
from soap.view import AssessmentView, ClaimView, ContextReferenceView, NoteView


def _tag(phase: str) -> dict:
    return {"tag": [{"system": FIXTURE_PHASE_SYSTEM, "code": phase}]}


def _bundle(resources: list[dict]) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "entry": [{"resource": resource} for resource in resources],
    }


def test_patient_context_excludes_post_visit_and_current_encounter_conditions():
    historical = {
        "resourceType": "Condition",
        "id": "history",
        "meta": _tag(PRE_VISIT_PHASE),
        "clinicalStatus": {"coding": [{"code": "resolved"}]},
        "code": {
            "coding": [
                {"system": "http://snomed.info/sct", "code": "x"},
                {
                    "system": WHO_ICD10_SYSTEM,
                    "code": "U07.1",
                    "display": "COVID-19",
                },
            ]
        },
    }
    leaked_gold = {
        "resourceType": "Condition",
        "id": "gold",
        "meta": _tag(POST_VISIT_PHASE),
        "code": {"coding": [{"system": WHO_ICD10_SYSTEM, "code": "G44.2"}]},
    }
    mislabeled_current = {
        "resourceType": "Condition",
        "id": "current",
        "meta": _tag(PRE_VISIT_PHASE),
        "encounter": {"reference": "Encounter/e1"},
        "code": {"coding": [{"system": WHO_ICD10_SYSTEM, "code": "G44.2"}]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/Patient/p1"):
            return httpx.Response(
                200,
                json={
                    "resourceType": "Patient",
                    "id": "p1",
                    "name": [{"use": "official", "given": ["Alexey"], "family": "Petrov"}],
                    "birthDate": "1990-04-12",
                    "gender": "male",
                },
            )
        if path.endswith("/Encounter/e1"):
            return httpx.Response(
                200,
                json={
                    "resourceType": "Encounter",
                    "id": "e1",
                    "subject": {"reference": "Patient/p1"},
                    "period": {"start": "2026-01-01T10:00:00Z"},
                },
            )
        if path.endswith("/Condition"):
            return httpx.Response(
                200, json=_bundle([historical, leaked_gold, mislabeled_current])
            )
        return httpx.Response(200, json=_bundle([]))

    async def exercise():
        async with httpx.AsyncClient(
            base_url="http://ehr.test/fhir/",
            transport=httpx.MockTransport(handler),
        ) as client:
            gateway = FhirR4EhrGateway(
                EhrSettings(enabled=True, base_url="http://ehr.test/fhir"),
                client=client,
            )
            return await gateway.get_patient_context(
                patient_ref="Patient/p1", encounter_ref="Encounter/e1"
            )

    context = asyncio.run(exercise())

    assert context.patient_name == "Alexey Petrov"
    assert [condition.reference for condition in context.conditions] == [
        "Condition/history"
    ]
    assert context.conditions[0].code is not None
    assert context.conditions[0].code.system == WHO_ICD10_SYSTEM
    assert context.conditions[0].code.code == "U07.1"


def test_sync_uses_conditional_create_and_approved_document_reference():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["header"] = request.headers.get("If-None-Exist")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "resourceType": "DocumentReference",
                "id": "doc-001",
                "meta": {"versionId": "1"},
            },
        )

    now = datetime.now(timezone.utc)
    claim = ClaimView(
        id=Id.new(),
        claim="headache",
        evidence_text="I have a headache",
        turn_id=Id.new(),
    )
    assessment = AssessmentView(
        id=Id.new(),
        claim="tension headache with prior migraine history",
        evidence_text="The pain feels like pressure",
        turn_id=Id.new(),
        context_references=[
            ContextReferenceView(
                reference="Condition/history",
                resource_type="Condition",
                category="condition",
                display="Migraine",
            )
        ],
    )
    report = ReportView(
        id=Id.new(),
        notes=[
            NoteView(
                id=Id.new(),
                subjective=claim,
                objective=claim,
                assessment=assessment,
                plan=claim,
            )
        ],
        created_at=now,
        updated_at=now,
        context_status=ContextStatus.AVAILABLE,
    )
    record = ReportRecord(
        report_id=str(report.id),
        dialogue_id=str(Id.new()),
        report=report,
        patient_ref="Patient/p1",
        encounter_ref="Encounter/e1",
        approval_status=ApprovalStatus.APPROVED,
        approved_by="Practitioner/gp1",
        approved_at=now,
    )

    async def exercise():
        async with httpx.AsyncClient(
            base_url="http://ehr.test/fhir/",
            transport=httpx.MockTransport(handler),
        ) as client:
            gateway = FhirR4EhrGateway(
                EhrSettings(enabled=True, base_url="http://ehr.test/fhir"),
                client=client,
            )
            return await gateway.sync_report(record)

    result = asyncio.run(exercise())
    body = captured["body"]
    attachment = body["content"][0]["attachment"]

    assert result.remote_reference == "DocumentReference/doc-001"
    assert captured["header"] == f"identifier=urn:medcopilot:soap-report|{report.id}"
    assert body["subject"]["reference"] == "Patient/p1"
    assert body["context"]["encounter"][0]["reference"] == "Encounter/e1"
    assert body["author"][0]["reference"] == "Practitioner/gp1"
    assert body["meta"]["tag"][0]["code"] == POST_VISIT_PHASE
    markdown = base64.b64decode(attachment["data"]).decode("utf-8")
    assert "MedCoPilot SOAP report" in markdown
    assert "FHIR context status: available" in markdown
    assert "> FHIR context: Condition/history" in markdown
