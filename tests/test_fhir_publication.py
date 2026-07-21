from __future__ import annotations

import base64
import json
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from dialogue.dialogue import Dialogue
from ehr.fhir import (
    ICD10_SYSTEM,
    FhirResourceMismatch,
    FhirSourceResources,
    build_fhir_document,
)
from ehr.publication import EhrPublication
from shared.value_objects import Id
from soap.correction import SoapReportCorrection
from soap.soap import (
    AssessmentClaim,
    IcdCoding,
    SoapClaim,
    SoapNote,
    SoapReport,
    TurnCitation,
)

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _publication() -> EhrPublication:
    dialogue = Dialogue.start()
    patient = dialogue.add_turn("patient", "My head has hurt for three days.")
    doctor = dialogue.add_turn("doctor", "This is a tension headache.")
    cite_patient = TurnCitation(patient.id, "head has hurt")
    cite_doctor = TurnCitation(doctor.id)
    report = SoapReport(
        id=Id.new(),
        notes=[
            SoapNote(
                id=Id.new(),
                subjective=[
                    SoapClaim(
                        id=Id.new(),
                        text="Headache for three days.",
                        citations=[cite_patient],
                    )
                ],
                assessment=[
                    AssessmentClaim(
                        id=Id.new(),
                        text="Tension headache.",
                        citations=[cite_doctor],
                        icd=IcdCoding(
                            code="G44.2",
                            name="Tension-type headache",
                            classifier_url="https://icd.who.int/browse10/2019/en#/G44.2",
                        ),
                    ),
                    AssessmentClaim(
                        id=Id.new(),
                        text="Monitor for migraine features.",
                        citations=[cite_doctor],
                    ),
                ],
                plan=[
                    SoapClaim(
                        id=Id.new(),
                        text="Use paracetamol as needed.",
                        citations=[cite_doctor],
                    )
                ],
            )
        ],
    )
    correction = SoapReportCorrection.start(report, created_at=_NOW)
    correction.verify("doctor-1", at=_NOW)
    return EhrPublication.request(
        correction,
        dialogue,
        patient_ref="Patient/p1",
        encounter_ref="Encounter/e1",
        author_ref="Practitioner/d1",
        at=_NOW,
    )


def _resources() -> FhirSourceResources:
    return FhirSourceResources(
        patient={
            "resourceType": "Patient",
            "id": "p1",
            "meta": {"versionId": "4"},
            "name": [{"family": "Petrov", "given": ["Alexey"]}],
        },
        encounter={
            "resourceType": "Encounter",
            "id": "e1",
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "AMB",
            },
            "subject": {"reference": "Patient/p1"},
            "participant": [{"individual": {"reference": "Practitioner/d1"}}],
        },
        practitioner={
            "resourceType": "Practitioner",
            "id": "d1",
            "active": True,
            "name": [{"family": "Sokolova", "given": ["Elena"]}],
        },
    )


def _resource(bundle: dict, resource_type: str) -> list[dict]:
    return [
        entry["resource"]
        for entry in bundle["entry"]
        if entry["resource"]["resourceType"] == resource_type
    ]


def test_document_bundle_is_deterministic_complete_and_internally_resolved() -> None:
    publication = _publication()
    resources = _resources()
    original_resources = deepcopy(resources)

    first = build_fhir_document(
        publication, resources, identifier_system="urn:medcopilot:publication"
    )
    second = build_fhir_document(
        publication, resources, identifier_system="urn:medcopilot:publication"
    )

    assert first == second
    assert resources == original_resources
    assert first["type"] == "document"
    assert first["identifier"] == {
        "system": "urn:medcopilot:publication",
        "value": str(publication.id),
    }
    assert first["entry"][0]["resource"]["resourceType"] == "Composition"
    full_urls = {entry["fullUrl"] for entry in first["entry"]}
    assert len(full_urls) == len(first["entry"])
    assert all(url.startswith("urn:uuid:") for url in full_urls)

    composition = first["entry"][0]["resource"]
    assert composition["status"] == "final"
    assert composition["identifier"] == first["identifier"]
    assert composition["subject"]["reference"] in full_urls
    assert composition["encounter"]["reference"] in full_urls
    assert composition["author"][0]["reference"] in full_urls
    assert composition["attester"][0]["party"]["reference"] in full_urls

    note = composition["section"][0]
    assert [section["title"] for section in note["section"]] == [
        "Subjective",
        "Objective",
        "Assessment",
        "Plan",
    ]
    assessment = note["section"][2]
    assert len(assessment["entry"]) == 1
    assert "Monitor for migraine features." in assessment["text"]["div"]

    conditions = _resource(first, "Condition")
    assert len(conditions) == 1
    coding = conditions[0]["code"]["coding"][0]
    assert coding == {
        "system": ICD10_SYSTEM,
        "code": "G44.2",
        "display": "Tension-type headache",
    }
    assert conditions[0]["subject"]["reference"] in full_urls
    assert conditions[0]["encounter"]["reference"] in full_urls

    encounter = _resource(first, "Encounter")[0]
    assert encounter["subject"]["reference"] in full_urls
    assert encounter["participant"][0]["individual"]["reference"] in full_urls

    dialogue = _resource(first, "DocumentReference")[0]
    attachment = dialogue["content"][0]["attachment"]
    decoded = json.loads(base64.b64decode(attachment["data"]))
    assert [(turn["speaker"], turn["text"]) for turn in decoded["turns"]] == [
        ("patient", "My head has hurt for three days."),
        ("doctor", "This is a tension headache."),
    ]


def test_uncoded_assessment_stays_in_narrative_without_fake_condition() -> None:
    publication = _publication()
    document = build_fhir_document(publication, _resources(), identifier_system="urn:test")

    conditions = _resource(document, "Condition")
    assessment_div = document["entry"][0]["resource"]["section"][0]["section"][2]["text"]["div"]

    assert len(conditions) == 1
    assert "Monitor for migraine features." in assessment_div


def test_encounter_for_a_different_patient_is_rejected() -> None:
    publication = _publication()
    resources = _resources()
    resources.encounter["subject"] = {"reference": "Patient/other"}

    with pytest.raises(FhirResourceMismatch):
        build_fhir_document(publication, resources, identifier_system="urn:test")
