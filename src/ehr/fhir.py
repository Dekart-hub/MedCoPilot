"""FHIR R4 document mapping and the outbound publication gateway port."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from typing import Any
from uuid import uuid5

from .publication import (
    EhrPublication,
    SnapshotClaim,
    SnapshotNote,
    canonical_json,
    validate_reference,
)

FHIR_JSON = dict[str, Any]
ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10"
XHTML_NS = "http://www.w3.org/1999/xhtml"

_SECTION_CODES = (
    ("subjective", "Subjective", "10164-2"),
    ("objective", "Objective", "10210-3"),
    ("assessment", "Assessment", "51848-0"),
    ("plan", "Plan", "18776-5"),
)


class FhirGatewayError(Exception):
    """Raised when the remote FHIR service cannot accept a publication."""


class FhirResourceMismatch(FhirGatewayError):
    """Raised when a referenced remote resource has the wrong identity."""


@dataclass(frozen=True, slots=True)
class FhirSourceResources:
    patient: FHIR_JSON
    encounter: FHIR_JSON
    practitioner: FHIR_JSON


@dataclass(frozen=True, slots=True)
class FhirDeliveryResult:
    remote_reference: str
    remote_version: str | None = None


class FhirPublicationGateway(ABC):
    @abstractmethod
    async def deliver(self, publication: EhrPublication) -> FhirDeliveryResult:
        """Create or resolve the document identified by ``publication.id``."""

    async def aclose(self) -> None:
        """Release transport resources when the adapter owns them."""
        return None


def build_fhir_document(
    publication: EhrPublication,
    resources: FhirSourceResources,
    *,
    identifier_system: str,
) -> FHIR_JSON:
    publication.verify_snapshot()
    if not identifier_system.strip():
        raise ValueError("FHIR identifier system must not be empty")

    refs = {
        publication.patient_ref: _urn(publication, publication.patient_ref),
        publication.encounter_ref: _urn(publication, publication.encounter_ref),
        publication.author_ref: _urn(publication, publication.author_ref),
    }
    patient = _source_resource(resources.patient, publication.patient_ref, "Patient", refs)
    encounter = _source_resource(resources.encounter, publication.encounter_ref, "Encounter", refs)
    practitioner = _source_resource(
        resources.practitioner,
        publication.author_ref,
        "Practitioner",
        refs,
    )
    if encounter.get("subject", {}).get("reference") != refs[publication.patient_ref]:
        raise FhirResourceMismatch(
            f"{publication.encounter_ref} does not belong to {publication.patient_ref}"
        )

    condition_entries, condition_refs = _conditions(publication, refs)
    dialogue_ref = _urn(publication, "DocumentReference/dialogue")
    dialogue = _dialogue_document(publication, refs)
    composition_ref = _urn(publication, "Composition/document")
    composition = _composition(
        publication,
        refs,
        dialogue_ref=dialogue_ref,
        condition_refs=condition_refs,
        identifier_system=identifier_system,
    )

    return {
        "resourceType": "Bundle",
        "id": str(publication.id),
        "identifier": {
            "system": identifier_system,
            "value": str(publication.id),
        },
        "type": "document",
        "timestamp": _fhir_datetime(publication.created_at),
        "entry": [
            {"fullUrl": composition_ref, "resource": composition},
            {"fullUrl": refs[publication.patient_ref], "resource": patient},
            {"fullUrl": refs[publication.encounter_ref], "resource": encounter},
            {"fullUrl": refs[publication.author_ref], "resource": practitioner},
            *condition_entries,
            {"fullUrl": dialogue_ref, "resource": dialogue},
        ],
    }


def _source_resource(
    resource: FHIR_JSON,
    reference: str,
    expected_type: str,
    refs: dict[str, str],
) -> FHIR_JSON:
    expected_id = validate_reference(reference, expected_type)
    if resource.get("resourceType") != expected_type or resource.get("id") != expected_id:
        raise FhirResourceMismatch(f"FHIR resource does not match {reference}")
    copied = deepcopy(resource)
    remapped = _remap_references(copied, refs)
    if not isinstance(remapped, dict):
        raise FhirResourceMismatch(f"FHIR resource does not match {reference}")
    return remapped


def _conditions(
    publication: EhrPublication, refs: dict[str, str]
) -> tuple[list[FHIR_JSON], dict[str, str]]:
    entries: list[FHIR_JSON] = []
    condition_refs: dict[str, str] = {}
    for note in publication.snapshot.notes:
        for claim in note.assessment:
            if claim.icd is None:
                continue
            key = f"Condition/{note.id}/{claim.id}"
            full_url = _urn(publication, key)
            condition_refs[claim.id] = full_url
            resource_id = _resource_id(publication, key)
            entries.append(
                {
                    "fullUrl": full_url,
                    "resource": {
                        "resourceType": "Condition",
                        "id": resource_id,
                        "clinicalStatus": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                    "code": "active",
                                }
                            ]
                        },
                        "verificationStatus": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                                    "code": "confirmed",
                                }
                            ]
                        },
                        "category": [
                            {
                                "coding": [
                                    {
                                        "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                        "code": "encounter-diagnosis",
                                    }
                                ]
                            }
                        ],
                        "code": {
                            "coding": [
                                {
                                    "system": ICD10_SYSTEM,
                                    "code": claim.icd.code,
                                    "display": claim.icd.name,
                                }
                            ],
                            "text": claim.text,
                        },
                        "subject": {"reference": refs[publication.patient_ref]},
                        "encounter": {"reference": refs[publication.encounter_ref]},
                        "recorder": {"reference": refs[publication.author_ref]},
                        "recordedDate": _fhir_datetime(publication.snapshot.verified_at),
                    },
                }
            )
    return entries, condition_refs


def _composition(
    publication: EhrPublication,
    refs: dict[str, str],
    *,
    dialogue_ref: str,
    condition_refs: dict[str, str],
    identifier_system: str,
) -> FHIR_JSON:
    sections = [
        _note_section(note, index, condition_refs)
        for index, note in enumerate(publication.snapshot.notes, start=1)
    ]
    sections.append(
        {
            "title": "Source dialogue",
            "code": {"text": "Doctor-patient dialogue"},
            "entry": [{"reference": dialogue_ref}],
        }
    )
    return {
        "resourceType": "Composition",
        "id": _resource_id(publication, "Composition/document"),
        "identifier": {
            "system": identifier_system,
            "value": str(publication.id),
        },
        "status": "final",
        "type": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "11506-3",
                    "display": "Progress note",
                }
            ]
        },
        "subject": {"reference": refs[publication.patient_ref]},
        "encounter": {"reference": refs[publication.encounter_ref]},
        "date": _fhir_datetime(publication.snapshot.verified_at),
        "author": [{"reference": refs[publication.author_ref]}],
        "title": "MedCoPilot verified SOAP report",
        "attester": [
            {
                "mode": "legal",
                "time": _fhir_datetime(publication.snapshot.verified_at),
                "party": {"reference": refs[publication.author_ref]},
            }
        ],
        "section": sections,
    }


def _note_section(note: SnapshotNote, index: int, condition_refs: dict[str, str]) -> FHIR_JSON:
    subsections: list[FHIR_JSON] = []
    for field_name, title, loinc in _SECTION_CODES:
        claims: tuple[SnapshotClaim, ...] = getattr(note, field_name)
        section: FHIR_JSON = {
            "title": title,
            "code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": title}]},
            "text": {"status": "generated", "div": _claims_xhtml(claims)},
        }
        if field_name == "assessment":
            entries = [
                {"reference": condition_refs[claim.id]}
                for claim in claims
                if claim.id in condition_refs
            ]
            if entries:
                section["entry"] = entries
        subsections.append(section)
    return {
        "title": f"SOAP Note {index}",
        "code": {"text": "SOAP note"},
        "section": subsections,
    }


def _claims_xhtml(claims: tuple[SnapshotClaim, ...]) -> str:
    if not claims:
        return f'<div xmlns="{XHTML_NS}"><p>No information documented.</p></div>'
    items: list[str] = []
    for claim in claims:
        citations = "; ".join(
            (
                f"turn {escape(citation.turn_id)}: {escape(citation.quote)}"
                if citation.quote is not None
                else f"turn {escape(citation.turn_id)}"
            )
            for citation in claim.citations
        )
        coding = (
            f" (ICD-10 {escape(claim.icd.code)}: {escape(claim.icd.name)})"
            if claim.icd is not None
            else ""
        )
        items.append(
            f"<li>{escape(claim.text)}{coding}<br/><small>Evidence: {citations}</small></li>"
        )
    return f'<div xmlns="{XHTML_NS}"><ul>{"".join(items)}</ul></div>'


def _dialogue_document(publication: EhrPublication, refs: dict[str, str]) -> FHIR_JSON:
    dialogue = canonical_json(
        {
            "schema_version": publication.snapshot.schema_version,
            "dialogue_id": publication.snapshot.dialogue_id,
            "turns": [
                {"id": turn.id, "speaker": turn.speaker, "text": turn.text}
                for turn in publication.snapshot.dialogue_turns
            ],
        }
    )
    return {
        "resourceType": "DocumentReference",
        "id": _resource_id(publication, "DocumentReference/dialogue"),
        "status": "current",
        "docStatus": "final",
        "type": {"text": "Source doctor-patient dialogue"},
        "subject": {"reference": refs[publication.patient_ref]},
        "date": _fhir_datetime(publication.created_at),
        "author": [{"reference": refs[publication.author_ref]}],
        "content": [
            {
                "attachment": {
                    "contentType": "application/json",
                    "language": "en",
                    "data": base64.b64encode(dialogue.encode()).decode(),
                    "title": "MedCoPilot source dialogue",
                    "creation": _fhir_datetime(publication.created_at),
                }
            }
        ],
        "context": {"encounter": [{"reference": refs[publication.encounter_ref]}]},
    }


def _remap_references(value: Any, refs: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [_remap_references(item, refs) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                refs[item]
                if key == "reference" and isinstance(item, str) and item in refs
                else _remap_references(item, refs)
            )
            for key, item in value.items()
        }
    return value


def _urn(publication: EhrPublication, key: str) -> str:
    return f"urn:uuid:{_resource_id(publication, key)}"


def _resource_id(publication: EhrPublication, key: str) -> str:
    return str(uuid5(publication.id.value, key))


def _fhir_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
