from __future__ import annotations

import asyncio
import base64
import re
from datetime import datetime
from typing import Any

import httpx

from config.settings import EhrSettings
from ehr import (
    FIXTURE_PHASE_SYSTEM,
    POST_VISIT_PHASE,
    PRE_VISIT_PHASE,
    WHO_ICD10_SYSTEM,
    EhrCoding,
    EhrGateway,
    EhrGatewayError,
    EhrResourceSummary,
    EhrSyncResult,
    InvalidEhrReferenceError,
    PatientContext,
    ReportRecord,
)

_FHIR_ID = re.compile(r"^[A-Za-z0-9.-]{1,64}$")


class FhirR4EhrGateway(EhrGateway):
    """Small FHIR R4 client intentionally scoped to the external mock EHR."""

    def __init__(
        self,
        settings: EhrSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=f"{str(settings.base_url).rstrip('/')}/",
            timeout=settings.timeout_seconds,
            headers={"Accept": "application/fhir+json"},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_patient_context(
        self, *, patient_ref: str, encounter_ref: str
    ) -> PatientContext:
        patient_id = _reference_id(patient_ref, "Patient")
        _reference_id(encounter_ref, "Encounter")

        patient, encounter = await asyncio.gather(
            self._request_json("GET", patient_ref),
            self._request_json("GET", encounter_ref),
        )
        actual_subject = encounter.get("subject", {}).get("reference")
        if actual_subject != patient_ref:
            raise InvalidEhrReferenceError(
                f"{encounter_ref} belongs to {actual_subject or 'no patient'}, "
                f"not {patient_ref}"
            )

        queries = (
            ("Condition", {"patient": patient_id}),
            ("AllergyIntolerance", {"patient": patient_id}),
            ("MedicationRequest", {"subject": patient_ref}),
            ("Observation", {"subject": patient_ref}),
        )
        bundles = await asyncio.gather(
            *(
                self._request_json("GET", resource_type, params=params)
                for resource_type, params in queries
            )
        )
        resources = {
            resource_type: [
                resource
                for resource in _bundle_resources(bundle, resource_type)
                if _is_pre_visit(resource)
                and resource.get("encounter", {}).get("reference") != encounter_ref
            ]
            for (resource_type, _), bundle in zip(queries, bundles, strict=True)
        }

        return PatientContext(
            patient_ref=patient_ref,
            encounter_ref=encounter_ref,
            encounter_start=encounter.get("period", {}).get("start"),
            patient_name=_patient_name(patient),
            birth_date=patient.get("birthDate"),
            gender=patient.get("gender"),
            conditions=[_summarize(r) for r in resources["Condition"]],
            allergies=[_summarize(r) for r in resources["AllergyIntolerance"]],
            medications=[_summarize(r) for r in resources["MedicationRequest"]],
            observations=[_summarize(r) for r in resources["Observation"]],
        )

    async def sync_report(self, record: ReportRecord) -> EhrSyncResult:
        if not record.patient_ref or not record.encounter_ref:
            raise InvalidEhrReferenceError("Patient and encounter references are required")
        if not record.approved_by or record.approved_at is None:
            raise EhrGatewayError("An approved report is required for EHR sync")
        _reference_id(record.patient_ref, "Patient")
        _reference_id(record.encounter_ref, "Encounter")
        _reference_id(record.approved_by, "Practitioner")

        document = _document_reference(record, self._settings.identifier_system)
        response = await self._request(
            "POST",
            "DocumentReference",
            json=document,
            headers={
                "Content-Type": "application/fhir+json",
                "If-None-Exist": (
                    f"identifier={self._settings.identifier_system}|{record.report_id}"
                ),
            },
        )
        payload = _json_or_empty(response)
        resource_id = payload.get("id")
        if not resource_id:
            resource_id = _id_from_location(response.headers.get("Location"))
        if not resource_id:
            raise EhrGatewayError(
                "Mock EHR accepted DocumentReference but returned no resource id"
            )
        version_id = payload.get("meta", {}).get("versionId")
        return EhrSyncResult(
            remote_reference=f"DocumentReference/{resource_id}",
            version_id=version_id,
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return _json_or_empty(await self._request(method, path, **kwargs))

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise EhrGatewayError(f"Mock EHR is unavailable: {exc}") from exc
        if response.status_code >= 400:
            detail = _operation_outcome_detail(response)
            raise EhrGatewayError(
                f"Mock EHR returned HTTP {response.status_code}: {detail}"
            )
        return response


def _reference_id(reference: str, expected_type: str) -> str:
    prefix = f"{expected_type}/"
    if not reference.startswith(prefix):
        raise InvalidEhrReferenceError(
            f"Expected {expected_type}/{{id}}, got {reference!r}"
        )
    resource_id = reference.removeprefix(prefix)
    if not _FHIR_ID.fullmatch(resource_id):
        raise InvalidEhrReferenceError(f"Invalid FHIR reference: {reference!r}")
    return resource_id


def _bundle_resources(
    bundle: dict[str, Any], expected_type: str
) -> list[dict[str, Any]]:
    if bundle.get("resourceType") != "Bundle":
        raise EhrGatewayError(f"FHIR search for {expected_type} returned no Bundle")
    return [
        resource
        for entry in bundle.get("entry", [])
        if (resource := entry.get("resource"))
        and resource.get("resourceType") == expected_type
    ]


def _is_pre_visit(resource: dict[str, Any]) -> bool:
    return any(
        tag.get("system") == FIXTURE_PHASE_SYSTEM
        and tag.get("code") == PRE_VISIT_PHASE
        for tag in resource.get("meta", {}).get("tag", [])
    )


def _patient_name(patient: dict[str, Any]) -> str | None:
    names = patient.get("name", [])
    if not names:
        return None
    preferred = next((name for name in names if name.get("use") == "official"), names[0])
    if preferred.get("text"):
        return preferred["text"]
    parts = [*preferred.get("given", []), preferred.get("family")]
    rendered = " ".join(part for part in parts if part)
    return rendered or None


def _summarize(resource: dict[str, Any]) -> EhrResourceSummary:
    resource_type = resource["resourceType"]
    concept = _resource_concept(resource)
    return EhrResourceSummary(
        reference=f"{resource_type}/{resource['id']}",
        resource_type=resource_type,
        code=_coding(concept),
        status=_resource_status(resource),
        effective_at=_effective_at(resource),
        value=_resource_value(resource),
    )


def _resource_concept(resource: dict[str, Any]) -> dict[str, Any]:
    resource_type = resource.get("resourceType")
    if resource_type == "MedicationRequest":
        return resource.get("medicationCodeableConcept", {})
    return resource.get("code", {})


def _coding(concept: dict[str, Any]) -> EhrCoding | None:
    codings = concept.get("coding", [])
    selected = next(
        (coding for coding in codings if coding.get("system") == WHO_ICD10_SYSTEM),
        codings[0] if codings else None,
    )
    if selected is None and not concept.get("text"):
        return None
    selected = selected or {}
    return EhrCoding(
        system=selected.get("system"),
        code=selected.get("code"),
        display=selected.get("display") or concept.get("text"),
    )


def _resource_status(resource: dict[str, Any]) -> str | None:
    if resource.get("status"):
        return resource["status"]
    clinical_status = resource.get("clinicalStatus", {}).get("coding", [])
    return clinical_status[0].get("code") if clinical_status else None


def _effective_at(resource: dict[str, Any]) -> str | None:
    return next(
        (
            resource.get(field)
            for field in (
                "effectiveDateTime",
                "recordedDate",
                "authoredOn",
                "onsetDateTime",
            )
            if resource.get(field)
        ),
        None,
    )


def _resource_value(resource: dict[str, Any]) -> str | None:
    if value := resource.get("valueString"):
        return value
    if quantity := resource.get("valueQuantity"):
        number = quantity.get("value")
        unit = quantity.get("unit") or quantity.get("code")
        return " ".join(str(part) for part in (number, unit) if part is not None)
    if concept := resource.get("valueCodeableConcept"):
        coding = _coding(concept)
        return coding.display if coding else None
    return None


def _document_reference(
    record: ReportRecord, identifier_system: str
) -> dict[str, Any]:
    assert record.patient_ref is not None
    assert record.encounter_ref is not None
    assert record.approved_by is not None
    assert record.approved_at is not None
    markdown = _render_report_markdown(record)
    return {
        "resourceType": "DocumentReference",
        "meta": {
            "tag": [
                {"system": FIXTURE_PHASE_SYSTEM, "code": POST_VISIT_PHASE}
            ]
        },
        "identifier": [
            {"system": identifier_system, "value": record.report_id}
        ],
        "status": "current",
        "docStatus": "final",
        "type": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "34109-9",
                    "display": "Note",
                }
            ],
            "text": "Clinician-approved SOAP note",
        },
        "subject": {"reference": record.patient_ref},
        "date": record.approved_at.isoformat(),
        "author": [{"reference": record.approved_by}],
        "description": f"MedCoPilot SOAP report {record.report_id}",
        "content": [
            {
                "attachment": {
                    "contentType": "text/markdown",
                    "data": base64.b64encode(markdown.encode("utf-8")).decode("ascii"),
                    "title": "Clinician-approved SOAP note",
                    "creation": record.approved_at.isoformat(),
                }
            }
        ],
        "context": {"encounter": [{"reference": record.encounter_ref}]},
    }


def _render_report_markdown(record: ReportRecord) -> str:
    lines = [
        "# MedCoPilot SOAP report",
        "",
        f"Report: {record.report_id}",
        f"Dialogue: {record.dialogue_id}",
        f"Patient: {record.patient_ref}",
        f"Encounter: {record.encounter_ref}",
        f"FHIR context status: {record.report.context_status}",
    ]
    for index, note in enumerate(record.report.notes, 1):
        lines.extend(["", f"## Problem {index}"])
        sections = (
            ("S — Subjective", note.subjective),
            ("O — Objective", note.objective),
            ("A — Assessment", note.assessment),
            ("P — Plan", note.plan),
        )
        for heading, claim in sections:
            lines.extend(
                [
                    "",
                    f"### {heading}",
                    claim.claim,
                    "",
                    f"> Evidence: {claim.evidence_text}",
                ]
            )
            if claim.context_references:
                references = ", ".join(
                    reference.reference for reference in claim.context_references
                )
                lines.append(f"> FHIR context: {references}")
        if note.assessment.codings:
            lines.extend(["", "ICD candidates:"])
            lines.extend(
                f"- {coding.code} — {coding.title}"
                for coding in note.assessment.codings
            )
    return "\n".join(lines) + "\n"


def _json_or_empty(response: httpx.Response) -> dict[str, Any]:
    if not response.content:
        return {}
    try:
        payload = response.json()
    except ValueError as exc:
        raise EhrGatewayError("Mock EHR returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise EhrGatewayError("Mock EHR returned an invalid FHIR payload")
    return payload


def _operation_outcome_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300] or "no response body"
    issues = payload.get("issue", []) if isinstance(payload, dict) else []
    details = [
        issue.get("diagnostics") or issue.get("details", {}).get("text")
        for issue in issues
    ]
    rendered = "; ".join(detail for detail in details if detail)
    return rendered or "FHIR operation failed"


def _id_from_location(location: str | None) -> str | None:
    if not location:
        return None
    parts = location.split("?", 1)[0].rstrip("/").split("/")
    try:
        index = len(parts) - 1 - parts[::-1].index("DocumentReference")
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    resource_id = parts[index + 1]
    return resource_id if _FHIR_ID.fullmatch(resource_id) else None
