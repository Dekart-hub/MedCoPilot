"""HTTP adapter for idempotent FHIR R4 document publication."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from config.settings import Settings
from ehr.fhir import (
    FHIR_JSON,
    FhirDeliveryResult,
    FhirGatewayError,
    FhirPublicationGateway,
    FhirResourceMismatch,
    FhirSourceResources,
    build_fhir_document,
)
from ehr.publication import EhrPublication, validate_reference


def build_fhir_publication_gateway(
    settings: Settings,
) -> FhirR4PublicationGateway:
    return FhirR4PublicationGateway(
        base_url=settings.fhir_base_url,
        identifier_system=settings.fhir_identifier_system,
        timeout_seconds=settings.fhir_timeout_seconds,
    )


class FhirR4PublicationGateway(FhirPublicationGateway):
    def __init__(
        self,
        *,
        base_url: str,
        identifier_system: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._identifier_system = identifier_system
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=f"{base_url.rstrip('/')}/",
            timeout=timeout_seconds,
            headers={"Accept": "application/fhir+json"},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def deliver(self, publication: EhrPublication) -> FhirDeliveryResult:
        existing = await self._find(publication)
        if existing is not None:
            return existing

        resources = await self._load_resources(publication)
        document = build_fhir_document(
            publication,
            resources,
            identifier_system=self._identifier_system,
        )
        try:
            response = await self._request(
                "POST",
                "Bundle",
                json=document,
                headers={
                    "Content-Type": "application/fhir+json",
                    "If-None-Exist": (f"identifier={self._identifier_query(publication)}"),
                },
            )
        except _FhirHttpError as exc:
            if exc.status_code not in {409, 412}:
                raise
            existing = await self._find(publication)
            if existing is None:
                raise
            return existing

        result = _delivery_result(response)
        if result is not None:
            return result
        existing = await self._find(publication)
        if existing is None:
            raise FhirGatewayError("FHIR accepted the document but returned no Bundle identity")
        return existing

    async def _load_resources(self, publication: EhrPublication) -> FhirSourceResources:
        patient, encounter, practitioner = await asyncio.gather(
            self._get_resource(publication.patient_ref, "Patient"),
            self._get_resource(publication.encounter_ref, "Encounter"),
            self._get_resource(publication.author_ref, "Practitioner"),
        )
        return FhirSourceResources(
            patient=patient,
            encounter=encounter,
            practitioner=practitioner,
        )

    async def _get_resource(self, reference: str, resource_type: str) -> FHIR_JSON:
        resource_id = validate_reference(reference, resource_type)
        resource = _json(await self._request("GET", reference))
        if resource.get("resourceType") != resource_type or resource.get("id") != resource_id:
            raise FhirResourceMismatch(f"FHIR resource does not match {reference}")
        return resource

    async def _find(self, publication: EhrPublication) -> FhirDeliveryResult | None:
        response = await self._request(
            "GET",
            "Bundle",
            params={"identifier": self._identifier_query(publication), "_count": "1"},
        )
        search = _json(response)
        if search.get("resourceType") != "Bundle":
            raise FhirGatewayError("FHIR Bundle search returned a non-Bundle response")
        for entry in search.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Bundle" and resource.get("id"):
                return FhirDeliveryResult(
                    remote_reference=f"Bundle/{resource['id']}",
                    remote_version=resource.get("meta", {}).get("versionId"),
                )
        return None

    def _identifier_query(self, publication: EhrPublication) -> str:
        return f"{self._identifier_system}|{publication.id}"

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise FhirGatewayError(f"FHIR service is unavailable: {exc}") from exc
        if response.status_code >= 400:
            raise _FhirHttpError(
                response.status_code,
                f"FHIR returned HTTP {response.status_code}: {_outcome_detail(response)}",
            )
        return response


class _FhirHttpError(FhirGatewayError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _delivery_result(response: httpx.Response) -> FhirDeliveryResult | None:
    resource = _json(response)
    resource_id = resource.get("id")
    version = resource.get("meta", {}).get("versionId")
    if resource.get("resourceType") == "Bundle" and resource_id:
        return FhirDeliveryResult(remote_reference=f"Bundle/{resource_id}", remote_version=version)
    location = response.headers.get("Location") or response.headers.get("Content-Location")
    if location is None:
        return None
    parts = location.rstrip("/").split("/")
    try:
        index = len(parts) - 1 - parts[::-1].index("Bundle")
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    resource_id = parts[index + 1]
    remote_version = (
        parts[index + 3] if index + 3 < len(parts) and parts[index + 2] == "_history" else None
    )
    return FhirDeliveryResult(
        remote_reference=f"Bundle/{resource_id}", remote_version=remote_version
    )


def _json(response: httpx.Response) -> FHIR_JSON:
    if not response.content:
        return {}
    try:
        data = response.json()
    except ValueError as exc:
        raise FhirGatewayError("FHIR returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise FhirGatewayError("FHIR returned a non-object JSON response")
    return data


def _outcome_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or "no response body"
    if isinstance(payload, dict):
        issues = payload.get("issue", [])
        if issues:
            issue = issues[0]
            return str(issue.get("diagnostics") or issue.get("details", {}).get("text"))
    return response.text or "no response body"
