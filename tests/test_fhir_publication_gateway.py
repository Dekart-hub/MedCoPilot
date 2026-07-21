from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest

from dialogue.dialogue import Dialogue
from ehr.fhir import FhirGatewayError
from ehr.publication import EhrPublication
from infra.fhir import FhirR4PublicationGateway
from shared.value_objects import Id
from soap.correction import SoapReportCorrection
from soap.soap import SoapReport

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def _publication() -> EhrPublication:
    dialogue = Dialogue.start()
    dialogue.add_turn("patient", "Headache")
    correction = SoapReportCorrection.start(SoapReport(id=Id.new()), created_at=_NOW)
    correction.verify("doctor-1", at=_NOW)
    return EhrPublication.request(
        correction,
        dialogue,
        patient_ref="Patient/p1",
        encounter_ref="Encounter/e1",
        author_ref="Practitioner/d1",
        at=_NOW,
    )


def _source(reference: str) -> dict:
    resource_type, resource_id = reference.split("/")
    resource: dict = {"resourceType": resource_type, "id": resource_id}
    if resource_type == "Encounter":
        resource.update(
            {
                "status": "finished",
                "class": {"code": "AMB"},
                "subject": {"reference": "Patient/p1"},
            }
        )
    return resource


def _search(resources: list[dict]) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "entry": [{"resource": resource} for resource in resources],
    }


def test_gateway_loads_context_and_conditionally_creates_document() -> None:
    publication = _publication()
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/Bundle"):
            captured["query"] = request.url.params.get("identifier")
            return httpx.Response(200, json=_search([]))
        if request.method == "GET":
            return httpx.Response(200, json=_source(request.url.path.split("/fhir/")[-1]))
        captured["header"] = request.headers.get("If-None-Exist")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            headers={"Location": "http://ehr.test/fhir/Bundle/b1/_history/3"},
        )

    async def exercise():
        async with httpx.AsyncClient(
            base_url="http://ehr.test/fhir/",
            transport=httpx.MockTransport(handler),
        ) as client:
            gateway = FhirR4PublicationGateway(
                base_url="http://ehr.test/fhir",
                identifier_system="urn:medcopilot:publication",
                timeout_seconds=1,
                client=client,
            )
            return await gateway.deliver(publication)

    result = asyncio.run(exercise())
    identifier = f"urn:medcopilot:publication|{publication.id}"

    assert result.remote_reference == "Bundle/b1"
    assert result.remote_version == "3"
    assert captured["query"] == identifier
    assert captured["header"] == f"identifier={identifier}"
    assert captured["body"]["type"] == "document"  # type: ignore[index]


def test_repeat_delivery_resolves_existing_bundle_without_second_create() -> None:
    publication = _publication()
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        return httpx.Response(
            200,
            json=_search(
                [
                    {
                        "resourceType": "Bundle",
                        "id": "existing",
                        "meta": {"versionId": "7"},
                    }
                ]
            ),
        )

    async def exercise():
        async with httpx.AsyncClient(
            base_url="http://ehr.test/fhir/",
            transport=httpx.MockTransport(handler),
        ) as client:
            gateway = FhirR4PublicationGateway(
                base_url="http://ehr.test/fhir",
                identifier_system="urn:test",
                timeout_seconds=1,
                client=client,
            )
            return await gateway.deliver(publication)

    result = asyncio.run(exercise())

    assert result.remote_reference == "Bundle/existing"
    assert result.remote_version == "7"
    assert requests == [("GET", "/fhir/Bundle")]


def test_lost_create_response_is_recovered_by_identifier_lookup() -> None:
    publication = _publication()
    created = False
    post_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal created, post_count
        if request.method == "GET" and request.url.path.endswith("/Bundle"):
            resources = [{"resourceType": "Bundle", "id": "recovered"}] if created else []
            return httpx.Response(200, json=_search(resources))
        if request.method == "GET":
            return httpx.Response(200, json=_source(request.url.path.split("/fhir/")[-1]))
        post_count += 1
        created = True
        raise httpx.ReadError("response was lost", request=request)

    async def exercise():
        async with httpx.AsyncClient(
            base_url="http://ehr.test/fhir/",
            transport=httpx.MockTransport(handler),
        ) as client:
            gateway = FhirR4PublicationGateway(
                base_url="http://ehr.test/fhir",
                identifier_system="urn:test",
                timeout_seconds=1,
                client=client,
            )
            with pytest.raises(FhirGatewayError):
                await gateway.deliver(publication)
            return await gateway.deliver(publication)

    result = asyncio.run(exercise())

    assert result.remote_reference == "Bundle/recovered"
    assert post_count == 1
