"""
Pytest tests for the MedCoPilot SOAP Documentation Service.

Covers:
    - /health returns a valid HealthResponse
    - /generate-soap validates the request contract (Pydantic)
    - /generate-soap returns 422 for invalid input
    - /generate-soap returns a valid SoapResponse for valid input
    - Pydantic models serialize/deserialize correctly
    - Tier 0 short-circuit behavior (tier_1/tier_2 are None when tier_0 fails)
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import (
    EHRMetadata,
    EvaluationScores,
    ExecutionMetadata,
    GenerationSettings,
    HealthResponse,
    HealthStatus,
    Language,
    RubricDimension,
    SOAPItem,
    SoapNote,
    SoapRequest,
    SoapResponse,
    Tier0Validation,
    Tier1Evaluation,
    Tier2ClinicalRubric,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #

def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in {"healthy", "degraded", "unhealthy"}
    assert data["service"] == "medcopilot-soap"
    assert "checks" in data
    assert isinstance(data["checks"], dict)


def test_health_response_model_parses(client: TestClient) -> None:
    response = client.get("/health")
    parsed = HealthResponse.model_validate(response.json())
    assert parsed.status in HealthStatus
    assert parsed.version == "1.0.0"


# --------------------------------------------------------------------------- #
# /generate-soap — request validation
# --------------------------------------------------------------------------- #

def test_generate_soap_rejects_empty_transcript(client: TestClient) -> None:
    response = client.post("/generate-soap", json={"transcript": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_generate_soap_rejects_too_short_transcript(client: TestClient) -> None:
    response = client.post("/generate-soap", json={"transcript": "hi"})
    assert response.status_code == 422


def test_generate_soap_rejects_missing_transcript(client: TestClient) -> None:
    response = client.post("/generate-soap", json={})
    assert response.status_code == 422


def test_generate_soap_rejects_invalid_language(client: TestClient) -> None:
    response = client.post(
        "/generate-soap",
        json={
            "transcript": "Doctor: Hi.\nPatient: Hello.",
            "settings": {"language": "klingon"},
        },
    )
    assert response.status_code == 422


def test_generate_soap_rejects_out_of_range_threshold(client: TestClient) -> None:
    response = client.post(
        "/generate-soap",
        json={
            "transcript": "Doctor: Hi.\nPatient: Hello.",
            "settings": {"confidence_threshold": 1.5},
        },
    )
    assert response.status_code == 422


def test_generate_soap_minimal_contract_accepted(client: TestClient) -> None:
    """
    The minimal contract from the proposal is just { transcript: "..." }.
    Everything else is optional.
    """
    response = client.post(
        "/generate-soap",
        json={"transcript": "Doctor: What brings you in?\nPatient: Chest pain."},
    )
    assert response.status_code == 200


def test_generate_soap_full_contract_accepted(client: TestClient) -> None:
    response = client.post(
        "/generate-soap",
        json={
            "transcript": "Doctor: What brings you in?\nPatient: Chest pain.",
            "patient_id": "patient-12345",
            "encounter_id": "encounter-9876",
            "settings": {
                "language": "en",
                "include_pii_deid": True,
                "confidence_threshold": 0.6,
            },
        },
    )
    assert response.status_code == 200


def test_generate_soap_response_structure(client: TestClient) -> None:
    """Verify the response has all required top-level fields."""
    response = client.post(
        "/generate-soap",
        json={"transcript": "Doctor: Hi.\nPatient: My chest hurts."},
    )
    assert response.status_code == 200

    data = response.json()
    assert "note" in data
    assert "scores" in data
    assert "ehr" in data
    assert "flags" in data
    assert "metadata" in data

    # Verify note structure
    assert "subjective" in data["note"]
    assert "objective" in data["note"]
    assert "assessment" in data["note"]
    assert "plan" in data["note"]

    # Verify scores structure
    assert "tier_0_structural" in data["scores"]
    assert "composite_confidence_score" in data["scores"]
    assert "needs_review" in data["scores"]


# --------------------------------------------------------------------------- #
# Pydantic model round-trips
# --------------------------------------------------------------------------- #

def test_soap_item_round_trip() -> None:
    item = SOAPItem(
        text="Exertional chest discomfort for ~1 week",
        evidence_quote="tight feeling in my chest when I climb stairs for about a week",
        groundedness_score=0.94,
        is_flagged=False,
    )
    data = item.model_dump()
    restored = SOAPItem.model_validate(data)
    assert restored == item


def test_soap_item_without_groundedness_for_ap() -> None:
    """A/P items should not have groundedness_score (it's None by design)."""
    item = SOAPItem(
        text="Exertional chest discomfort, query angina",
        evidence_quote="tight feeling in my chest when I climb stairs",
    )
    assert item.groundedness_score is None
    assert item.is_flagged is False


def test_tier0_short_circuit_behavior() -> None:
    """
    When tier_0 fails, tier_1 and tier_2 should be None (short-circuit).
    This is a key design decision from the proposal.
    """
    scores = EvaluationScores(
        tier_0_structural=Tier0Validation(
            passed=False,
            resolved_citations_count=0,
            failure_reason="Citation not found in transcript",
        ),
        tier_1_groundedness=None,  # short-circuit
        tier_2_clinical=None,  # short-circuit
        composite_confidence_score=0.0,
        needs_review=True,
    )
    assert scores.tier_0_structural.passed is False
    assert scores.tier_1_groundedness is None
    assert scores.tier_2_clinical is None
    assert scores.needs_review is True


def test_tier2_clinical_rubric_rejects_out_of_range() -> None:
    with pytest.raises(Exception):
        Tier2ClinicalRubric(
            section="assessment",
            appropriateness=RubricDimension(rationale="ok", score=6),  # out of range
            completeness=RubricDimension(rationale="ok", score=4),
            placement=RubricDimension(rationale="ok", score=5),
            conciseness=RubricDimension(rationale="ok", score=5),
            safety=RubricDimension(rationale="ok", score=5),
        )


def test_tier2_clinical_rubric_rejects_invalid_section() -> None:
    with pytest.raises(Exception):
        Tier2ClinicalRubric(
            section="subjective",  # rubric is only for assessment/plan
            appropriateness=RubricDimension(rationale="ok", score=4),
            completeness=RubricDimension(rationale="ok", score=4),
            placement=RubricDimension(rationale="ok", score=5),
            conciseness=RubricDimension(rationale="ok", score=5),
            safety=RubricDimension(rationale="ok", score=5),
        )


def test_ehr_metadata_structure() -> None:
    ehr = EHRMetadata(
        synced=True,
        document_reference_id="DocumentReference/dr-abc123",
        resource_type="DocumentReference",
        fhir_endpoint="http://mock-ehr:8080/fhir",
        failure_reason=None,
    )
    assert ehr.synced is True
    assert ehr.resource_type == "DocumentReference"


def test_soap_response_round_trip() -> None:
    """Full response model should serialize and deserialize cleanly."""
    response = SoapResponse(
        note=SoapNote(
            subjective=[
                SOAPItem(
                    text="Exertional chest discomfort",
                    evidence_quote="tight feeling in my chest when I climb stairs",
                    groundedness_score=0.94,
                )
            ],
            objective=[],
            assessment=[
                SOAPItem(
                    text="Query angina",
                    evidence_quote="tight feeling in my chest when I climb stairs",
                )
            ],
            plan=[
                SOAPItem(
                    text="ECG at rest",
                    evidence_quote="tight feeling in my chest when I climb stairs",
                )
            ],
        ),
        scores=EvaluationScores(
            tier_0_structural=Tier0Validation(
                passed=True,
                resolved_citations_count=3,
            ),
            tier_1_groundedness=Tier1Evaluation(
                groundedness_score=0.94,
                fabrication_flags=[],
            ),
            tier_2_clinical=[
                Tier2ClinicalRubric(
                    section="assessment",
                    appropriateness=RubricDimension(rationale="ok", score=5),
                    completeness=RubricDimension(rationale="ok", score=4),
                    placement=RubricDimension(rationale="ok", score=5),
                    conciseness=RubricDimension(rationale="ok", score=5),
                    safety=RubricDimension(rationale="ok", score=5),
                )
            ],
            composite_confidence_score=0.87,
            needs_review=False,
        ),
        ehr=EHRMetadata(
            synced=True,
            document_reference_id="DocumentReference/dr-abc123",
            resource_type="DocumentReference",
            fhir_endpoint="http://mock-ehr:8080/fhir",
        ),
        flags=[],
        metadata=ExecutionMetadata(
            generator_model="claude-3-5-sonnet-20241022",
            judge_model="claude-3-haiku-20240307",
            tokens_used=1842,
            latency_ms=3200,
            preprocessing_applied=["clean", "speaker-turn split", "PII de-id"],
        ),
    )
    data = response.model_dump()
    restored = SoapResponse.model_validate(data)
    assert restored == response


def test_generation_settings_defaults() -> None:
    """Defaults should match the proposal's recommended values."""
    settings = GenerationSettings()
    assert settings.language == Language.EN
    assert settings.include_pii_deid is True
    assert settings.confidence_threshold == 0.6


def test_soap_request_minimal() -> None:
    """Only transcript is required; everything else is optional."""
    req = SoapRequest(transcript="Doctor: Hi.\nPatient: Hello.")
    assert req.patient_id is None
    assert req.encounter_id is None
    assert req.settings is None