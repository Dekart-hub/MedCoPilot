"""
FastAPI application for the MedCoPilot SOAP Documentation Service.

Two endpoints:
    - GET  /health         — liveness + readiness probe for orchestrators
    - POST /generate-soap  — main business endpoint: transcript -> SOAP note

The POST endpoint orchestrates the pipeline from the proposal:
    1. Input validation (Pydantic, automatic)
    2. Preprocessing (clean · speaker-turn split · PII de-id)
    3. LLM Extraction -> items with evidence quotes
    4. Tier 0 — structural validation (deterministic gate)
    5. Tier 1 — groundedness via DeBERTa-MNLI (S/O only)
    6. Tier 2 — LLM-as-judge rubric (A/P only)
    7. Composite score + hallucination flags
    8. EHR sync -> FHIR DocumentReference
    9. Return SOAP JSON { note · evidence · scores · flags }
"""

from fastapi import FastAPI, HTTPException, status

from .models import (
    EHRMetadata,
    EvaluationScores,
    ExecutionMetadata,
    HealthResponse,
    HealthStatus,
    RubricDimension,
    SOAPItem,
    SoapNote,
    SoapRequest,
    SoapResponse,
    Tier0Validation,
    Tier1Evaluation,
    Tier2ClinicalRubric,
)

app = FastAPI(
    title="MedCoPilot — SOAP Documentation Service",
    version="1.0.0",
    description=(
        "Transcript in, verifiable SOAP note out. "
        "Every extracted fact traces back to what was actually said."
    ),
)


# --------------------------------------------------------------------------- #
# Health check
# --------------------------------------------------------------------------- #

@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["System"],
    summary="Liveness and readiness probe",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for orchestrators (Kubernetes, load balancers).

    Checks reachability of external dependencies:
        - LLM provider (Claude)
        - Mock EHR (FHIR)
    """
    # TODO: replace with real dependency checks
    checks = {
        "llm_reachable": True,
        "fhir_mock_reachable": True,
    }
    all_ok = all(checks.values())

    return HealthResponse(
        status=HealthStatus.HEALTHY if all_ok else HealthStatus.DEGRADED,
        checks=checks,
    )


# --------------------------------------------------------------------------- #
# Main business endpoint
# --------------------------------------------------------------------------- #

@app.post(
    "/generate-soap",
    response_model=SoapResponse,
    status_code=status.HTTP_200_OK,
    tags=["Core Processing"],
    summary="Generate a verifiable SOAP note from a consultation transcript",
)
async def generate_soap(request: SoapRequest) -> SoapResponse:
    """
    Main business endpoint.

    Pipeline:
        1. Validate input (Pydantic does this automatically)
        2. Preprocessing: clean · speaker-turn split · PII de-id
        3. LLM Extraction (Claude) — each item = { text, evidence_quote }
        4. Tier 0 — structural validation (deterministic gate)
        5. Tier 1 — groundedness via DeBERTa-MNLI (S/O only)
        6. Tier 2 — LLM-as-judge clinical rubric (A/P only)
        7. Composite score + hallucination flags
        8. EHR sync -> mock EHR (FHIR DocumentReference)
        9. Return SOAP JSON { note · evidence · scores · flags }
    """
    # Input validation is handled by Pydantic before we get here.
    # If we reach this point, request.transcript is a valid non-empty string.

    try:
        # ------------------------------------------------------------------ #
        # TODO: implement the full pipeline here
        # ------------------------------------------------------------------ #
        #   preprocessed = await preprocess(request.transcript, request.settings)
        #   raw_items    = await llm_extract(preprocessed)
        #   soap_note    = map_to_soap(raw_items)
        #
        #   tier0 = validate_structure(soap_note, request.transcript)
        #   if not tier0.passed:
        #       # short-circuit: no Tier 1/2, no model cost
        #       return SoapResponse(
        #           note=soap_note,
        #           scores=EvaluationScores(
        #               tier_0_structural=tier0,
        #               tier_1_groundedness=None,
        #               tier_2_clinical=None,
        #               composite_confidence_score=0.0,
        #               needs_review=True,
        #           ),
        #           flags=["tier_0_failed"],
        #           metadata=...,
        #       )
        #
        #   tier1 = compute_groundedness(soap_note.subjective + soap_note.objective)
        #   tier2 = judge_clinical_quality(soap_note.assessment, soap_note.plan)
        #   composite = blend_scores(tier1, tier2)
        #   ehr_meta = await sync_to_ehr(soap_note, request.patient_id, request.encounter_id)
        #   return SoapResponse(...)
        # ------------------------------------------------------------------ #

        # Mock response demonstrating the finalized JSON schema structure
        return _build_mock_response(request)

    except ValueError as exc:
        # Domain-level validation errors (e.g. LLM returned malformed JSON)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid data: {exc}",
        )
    except Exception as exc:
        # Catch-all for unexpected failures (LLM down, FHIR unreachable, etc.)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during SOAP generation: {exc}",
        )


# --------------------------------------------------------------------------- #
# Mock response builder (temporary — will be replaced by real pipeline)
# --------------------------------------------------------------------------- #

def _build_mock_response(request: SoapRequest) -> SoapResponse:
    """
    Build a mock response that conforms exactly to the finalized JSON schemas.
    This is a placeholder until the real pipeline is implemented.
    """
    soap_note = SoapNote(
        subjective=[
            SOAPItem(
                text="Patient reports chest tightness on stairs for ~1 day",
                evidence_quote="My chest feels tight on the stairs. It started yesterday.",
                groundedness_score=0.96,
                is_flagged=False,
            )
        ],
        objective=[],
        assessment=[
            SOAPItem(
                text="Exertional chest discomfort, query angina",
                evidence_quote="My chest feels tight on the stairs",
                groundedness_score=None,  # A/P items don't have groundedness
                is_flagged=False,
            )
        ],
        plan=[
            SOAPItem(
                text="Refer to cardiology for exercise tolerance test within 1 week",
                evidence_quote="My chest feels tight on the stairs",
                groundedness_score=None,
                is_flagged=False,
            )
        ],
    )

    scores = EvaluationScores(
        tier_0_structural=Tier0Validation(
            passed=True,
            resolved_citations_count=3,
            failure_reason=None,
        ),
        tier_1_groundedness=Tier1Evaluation(
            groundedness_score=0.96,
            fabrication_flags=[],
        ),
        tier_2_clinical=[
            Tier2ClinicalRubric(
                section="assessment",
                appropriateness=RubricDimension(
                    rationale="Angina query is warranted by exertional nature of symptoms.",
                    score=5,
                ),
                completeness=RubricDimension(
                    rationale="Differential could be broader; safety-netting implicit.",
                    score=4,
                ),
                placement=RubricDimension(
                    rationale="All items correctly placed under Assessment.",
                    score=5,
                ),
                conciseness=RubricDimension(
                    rationale="Concise, no verbosity.",
                    score=5,
                ),
                safety=RubricDimension(
                    rationale="No fabricated clinical claims.",
                    score=5,
                ),
            ),
            Tier2ClinicalRubric(
                section="plan",
                appropriateness=RubricDimension(
                    rationale="Cardiology referral is clinically appropriate.",
                    score=5,
                ),
                completeness=RubricDimension(
                    rationale="Timeline specified; safety-netting could be explicit.",
                    score=4,
                ),
                placement=RubricDimension(
                    rationale="Correctly placed under Plan.",
                    score=5,
                ),
                conciseness=RubricDimension(
                    rationale="Direct and actionable.",
                    score=5,
                ),
                safety=RubricDimension(
                    rationale="No unsafe recommendations.",
                    score=5,
                ),
            ),
        ],
        composite_confidence_score=0.94,
        needs_review=False,
    )

    ehr = EHRMetadata(
        synced=True,
        document_reference_id=f"DocumentReference/mock-{request.patient_id or 'anonymous'}",
        resource_type="DocumentReference",
        fhir_endpoint="http://mock-ehr:8080/fhir",
        failure_reason=None,
    )

    metadata = ExecutionMetadata(
        generator_model="claude-3-5-sonnet-20241022",
        judge_model="claude-3-haiku-20240307",
        tokens_used=1842,
        latency_ms=3200,
        preprocessing_applied=["clean", "speaker-turn split", "PII de-id"],
        extra={"speaker_turns": 4, "pii_redacted": 0},
    )

    return SoapResponse(
        note=soap_note,
        scores=scores,
        ehr=ehr,
        flags=[],
        metadata=metadata,
    )