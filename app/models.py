"""
Pydantic models for the MedCoPilot SOAP Documentation Service.

Contract is derived directly from the project proposal (Section 3, Figure 1):
    - Input:  POST /generate-soap { transcript: "..." }
    - Output: SOAP JSON { note · evidence · scores · flags }

Key design decisions:
    1. Every SOAP item (S/O/A/P) carries a verbatim evidence quote (citation).
    2. Tier 0 is a deterministic gate; if it fails, Tier 1/2 are None (short-circuit).
    3. Tier 1 (NLI groundedness) applies only to S/O (extractive).
    4. Tier 2 (LLM-as-judge rubric) applies only to A/P (inferential).
    5. EHR sync produces a FHIR DocumentReference with full metadata.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #

class Language(str, Enum):
    """Supported languages for transcript and output."""
    EN = "en"
    RU = "ru"


class GenerationSettings(BaseModel):
    """
    Optional generation settings.
    Kept optional so the minimal contract { transcript: "..." } still works.
    """
    language: Language = Field(
        Language.EN,
        description="Language of the transcript and generated output"
    )
    include_pii_deid: bool = Field(
        True,
        description="Whether to apply PII de-identification during preprocessing"
    )
    confidence_threshold: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Groundedness threshold below which a claim is flagged"
    )


class SoapRequest(BaseModel):
    """
    Input schema for the SOAP generation endpoint.
    Minimal contract: { transcript: "..." }. All other fields are optional.
    """
    transcript: str = Field(
        ...,
        min_length=10,
        max_length=50_000,
        description="Raw, unformatted text transcript of the doctor-patient consultation.",
        examples=[
            "Doctor: Hello, what brings you here today?\n"
            "Patient: My chest feels tight on the stairs. It started yesterday.\n"
            "Doctor: Any pain radiating?\n"
            "Patient: No, just tight when climbing."
        ]
    )
    patient_id: Optional[str] = Field(
        None,
        description="Target Patient Identifier in the EHR system for FHIR DocumentReference.subject",
        examples=["patient-uuid-12345"]
    )
    encounter_id: Optional[str] = Field(
        None,
        description="Target Encounter/Visit Identifier in the EHR",
        examples=["encounter-uuid-67890"]
    )
    settings: Optional[GenerationSettings] = Field(
        None,
        description="Optional generation settings"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transcript": (
                        "Doctor: Hello, what brings you here today?\n"
                        "Patient: My chest feels tight on the stairs.\n"
                        "Doctor: Any pain radiating?\n"
                        "Patient: No, just tight when climbing."
                    ),
                    "patient_id": "patient-uuid-12345",
                    "encounter_id": "encounter-uuid-67890",
                    "settings": {
                        "language": "en",
                        "include_pii_deid": True,
                        "confidence_threshold": 0.6
                    }
                }
            ]
        }
    }


# --------------------------------------------------------------------------- #
# SOAP note structure
# --------------------------------------------------------------------------- #

class SOAPItem(BaseModel):
    """
    A single SOAP note item with its verbatim evidence quote (citation).

    Per the proposal (Section 2):
        "Each extracted item carries a verbatim evidence quote from the
        transcript — its citation."

    - For S/O (extractive): groundedness_score is populated via DeBERTa-MNLI.
    - For A/P (inferential): groundedness_score is None because NLI would
      wrongly penalize valid clinical inference (see proposal Section 5).
    """
    text: str = Field(..., description="Text of the note item (clinical claim or reasoning)")
    evidence_quote: str = Field(
        ...,
        description="Verbatim quote from the transcript that supports this item"
    )
    groundedness_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="P(entailment) from DeBERTa-MNLI. Populated only for S/O (Tier 1)"
    )
    is_flagged: bool = Field(
        False,
        description="True if this item is flagged as a possible hallucination"
    )


class SoapNote(BaseModel):
    """
    The structured SOAP note. Each section is a list of SOAPItem instances.
    All four sections must be present (Tier 0 validates this).
    """
    subjective: List[SOAPItem] = Field(
        default_factory=list,
        description="S — subjective data (symptoms, history, patient statements)"
    )
    objective: List[SOAPItem] = Field(
        default_factory=list,
        description="O — objective data (vitals, exam findings, labs)"
    )
    assessment: List[SOAPItem] = Field(
        default_factory=list,
        description="A — assessment/diagnosis (inferential)"
    )
    plan: List[SOAPItem] = Field(
        default_factory=list,
        description="P — plan (treatment, workup, follow-up)"
    )


# --------------------------------------------------------------------------- #
# Tiered quality evaluation
# --------------------------------------------------------------------------- #

class Tier0Validation(BaseModel):
    """
    Tier 0: Structural deterministic gate.

    Per the proposal (Section 5):
        "Schema is valid, all four SOAP sections are populated, JSON is
        well-formed, and every cited evidence quote actually resolves to a
        span in the transcript (plain string match). Failures short-circuit
        the request. No model, no cost."
    """
    passed: bool = Field(
        ...,
        description="True if the output structurally matches the transcript and schema rules"
    )
    resolved_citations_count: int = Field(
        ...,
        ge=0,
        description="Number of verbatim quotes successfully validated against the raw text"
    )
    failure_reason: Optional[str] = Field(
        None,
        description="Human-readable reason if passed=False"
    )


class Tier1Evaluation(BaseModel):
    """
    Tier 1: Groundedness verification for extractive sections (S/O).

    Per the proposal (Section 5):
        "Run (evidence ⊢ text) through a pretrained NLI cross-encoder
        (DeBERTa-v3-MNLI) → P(entailment) per claim. Groundedness score =
        aggregate entailment; any claim below threshold is flagged."
    """
    groundedness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate fraction of S/O claims entailed by their cited span"
    )
    fabrication_flags: List[str] = Field(
        default_factory=list,
        description="Specific S/O claims flagged as potential hallucinations"
    )


class RubricDimension(BaseModel):
    """
    Pointwise analytic rubric dimension evaluated by the LLM-as-judge.

    Per the proposal (Section 5):
        "Emitting a one-sentence rationale before each score (G-Eval /
        chain-of-thought style) as structured JSON."
    """
    rationale: str = Field(
        ...,
        description="A one-sentence analytical justification generated prior to the score"
    )
    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Integer rating from 1 (poor) to 5 (excellent)"
    )


class Tier2ClinicalRubric(BaseModel):
    """
    Tier 2: Clinical quality rubric applied to inferential sections (A/P).

    Per the proposal (Section 5), five dimensions:
        1. Clinical appropriateness
        2. Completeness (follow-up, safety-netting)
        3. Correct section placement
        4. Conciseness (no redundancy/padding)
        5. Safety / no-fabrication
    """
    section: Literal["assessment", "plan"] = Field(
        ...,
        description="Which SOAP section this rubric evaluates"
    )
    appropriateness: RubricDimension = Field(
        ..., description="Is the A/P clinically logical given S+O?"
    )
    completeness: RubricDimension = Field(
        ..., description="Are essential safety nets, follow-ups present?"
    )
    placement: RubricDimension = Field(
        ..., description="Are all points categorized under the correct SOAP headers?"
    )
    conciseness: RubricDimension = Field(
        ..., description="Is the text clear of redundancy and LLM verbosity bias?"
    )
    safety: RubricDimension = Field(
        ..., description="Any clinical claims with no plausible basis?"
    )


class EvaluationScores(BaseModel):
    """
    Aggregated three-tier evaluation framework.

    Per the proposal (Section 5):
        "composite = Tier0(pass/fail gate) → weighted blend of Tier-1
        groundedness and Tier-2 rubric (e.g. 0.5·groundedness + 0.5·(rubric/5)),
        with any unsupported S/O claim forcing a 'needs review' flag."

    If tier_0_structural.passed is False, tier_1 and tier_2 are None
    (short-circuit — no model cost).
    """
    tier_0_structural: Tier0Validation
    tier_1_groundedness: Optional[Tier1Evaluation] = Field(
        None,
        description="None if tier_0_structural.passed is False (short-circuit)"
    )
    tier_2_clinical: Optional[List[Tier2ClinicalRubric]] = Field(
        None,
        description="Rubric scores for Assessment and Plan. None if tier_0 failed."
    )
    composite_confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted blend of Tier 1 groundedness and Tier 2 rubric"
    )
    needs_review: bool = Field(
        ...,
        description="True if any S/O claim is unsupported or composite is low"
    )


# --------------------------------------------------------------------------- #
# EHR sync metadata (FHIR DocumentReference)
# --------------------------------------------------------------------------- #

class EHRMetadata(BaseModel):
    """
    Metadata about the sync to the mock EHR (FHIR DocumentReference).

    Per the proposal (Section 2):
        "EHR sync: pushes the validated note to a mock EHR as a FHIR
        DocumentReference (demonstrates realistic integration)."
    """
    synced: bool = Field(
        ...,
        description="Whether the note was successfully pushed to the mock EHR"
    )
    document_reference_id: Optional[str] = Field(
        None,
        description="ID of the created FHIR DocumentReference resource",
        examples=["DocumentReference/example-1234"]
    )
    resource_type: Literal["DocumentReference"] = Field(
        "DocumentReference",
        description="FHIR resource type (fixed)"
    )
    fhir_endpoint: Optional[str] = Field(
        None,
        description="URL of the mock EHR that received the document",
        examples=["http://mock-ehr:8080/fhir"]
    )
    failure_reason: Optional[str] = Field(
        None,
        description="Human-readable reason if synced=False"
    )


# --------------------------------------------------------------------------- #
# Execution metadata
# --------------------------------------------------------------------------- #

class ExecutionMetadata(BaseModel):
    """
    System observability telemetry: processing statistics, latency, resource costs.
    """
    generator_model: str = Field(
        ...,
        description="The model name used to extract and construct the SOAP note"
    )
    judge_model: Optional[str] = Field(
        None,
        description="The independent model name utilized as the Tier 2 clinical judge"
    )
    tokens_used: Optional[int] = Field(
        None,
        ge=0,
        description="Total tokens consumed across all LLM calls"
    )
    latency_ms: int = Field(
        ...,
        ge=0,
        description="Total processing timeline elapsed in milliseconds"
    )
    preprocessing_applied: List[str] = Field(
        default_factory=list,
        description="Completed pipeline preprocessing routines (e.g., PII de-id, speaker split)"
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional telemetry (speaker turns, PII redactions, etc.)"
    )


# --------------------------------------------------------------------------- #
# Top-level response
# --------------------------------------------------------------------------- #

class SoapResponse(BaseModel):
    """
    Final API response.

    Contract from the proposal (Section 3, Figure 1):
        SOAP JSON { note · evidence · scores · flags }
    """
    note: SoapNote
    scores: EvaluationScores
    ehr: Optional[EHRMetadata] = Field(
        None,
        description="FHIR DocumentReference sync metadata"
    )
    flags: List[str] = Field(
        default_factory=list,
        description="Top-level warning flags for the human-in-the-loop review interface"
    )
    metadata: ExecutionMetadata

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "note": {
                        "subjective": [
                            {
                                "text": "Patient reports chest tightness on stairs for ~1 day",
                                "evidence_quote": "My chest feels tight on the stairs. It started yesterday.",
                                "groundedness_score": 0.96,
                                "is_flagged": False
                            }
                        ],
                        "objective": [],
                        "assessment": [
                            {
                                "text": "Exertional chest discomfort, query angina",
                                "evidence_quote": "My chest feels tight on the stairs",
                                "groundedness_score": None,
                                "is_flagged": False
                            }
                        ],
                        "plan": [
                            {
                                "text": "Refer to cardiology for exercise tolerance test within 1 week",
                                "evidence_quote": "My chest feels tight on the stairs",
                                "groundedness_score": None,
                                "is_flagged": False
                            }
                        ]
                    },
                    "scores": {
                        "tier_0_structural": {
                            "passed": True,
                            "resolved_citations_count": 3,
                            "failure_reason": None
                        },
                        "tier_1_groundedness": {
                            "groundedness_score": 0.96,
                            "fabrication_flags": []
                        },
                        "tier_2_clinical": [
                            {
                                "section": "assessment",
                                "appropriateness": {"rationale": "Angina query is warranted.", "score": 5},
                                "completeness": {"rationale": "Differential could be broader.", "score": 4},
                                "placement": {"rationale": "Correctly placed.", "score": 5},
                                "conciseness": {"rationale": "Concise.", "score": 5},
                                "safety": {"rationale": "No fabrication.", "score": 5}
                            }
                        ],
                        "composite_confidence_score": 0.94,
                        "needs_review": False
                    },
                    "ehr": {
                        "synced": True,
                        "document_reference_id": "DocumentReference/dr-abc123",
                        "resource_type": "DocumentReference",
                        "fhir_endpoint": "http://mock-ehr:8080/fhir",
                        "failure_reason": None
                    },
                    "flags": [],
                    "metadata": {
                        "generator_model": "claude-3-5-sonnet-20241022",
                        "judge_model": "claude-3-haiku-20240307",
                        "tokens_used": 1842,
                        "latency_ms": 3200,
                        "preprocessing_applied": ["clean", "speaker-turn split", "PII de-id"],
                        "extra": {"speaker_turns": 4, "pii_redacted": 0}
                    }
                }
            ]
        }
    }


# --------------------------------------------------------------------------- #
# Health check
# --------------------------------------------------------------------------- #

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """
    Response for /health with dependency checks.

    Per the proposal, the service depends on:
        - LLM provider (Claude)
        - Mock EHR (FHIR)
    """
    status: HealthStatus
    service: str = "medcopilot-soap"
    version: str = "1.0.0"
    checks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Per-dependency check results",
        examples=[{"llm_reachable": True, "fhir_mock_reachable": True}]
    )