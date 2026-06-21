# MedCoPilot — SOAP Documentation Service

A standalone microservice that transforms raw doctor-patient consultation transcripts into structured, verifiable SOAP notes with per-item citations and tiered quality evaluation.

## Overview

MedCoPilot addresses a critical problem in healthcare: doctors spend a large share of each visit on administrative documentation. This service takes a raw text transcript of a doctor-patient dialogue and returns a structured, validated, and verifiable SOAP note ready for EHR integration — where every extracted fact can be traced back to what was actually said.

### Key Features

- **Structured SOAP Extraction**: Converts raw transcripts into Subjective, Objective, Assessment, and Plan sections
- **Per-Item Citations**: Every extracted claim carries a verbatim evidence quote from the transcript
- **Tiered Quality Evaluation**:
  - **Tier 0**: Structural validation (deterministic gate)
  - **Tier 1**: Groundedness verification via NLI (DeBERTa-v3-MNLI) for S/O sections
  - **Tier 2**: Clinical quality rubric (LLM-as-judge) for A/P sections
- **FHIR Integration**: Syncs validated notes to mock EHR as FHIR DocumentReference
- **Human-in-the-Loop**: Review interface for corrections and training data generation

## Architecture

```
Transcript → Preprocessing → LLM Extraction → Tier 0 (Structural) → Tier 1 (NLI) → Tier 2 (Rubric) → EHR Sync → SOAP JSON
```

The service follows a deterministic spine with LLM stages wrapped by validation layers, ensuring failures are caught without model cost.

## Project Structure

```
MedCoPilot/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application with endpoints
│   └── models.py        # Pydantic models (request/response schemas)
├── tests/
│   └── test_api.py      # Pytest test suite
├── pyproject.toml       # Project configuration and dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd MedCoPilot
   ```

2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```
   
   This will:
   - Create a virtual environment (`.venv/`)
   - Install all dependencies from `pyproject.toml`
   - Generate `uv.lock` for reproducible builds

3. **Verify installation**:
   ```bash
   uv run python -c "import fastapi; print(f'FastAPI {fastapi.__version__} installed')"
   ```

### Running the Server

Start the FastAPI development server with auto-reload:

```bash
uv run uvicorn app.main:app --reload
```

The server will be available at: **http://localhost:8000**

### Running Tests

Execute the test suite:

```bash
uv run pytest tests/ -v
```

Expected output:
```
tests/test_api.py::test_health_returns_200 PASSED
tests/test_api.py::test_health_response_model_parses PASSED
...
======================== 18 passed in X.XXs =========================
```

## API Documentation

### Interactive Swagger UI

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Endpoints

#### `GET /health`

Health check endpoint for orchestrators (Kubernetes, load balancers).

**Response**:
```json
{
  "status": "healthy",
  "service": "medcopilot-soap",
  "version": "1.0.0",
  "checks": {
    "llm_reachable": true,
    "fhir_mock_reachable": true
  }
}
```

#### `POST /generate-soap`

Main business endpoint: generates a verifiable SOAP note from a consultation transcript.

**Request** (minimal contract):
```json
{
  "transcript": "Doctor: What brings you in today?\nPatient: My chest feels tight on the stairs. It started yesterday.\nDoctor: Any pain radiating?\nPatient: No, just tight when climbing."
}
```

**Request** (full contract with optional fields):
```json
{
  "transcript": "Doctor: Hello, what brings you here today?\nPatient: My chest feels tight on the stairs. It started yesterday.\nDoctor: Any pain radiating?\nPatient: No, just tight when climbing.",
  "patient_id": "patient-uuid-12345",
  "encounter_id": "encounter-uuid-67890",
  "settings": {
    "language": "en",
    "include_pii_deid": true,
    "confidence_threshold": 0.6
  }
}
```

**Response**:
```json
{
  "note": {
    "subjective": [
      {
        "text": "Patient reports chest tightness on stairs for ~1 day",
        "evidence_quote": "My chest feels tight on the stairs. It started yesterday.",
        "groundedness_score": 0.96,
        "is_flagged": false
      }
    ],
    "objective": [],
    "assessment": [
      {
        "text": "Exertional chest discomfort, query angina",
        "evidence_quote": "My chest feels tight on the stairs",
        "groundedness_score": null,
        "is_flagged": false
      }
    ],
    "plan": [
      {
        "text": "Refer to cardiology for exercise tolerance test within 1 week",
        "evidence_quote": "My chest feels tight on the stairs",
        "groundedness_score": null,
        "is_flagged": false
      }
    ]
  },
  "scores": {
    "tier_0_structural": {
      "passed": true,
      "resolved_citations_count": 3,
      "failure_reason": null
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
    "needs_review": false
  },
  "ehr": {
    "synced": true,
    "document_reference_id": "DocumentReference/dr-abc123",
    "resource_type": "DocumentReference",
    "fhir_endpoint": "http://mock-ehr:8080/fhir",
    "failure_reason": null
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
```

## Testing

The test suite covers:

- Health endpoint functionality
- Request validation (Pydantic)
- Minimal and full contract acceptance
- Response structure verification
- Pydantic model round-trips
- Tier 0 short-circuit behavior
- Edge cases and error handling

Run tests with verbose output:
```bash
uv run pytest tests/ -v
```

Run tests with coverage (requires `pytest-cov`):
```bash
uv run pytest tests/ --cov=app --cov-report=term-missing
```

## Development

### Adding Dependencies

Add new dependencies using uv:
```bash
uv add <package-name>
```

Example:
```bash
uv add httpx2  # For async HTTP client
```

### Code Style

The project uses:
- **Pydantic v2** for data validation
- **FastAPI** for REST API
- **pytest** for testing
- Type hints throughout

### Project Configuration

All project metadata and dependencies are defined in `pyproject.toml`:

```toml
[project]
name = "medcopilot"
version = "0.1.0"
description = "SOAP Documentation Service for medical transcripts"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.138.0",
    "uvicorn>=0.49.0",
    "pydantic>=2.13.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=9.1.1",
    "httpx>=0.28.1",
]
```

## Current Status

### Implemented
- FastAPI service with two endpoints (`/health`, `/generate-soap`)
- Complete Pydantic models matching the proposal contract
- Per-item citations (evidence quotes) for all SOAP sections
- Three-tier quality evaluation structure
- FHIR DocumentReference metadata
- Comprehensive test suite (18 tests)
- Swagger UI with examples

## Team

- Artem Levakov
- Ivan Alpatov
- Dmitrii Naumov
- Anatolii Astanin
- Aleksandr Romanov (Lead)

## License

This project is developed as part of an academic course.

---

**Note**: This is currently a mock implementation demonstrating the API contract. The actual pipeline (LLM extraction, NLI evaluation, EHR sync) is planned for subsequent development phases.