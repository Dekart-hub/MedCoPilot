# MedCoPilot — SOAP Documentation Service

A standalone microservice that turns a raw doctor–patient consultation
transcript into structured, verifiable SOAP notes. Every extracted claim
carries a verbatim evidence quote from the transcript, a deterministic
structural gate (Tier 0) verifies the citations, and a lexical grounding
score (Tier 1) flags claims that need human review.

## Status

Implemented:

- Extraction pipeline: planner segments the visit into clinical problems,
  an extractor emits one SOAP note per problem, each claim cites its source turn
- Tier 0 structural gate: citation-to-transcript resolution + empty-section flags
- Tier 1 lexical grounding: per-claim scores, threshold-based `is_flagged`,
  note-level `needs_review`
- ICD-10 candidate coding for the Assessment section (BM25 baseline)
- External HAPI FHIR R4 mock EHR: explicit patient/encounter linkage,
  bounded A/P context enrichment with separate FHIR provenance, clinician
  approval, and idempotent `DocumentReference` sync
- REST API (FastAPI) + Streamlit demo UI + Docker image
- Offline quality benchmark on ACI-Bench-Refined (LLM-as-judge, see below)

Not yet implemented (planned): production EHR authentication/vendor adapters
and the in-service Tier 2 clinical rubric (an offline LLM judge exists in the
benchmark).

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design principles.

## Quick start

## Offline Benchmarking

Quality of generated SOAP notes is measured offline on
[ACI-Bench-Refined](https://huggingface.co/datasets/ClinicianFOCUS/ACI-Bench-Refined)
(3-class LLM-as-judge: excellent / normal / bad, plus a manual spot-check).

```bash
uv sync --extra bench
uv run python scripts/fetch_aci_bench.py                  # download dataset (AGPL, gitignored)
uv run python scripts/bench_soap.py --split test          # reportable run (20 encounters)
uv run python scripts/bench_soap.py --resume <run_id>     # finish an interrupted run
uv run python scripts/bench_agreement.py runs/soap_bench/<run_id>/spot_check.csv
```

Artifacts land in `runs/soap_bench/<run_id>/`: `report.json`, `summary.md`
(class distribution, subscores, funnel, limitations) and `spot_check.csv`
for human verification.

## Project Structure
Prerequisites: Python 3.12+, [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Dekart-hub/MedCoPilot.git
cd MedCoPilot
uv sync                      # create .venv and install dependencies
cp .env.example .env         # then put your real OPENAI__API_KEY into .env
make run                     # serve http://localhost:8000 (Swagger UI at /docs)
```

Useful targets (see `make help`):

```bash
make dev            # run with auto-reload
make test           # run the test suite (uv run pytest -q)
make ui             # Streamlit demo UI (backend must be running separately)
make docker-build   # build the Docker image
make docker-run     # run the container with env from .env
make mock-ehr-up    # start the external HAPI FHIR R4 mock
make mock-ehr-seed  # load safe pre-visit fixtures
make mock-ehr-live-test   # run live adapter + FastAPI integration checks
make mock-ehr-down        # remove the mock service and its ephemeral data
```

### Docker

```bash
docker build -t medcopilot:latest .
docker run --rm --env-file .env -p 8000:8000 medcopilot:latest
curl http://localhost:8000/health   # {"status":"ok"}
```

## Configuration

Environment variables (nested sections use `__` as delimiter, read from `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI__API_KEY` | — (required) | key for the OpenAI-compatible LLM endpoint |
| `OPENAI__MODEL` | `gpt-4o-mini` | extraction model |
| `OPENAI__BASE_URL` | (OpenAI) | alternative OpenAI-compatible endpoint |
| `OPENAI__TEMPERATURE` | `0.0` | sampling temperature |
| `SCORING__REVIEW_THRESHOLD` | `0.6` | Tier 1 grounding score below which a claim is flagged |
| `EHR__ENABLED` | `false` | enable the external development-only mock EHR |
| `EHR__BASE_URL` | `http://localhost:8080/fhir` | FHIR R4 base URL |
| `EHR__TIMEOUT_SECONDS` | `10` | mock EHR request timeout |
| `EHR__IDENTIFIER_SYSTEM` | `urn:medcopilot:soap-report` | stable identifier system for conditional create |

## API

| Method & path | Purpose |
|---|---|
| `GET /health` | liveness |
| `GET /ready` | readiness (dependencies built) |
| `GET /api/v1/dialogues` | list dialogues |
| `POST /api/v1/dialogues` | create a dialogue from structured turns |
| `POST /api/v1/dialogues/from-text` | create a dialogue from raw text (one `role text` line per turn) |
| `GET /api/v1/dialogues/{id}` | fetch a dialogue |
| `GET /api/v1/ehr/dialogues/{id}/context` | fetch tagged pre-visit patient context |
| `POST /api/v1/reports` | generate a scored SOAP report for a dialogue |
| `GET /api/v1/reports/{id}/workflow` | inspect approval/sync state |
| `POST /api/v1/reports/{id}/approve` | record clinician approval |
| `POST /api/v1/reports/{id}/ehr-sync` | idempotently sync an approved `DocumentReference` |

Interactive docs: `http://localhost:8000/docs`.

Example — generate a report:

```bash
curl -s -X POST http://localhost:8000/api/v1/dialogues/from-text \
  -H 'Content-Type: application/json' \
  -d '{"text": "person My chest feels tight on the stairs since yesterday\nmedic Any pain radiating?\nperson No, just tight when climbing"}'
# -> {"id": "<dialogue_id>", ...}

curl -s -X POST http://localhost:8000/api/v1/reports \
  -H 'Content-Type: application/json' \
  -d '{"dialogue_id": "<dialogue_id>"}'
```

Response shape (abridged):

```json
{
  "soap_notes": [
    {
      "subjective": {
        "claim": "Chest tightness on exertion since yesterday",
        "evidence_text": "chest feels tight on the stairs since yesterday",
        "turn_id": "…",
        "grounding_score": 0.95,
        "is_flagged": false
      },
      "objective": { "…": "…" },
      "assessment": { "…": "…", "codings": [ { "code": "…", "title": "…" } ] },
      "plan": { "…": "…" },
      "tier0": {
        "passed": true,
        "empty_sections": [],
        "citations_total": 4,
        "citations_resolved": 4
      },
      "needs_review": false,
      "confidence": 0.93
    }
  ],
  "context_status": "not-linked",
  "context_error": null
}
```

## Mock EHR workflow

The mock EHR is an external, unauthenticated HAPI FHIR service and is disabled
by default. Start it, load only the pre-visit fixtures, and enable the adapter:

```bash
make mock-ehr-up
make mock-ehr-seed
EHR__ENABLED=true EHR__BASE_URL=http://localhost:8080/fhir make dev
```

Verify the linked sample through the application API:

```bash
curl -s http://localhost:8000/api/v1/ehr/dialogues/\
11111111-1111-1111-1111-111111111111/context
```

The seeded sample dialogue is deterministically linked to
`Patient/mock-patient-001` and `Encounter/mock-encounter-001`. Clinical context
is restricted to resources tagged `urn:medcopilot:fixture-phase|pre-visit` and
never includes a Condition attached to the current Encounter. The separate
post-visit bundle contains the gold diagnosis and can be loaded specifically to
test that boundary:

```bash
make mock-ehr-post-visit
make mock-ehr-live-test
```

The live test performs both a direct FHIR adapter check and a FastAPI context
check. It also synchronizes the same stable report twice and verifies HAPI
returns the same `DocumentReference`. Loading the post-visit bundle first proves
that the current diagnosis can exist in HAPI without appearing in app context.

Generating a report for a linked dialogue fetches this snapshot before LLM
extraction and supplies at most ten Conditions, allergies, medications, and
observations per category to Assessment/Plan. Used resources are returned in
`context_references`, separately from transcript `evidence_text` and `turn_id`.
Each reference must exactly match the fetched snapshot. If context retrieval
fails, the report is still generated from the transcript with
`context_status=unavailable` and `needs_review=true`.

Reports are stored as drafts in memory. Approve with
`{"clinician_ref":"Practitioner/mock-gp-001"}` before sync. Sync creates a FHIR
`DocumentReference` with a stable report identifier and `If-None-Exist`; a
repeated local request returns the prior result without another remote write.
See [mock_ehr/README.md](mock_ehr/README.md) for fixture/linkage details.

```bash
curl -s -X POST http://localhost:8000/api/v1/reports/<report_id>/approve \
  -H 'Content-Type: application/json' \
  -d '{"clinician_ref":"Practitioner/mock-gp-001"}'

curl -s -X POST \
  http://localhost:8000/api/v1/reports/<report_id>/ehr-sync
```

The mock uses ephemeral H2 storage. To return to a clean pre-visit-only server:

```bash
make mock-ehr-down
make mock-ehr-up
make mock-ehr-seed
```

## Offline benchmarking

Quality of generated SOAP notes is measured offline on
[ACI-Bench-Refined](https://huggingface.co/datasets/ClinicianFOCUS/ACI-Bench-Refined)
(3-class LLM-as-judge: excellent / normal / bad, plus a manual spot-check).

```bash
uv sync --extra bench
uv run python scripts/fetch_aci_bench.py                  # download dataset (AGPL, gitignored)
uv run python scripts/bench_soap.py --split test          # reportable run (20 encounters)
uv run python scripts/bench_soap.py --resume <run_id>     # finish an interrupted run
uv run python scripts/bench_agreement.py runs/soap_bench/<run_id>/spot_check.csv
```

Artifacts land in `runs/soap_bench/<run_id>/`: `report.json`, `summary.md`
(class distribution, subscores, funnel, limitations) and `spot_check.csv`
for human verification.

## Data & licensing

No model is trained or fine-tuned in this project — the tiered evaluation is
deliberately designed to need zero training data. ACI-Bench-Refined is used
for **evaluation only**; it is AGPL-licensed and therefore never committed to
this repository (`data/` is gitignored) — reproduce it locally with
`scripts/fetch_aci_bench.py`.

## Project structure

```
MedCoPilot/
├── src/
│   ├── app/           # FastAPI app + /api/v1 routes and DTOs
│   ├── bench/         # offline ACI-Bench benchmark (runner, judge, report)
│   ├── config/        # pydantic-settings (env-driven)
│   ├── di/            # dependency container and FastAPI deps
│   ├── dialogue/      # dialogue domain: turns, repository, use cases
│   ├── ehr/           # mock-EHR workflow, approval state, gateway contracts
│   ├── infra/         # OpenAI-compatible LLM and FHIR R4 adapters
│   ├── shared/        # ids, entities, prompts, langgraph agent
│   └── soap/          # SOAP domain: extractor, Tier 0 gate, scorer, coding, view
├── scripts/           # dataset fetch, benchmark CLI, eval utilities
├── tests/             # pytest suite
├── mock_ehr/          # linkage manifest and pre/post-visit FHIR fixtures
├── compose.mock-ehr.yml
├── ui/                # Streamlit demo client
├── Dockerfile
└── Makefile           # run / dev / test / ui / docker-* targets
```

## Team

Artem Levakov · Ivan Alpatov · Dmitrii Naumov · Anatolii Astanin ·
Aleksandr Romanov (Lead)

Academic course project (Industrial Project).
