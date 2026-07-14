# MedCoPilot — Architecture & Principles

Container 3 of the MedCoPilot platform: the SOAP Documentation Service.
One job: **transcript in, verifiable SOAP note out.**

## Core principle — verifiability

A SOAP note is only useful for an EHR if every fact can be traced back to
what was actually said. Therefore every extracted claim carries a verbatim
`evidence_quote` and a reference to the dialogue turn it came from. The LLM
does the hard part (extraction); everything around it is deterministic, so
failures are caught without model cost.

## Pipeline

```
transcript
  → Dialogue (speaker-labelled turns)
  → Planner (LLM): segments the visit into clinical problems
  → Extractor (LLM, one call per problem): SOAP note with per-claim citations
  → Tier 0 gate (deterministic): citations resolve? sections populated?
  → Tier 1 scorer (deterministic): lexical grounding per claim + review flags
  → ICD coding (BM25): candidate ICD-10 codes for the Assessment
  → ReportView: one joined tree per note → REST DTO / demo UI
  → clinician approval → conditional FHIR DocumentReference create
```

One dialogue may yield **multiple notes** (one per clinical problem) — a
supervisor ruling from 2026-07-01.

## Quality evaluation — the tier stack

| Tier | What | How | Status |
|---|---|---|---|
| 0 | Structural gate | Evidence quote must resolve to a span of the turn it cites (normalized substring match). Catches fabricated citations with zero model cost. Empty sections are flagged, not failed. | in service |
| 1 | Groundedness | Clipped unigram precision of the quote against its source turn; claims below `SCORING__REVIEW_THRESHOLD` (default 0.6) are flagged; any flag ⇒ note `needs_review`. | in service |
| 2 | Clinical quality | LLM-as-judge, 3-class rubric (excellent / normal / bad) over ACI-Bench-Refined with manual spot-check and judge-vs-human agreement (Cohen's κ). | offline benchmark (`src/bench/`) |
| 3 | Learned scorer | Fine-tuned cross-encoder on reviewer labels. | future work |

Design decision (supervisor-approved, 2026-07-01): Tier 1 stays **lexical**
instead of the originally proposed NLI cross-encoder — cheaper, deterministic,
and strong enough for the extractive S/O sections. NLI remains an option if
the benchmark shows lexical grounding is insufficient.

## Components

- `src/dialogue/` — dialogue aggregate (speaker-labelled turns), in-memory
  repository, creation use cases (structured or raw text).
- `src/soap/extractor/` — planner + extractor LangGraph agents; structured
  output guarantees the SOAP JSON schema by construction.
- `src/soap/score/` — `tier0.py` (structural gate) and `scorer.py` (lexical
  grounding). Both produce side-car aggregates keyed by note/claim ids; the
  domain write-model is never mutated by evaluation.
- `src/soap/coding/` — BM25 retrieval over the Russian ICD-10 (NSI) index,
  candidate codes for the Assessment claim (English ICD-10 migration pending).
- `src/soap/view.py` — the only place the side-car aggregates are joined into
  one linearized read-model consumed by the API and the UI.
- `src/bench/` — offline benchmark: dataset adapter, LLM judge, resumable
  runner, report + spot-check artifacts.
- `src/ehr/` — mock-EHR application boundary: report workflow state,
  clinician approval, patient-context DTOs, repository and gateway contracts.
- `src/infra/fhir.py` — FHIR R4 adapter for the external HAPI service. It reads
  bounded pre-visit context and maps approved reports to `DocumentReference`.
- `src/app/`, `src/di/`, `src/config/` — FastAPI app, dependency container,
  env-driven settings. The LLM client is pluggable (any OpenAI-compatible
  endpoint) so a local model can replace the cloud API.

## Mock EHR boundary

The mock EHR is a separate HAPI FHIR R4 process; it is never embedded into the
application. A Dialogue may carry explicit `Patient/{id}` and `Encounter/{id}`
references. The context read verifies that the Encounter belongs to the Patient
before searching Conditions, allergies, medications, and observations.

Fixture clinical resources are divided by the tag system
`urn:medcopilot:fixture-phase`. Only `pre-visit` resources are returned, and a
Condition referencing the current Encounter is excluded even if it is tagged
incorrectly. This keeps the post-visit/gold diagnosis out of application
context. Every returned item retains its FHIR resource reference as provenance.

Report synchronization is a separate state machine:

```text
generated draft → clinician-approved → syncing → synced | failed
```

Only approved, linked reports can be synchronized. Local state prevents a
second write after success; the remote request also uses the report id as a
stable identifier plus FHIR `If-None-Exist`. The synchronized artifact is one
final `DocumentReference` containing the approved SOAP report as Markdown.

## Data

- **ACI-Bench-Refined** (evaluation only, AGPL — downloaded by script, never
  committed). No model is trained or fine-tuned; the tier stack is designed
  to need zero training data.
- **Mock EHR fixtures** — one project-owned deterministic smoke case split into
  pre-visit and post-visit transaction bundles. The manifest explicitly maps
  case → Dialogue → Encounter → Patient → Condition. Dataset adapters for
  PriMock57 and Synthea remain future work.

## Known limitations

- The EHR adapter is development-only: no SMART-on-FHIR/OAuth, vendor profiles,
  production persistence, audit log, or PII controls are implemented.
- EHR context is exposed through a bounded API but is not yet injected into the
  SOAP extractor; dialogue turns remain the only evidence source for claims.
- Report workflow state is currently in memory and is lost when the app restarts.
- The planner's segmentation (`turn_indices`) is not yet consumed by the
  extractor — each extractor call currently sees the full dialogue.
- Tier 2 runs offline only; the service returns Tier 0/1 signals.
