# MedCoPilot

Clinical documentation service (Container 3): converts a doctor–patient dialogue
into structured **SOAP** notes (Subjective / Objective / Assessment / Plan) with
ICD coding and per-note confidence.

The baseline is built task-by-task for traceability. Scope, requirements and the
task breakdown live in GitHub issues — start from user story **#7 (SOAP extraction,
baseline)**.

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

## Development

Install the toolchain and the project's dev dependencies (ruff, mypy, pytest):

```bash
make install          # uv sync
```

Enable the git pre-commit hooks (ruff lint + format, mypy, whitespace/EOF/YAML):

```bash
uv run pre-commit install
```

Run the full quality gate — the same checks CI runs on every push and pull
request:

```bash
make check            # ruff check + mypy + pytest
```

Individual targets are also available: `make lint`, `make format`,
`make typecheck`, `make test`.

## Demo UI

`ui/app.py` is a small [Streamlit](https://streamlit.io/) app for exercising the
service by hand. It has two tabs — **SOAP extraction** and **Correction
workflow** — and is a **demo only** (out of scope for every DoD): a thin HTTP
client over the REST API, kept in its own optional `demo` dependency group so
the core install and the production image stay lean. It imports nothing from
`src` and has no automated tests.

The **SOAP extraction** tab: paste a dialogue, optionally pin a `patient_id`,
and see the extracted `SoapReport` — the four S/O/A/P sections, the ICD code on
Assessment claims, per-note confidence, and each claim linked back to the
dialogue turn it cites.

**The baseline API must be running first** (see `make run`, and bring up the
database and model server per the sections below):

```bash
uv run --group demo streamlit run ui/app.py
```

The `--group demo` flag installs Streamlit on first use. Point the app at a
non-default API with the sidebar field or the `MEDCOPILOT_API_URL` environment
variable (default `http://localhost:8000`).

### Demo: correction workflow

The **Correction workflow** tab drives the whole doctor-correction lifecycle
(story #8) against the same API. It lists every report (`GET /reports`) newest
first — each row shows its `created_at`, short `report_id` and source
`dialogue_id`, and the report you just extracted is pre-selected. Pick one and
**Open correction** to open or resume its draft (`POST /reports/{id}/correction`);
no report id is typed by hand (a **Load by id** expander is kept as a fallback).
The selected report's **source dialogue** is then shown in speaking order
(`GET /dialogues/{dialogue_id}`), so the reply text each citation points at sits
on screen next to the correction.

The screen then shows the correction's **status** (`draft` / `verified`) with the
`verified_by` / `verified_at` stamp once verified, and for every note its
**origin** — *copied from original* (`source_note_id` present) vs *doctor-added*
(`null`) — its full S/O/A/P sections, the ICD on Assessment claims, and each
citation resolved to its **source-turn text** (fetched with `GET /dialogues/{id}`).

While the correction is a draft you can, wired to the API:

- **edit a note** — replace its section text, pick which turns each claim cites,
  and re-code the ICD (`PUT …/notes/{note_id}`);
- **add a doctor-authored note** (`POST …/notes`);
- **delete a note** (`DELETE …/notes/{note_id}`);
- **verify** it under a `doctor_id` (`POST …/verify`).

Once verified the UI reflects the locked state; an **Attempt an edit** button
demonstrates the `409 {code, detail}` rejection, and **Reopen for editing**
(`POST …/reopen`) returns it to a draft. All API errors are rendered as readable
`code: detail` messages.

For a verified correction the same screen also loads `GET
/dialogues/{dialogue_id}/quality` and shows all four online-quality aggregates:
added notes, removed notes, changed characters and diagnosis changes. A table
below them lists every matched source/corrected Note pair with its character
distance and diagnosis-change flag. Draft and reopened corrections show the
metrics as unavailable until the next verification, matching the API lifecycle.

This tab relies on the read endpoint **`GET /dialogues/{id}`** — it returns a
dialogue with its ordered turns (`{id, speaker, text}`), or `404` if the
dialogue does not exist — so the UI can turn each citation's opaque `turn_id`
back into the text the doctor actually said. Unlike the demo UI, that endpoint
is real API surface and is unit-tested.

## SOAP correction workflow (story #8)

The generated report is the model's output and is **immutable**: `GET
/reports/{id}` always returns that original, before and after any correction. A
doctor's edits live in a separate, single working version — the *correction* —
which starts as a `draft` and becomes `verified` once the doctor has checked it.
Every corrected note keeps a `source_note_id` back to the original note it was
copied from (a doctor-added note has `source_note_id: null`).

The correction is keyed by the report it edits — at most one per report — so all
endpoints hang off `/reports/{report_id}/correction`:

| Method + path | Purpose |
|---|---|
| `POST /reports/{report_id}/correction` | Open the draft, or return the existing one (idempotent). |
| `GET /reports/{report_id}/correction` | Return the current draft/verified version. |
| `PUT /reports/{report_id}/correction/notes/{note_id}` | Replace a note's S/O/A/P sections, citations and ICD. |
| `POST /reports/{report_id}/correction/notes` | Add a doctor-authored note. |
| `DELETE /reports/{report_id}/correction/notes/{note_id}` | Delete a note. |
| `POST /reports/{report_id}/correction/verify` | Verify the draft (`draft → verified`). |
| `POST /reports/{report_id}/correction/reopen` | Return a verified correction to editing (`verified → draft`). |

Editing is only allowed while the correction is a `draft`; a `verified`
correction rejects every change until it is reopened.

### Request examples

Open (or resume) the correction:

```bash
curl -X POST http://localhost:8000/reports/$REPORT_ID/correction
```

Replace a note's content. Sections mirror the response shape; each claim must
cite at least one turn of the **source dialogue** (an ungrounded citation is
rejected), and only Assessment claims carry an `icd`:

```bash
curl -X PUT http://localhost:8000/reports/$REPORT_ID/correction/notes/$NOTE_ID \
  -H 'Content-Type: application/json' \
  -d '{
        "assessment": [
          {
            "text": "Migraine without aura.",
            "citations": [{"turn_id": "'$TURN_ID'", "quote": "headache for three days"}],
            "icd": {"code": "G43.0", "name": "Migraine without aura",
                    "classifier_url": "https://icd/G43.0"}
          }
        ],
        "plan": [
          {"text": "Start sumatriptan.", "citations": [{"turn_id": "'$TURN_ID'"}]}
        ]
      }'
```

Add a note (same body shape, no `note_id`), delete a note, then verify and, if
needed, reopen:

```bash
curl -X POST   http://localhost:8000/reports/$REPORT_ID/correction/notes -H 'Content-Type: application/json' -d '{ ... }'
curl -X DELETE http://localhost:8000/reports/$REPORT_ID/correction/notes/$NOTE_ID
curl -X POST   http://localhost:8000/reports/$REPORT_ID/correction/verify -H 'Content-Type: application/json' -d '{"doctor_id": "dr-house"}'
curl -X POST   http://localhost:8000/reports/$REPORT_ID/correction/reopen
```

> **`doctor_id` is not authentication.** It is only an attribution label stamped
> onto the verified version (recorded as `verified_by`) so the corrected note
> shows who signed off. It grants no access, is not verified against any
> identity provider, and must not be treated as a security or authorization
> mechanism. Real authn/authz is out of scope for this baseline.

### Errors

Correction errors return a stable body `{"code": "...", "detail": "..."}` with a
machine-readable `code`:

| Status | `code` | Cause |
|---|---|---|
| 404 | `report_not_found` | The report to correct does not exist. |
| 404 | `correction_not_found` | The report has no correction yet. |
| 404 | `note_not_found` | The note id does not belong to the correction. |
| 409 | `correction_not_editable` | Editing a `verified` correction (reopen it first). |
| 422 | `citation_not_in_source_dialogue` | A claim cites a turn absent from the source dialogue. |
| 422 | `empty_doctor_id` | `verify` was called without a non-blank `doctor_id`. |
| 422 | `duplicate_source_note` | Two corrected notes claim the same source note. |

## Online SOAP quality (story #10)

`GET /dialogues/{dialogue_id}/quality` calculates extraction quality against the
dialogue's current **verified** doctor correction. Metrics are calculated on
every request and are not persisted: the immutable generated `SoapReport` and
the latest saved correction remain the sources of truth.

| Response field | Meaning |
|---|---|
| `dialogue_id` | Dialogue whose extracted report is being evaluated. |
| `report_id` | Immutable generated report used as the comparison source. |
| `correction_id` | Doctor correction used as the verified target. |
| `notes_added` | Doctor-authored notes (`source_note_id: null`). |
| `notes_removed` | Original notes with no corrected note carrying their id. |
| `changed_characters` | Sum of character edit distances across matched notes only. |
| `diagnosis_changes` | Matched notes whose Assessment text or ICD tuple changed. |
| `note_diffs` | Per-matched-note ids, character distance and diagnosis-change flag. |

Corrected and original notes are matched only by `source_note_id`; text
similarity is never used for lineage. For each matched note, claim text is
concatenated in stable Subjective → Objective → Assessment → Plan order and a
unit-cost Levenshtein distance is calculated (insertion, deletion and
substitution each cost 1). Only newline forms are normalized. IDs, citations,
confidence and ICD metadata do not contribute to `changed_characters`.
Added/removed notes are counted separately and do not inflate character or
diagnosis metrics. An ICD-only change still increments `diagnosis_changes`.

The endpoint is intentionally unavailable while the correction is a draft. A
reopen immediately hides previously available metrics; the next verification
exposes a fresh calculation over the updated content.

```bash
curl http://localhost:8000/dialogues/$DIALOGUE_ID/quality
```

Quality errors use the same stable `{"code": "...", "detail": "..."}` body:

| Status | `code` | Cause |
|---|---|---|
| 404 | `report_not_found` | No extracted report exists for the dialogue. |
| 404 | `correction_not_found` | The report has no doctor correction. |
| 409 | `REPORT_NOT_VERIFIED` | The correction is still a draft or was reopened. |

## Model serving (vLLM + GPU)

The clinical reasoning stages (SOAP extraction, NLI groundedness scoring) call
**MedGemma 4B** (`google/medgemma-4b-it`) through an OpenAI-compatible API served
by [vLLM](https://docs.vllm.ai/). It runs as the `vllm` compose service, guarded
by the `gpu` profile so the default stack (`app` + `postgres`) stays CPU-only.

**Requirements:**

- An NVIDIA GPU with compute capability **>= 8.0** (Ampere or newer, e.g. L4 /
  A100 / RTX 30-series). gemma3 runs in `bfloat16`; float16 is numerically
  unstable for it and older cards (T4, P100) cannot run bf16.
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
  so Docker can pass the GPU into the container.
- ~8 GB of free VRAM for the weights plus headroom for the KV cache.
- MedGemma is **gated**: accept the licence at
  <https://huggingface.co/google/medgemma-4b-it>, then set `HF_TOKEN` in `.env`
  (see `.env.example`).

**Bring the model up:**

```bash
docker compose up vllm            # downloads the checkpoint on first run
curl http://localhost:8001/v1/models   # host port 8001 -> container 8000
```

The launch arguments live in `src/infra/vllm` (`VllmDeployment` /
`MEDGEMMA_4B`) as the reusable source of truth; the compose command mirrors
them. Within the compose network the app reaches the server at
`http://vllm:8000/v1` (`VLLM_BASE_URL`).

## Manual acceptance (SOAP extractor smoke test)

`scripts/smoke_extractor.py` is a QA-runs-it-by-hand acceptance check — **not**
a pytest test and not part of `make check` or CI. It pushes a few varied
dialogues (pneumonia, hypertension, back pain) through the real extractor on a
live vLLM server, validates each `SoapReport` against the SOAP invariants
(S/O/A/P structure, at least one note per dialogue, every claim citing a real
dialogue turn), and prints a per-dialogue pass/fail report. It exits non-zero if
any dialogue fails.

With the model server up (see above), run it against the host endpoint:

```bash
VLLM_BASE_URL=http://localhost:8001/v1 MODEL_ID=google/medgemma-4b-it \
    PYTHONPATH=src uv run python scripts/smoke_extractor.py
```

Required environment variables:

- `VLLM_BASE_URL` — OpenAI-compatible base URL of the vLLM server (e.g.
  `http://localhost:8001/v1`).
- `MODEL_ID` — served model id (e.g. `google/medgemma-4b-it`).
- `PYTHONPATH=src` — so the `src`-layout packages import.

## Manual acceptance (SOAP edit agent smoke test)

`scripts/smoke_editor.py` is the QA-runs-it-by-hand acceptance check for the LLM
SOAP edit agent (story #12) — **not** a pytest test and not part of `make check`
or CI (the unit suite in `tests/test_llm_editor.py` mocks the LLM). It builds a
doctor correction from a canned dialogue, asks the real agent on a live vLLM
server to satisfy a plain edit request (e.g. "add a follow-up plan note"), and
checks the returned `ProposalDraft` against the agent's contract: every operation
is add / update / delete, each update / delete targets a note that exists in the
correction, every proposed claim cites a real dialogue turn, and no ICD coding
can appear (the output schema has no channel for it). It prints a per-fixture
pass/fail report and exits non-zero if any fixture fails.

With the model server up (see above), run it against the host endpoint:

```bash
VLLM_BASE_URL=http://localhost:8001/v1 MODEL_ID=google/medgemma-4b-it \
    PYTHONPATH=src uv run python scripts/smoke_editor.py
```

The required environment variables are the same as for the extractor smoke test
above (`VLLM_BASE_URL`, `MODEL_ID`, `PYTHONPATH=src`).

## Demo / E2E

`scripts/e2e_smoke.py` is the end-to-end happy-path scenario `[#7/NFR-4]`: it
drives the whole stack over HTTP — `POST /dialogues`, `POST
/dialogues/{id}/report?patient_id=P001`, `GET /reports/{id}` — and asserts the
persisted `SoapReport` is a well-formed clinical document: at least one note,
all four S/O/A/P sections populated, a complete ICD coding on the Assessment, a
per-note groundedness `confidence` in `[0, 1]`, and every claim traced to a real
dialogue turn. It prints a readable pass/fail report and exits non-zero on
failure. Like the smoke tests, it is run by hand — not pytest, not CI.

### Full stack, one command (GPU machine)

Brings up everything: `postgres` (healthcheck) → `app` (auto-applies Alembic
migrations on startup, then serves) plus `vllm` serving MedGemma (the `app`
waits for it to pass its healthcheck before serving). Needs an NVIDIA GPU, the
NVIDIA Container Toolkit, and `HF_TOKEN` in `.env` (see "Model serving" above):

```bash
docker compose --profile gpu up -d   # wait until `app` and `vllm` are healthy
BASE_URL=http://localhost:8000 PYTHONPATH=src uv run python scripts/e2e_smoke.py
```

### Pragmatic variant: a MedGemma is already running

When a vLLM MedGemma is already serving on the host (single GPU already busy),
skip the compose `vllm` and run the `app` on the host against a Postgres and
that existing endpoint. Bring up just the database, then the app:

```bash
docker compose up -d postgres        # or any scratch Postgres you can reach

DATABASE_URL=postgresql+asyncpg://medcopilot:medcopilot@localhost:5432/medcopilot \
VLLM_BASE_URL=http://localhost:8001/v1 MODEL_ID=google/medgemma-4b-it \
NLI_CONFIDENCE_ENABLED=1 PYTHONPATH=src \
    uv run uvicorn app.main:app --port 8000     # migrations auto-apply on startup

BASE_URL=http://localhost:8000 PYTHONPATH=src uv run python scripts/e2e_smoke.py
```

`NLI_CONFIDENCE_ENABLED=1` turns on the NLI groundedness scorer so each note's
`confidence` is populated (off by default to avoid a tokenizer download). The
compose `postgres` service publishes no host port; point `DATABASE_URL` at
whatever database the host app can actually reach.

## Load test (E2E latency, NFR-1)

`scripts/load_test.py` measures the latency of the full extraction pipeline to
confirm **NFR-1 (P99 ≤ 5s)** `[#7/NFR-1]`. For each measured request it creates a
fresh dialogue (`POST /dialogues`) and then times the expensive extraction +
confidence-scoring step (`POST /dialogues/{id}/report`) — a fresh dialogue every
time so idempotency doesn't short-circuit the LLM call. It runs a warmup burst,
then N measured requests at a chosen concurrency, and prints P50/P95/P99 (plus
count/mean/max), stating PASS/FAIL against the 5s P99 target. It is a
dependency-light `asyncio` + `httpx` script — **not** pytest and not part of
`make check` or CI. It exits non-zero if P99 exceeds the threshold or any request
fails.

**Authoritative measurement — run it against the full compose stack** (the app,
Postgres and the GPU-served MedGemma), not a partial one:

```bash
docker compose --profile gpu up          # app + postgres + vllm (needs the GPU)
uv run python scripts/load_test.py --requests 100 --concurrency 8
```

Knobs (CLI flag, or the env var used as its default):

- `--base-url` / `LOAD_BASE_URL` — API base URL (default `http://localhost:8000`).
- `-n` / `--requests` / `LOAD_REQUESTS` — measured requests (default `30`).
- `-c` / `--concurrency` / `LOAD_CONCURRENCY` — in-flight requests (default `4`).
- `-w` / `--warmup` / `LOAD_WARMUP` — unmeasured warmup requests (default `2`).
- `--threshold` / `LOAD_P99_THRESHOLD` — P99 target in seconds (default `5.0`).
- `--timeout` / `LOAD_TIMEOUT` — per-request timeout in seconds (default `120`).
- `--patient-id` / `LOAD_PATIENT_ID` — optional EHR patient id for context.

Reading the output: `p99` is the latency the 5s target is judged on; `p50` and
`p95` show the typical and tail-approaching cost, `max` the worst single request.
The final line prints the P99 verdict — a green `PASS` means the tail latency met
NFR-1 under that load. The authoritative P99 number for NFR-1 sign-off comes from
this full-stack run; a smoke run against a stub only proves the harness works.
