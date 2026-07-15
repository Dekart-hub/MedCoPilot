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
baseline by hand: paste a dialogue, optionally pin a `patient_id`, and see the
extracted `SoapReport` — the four S/O/A/P sections, the ICD code on Assessment
claims, per-note confidence, and each claim linked back to the dialogue turn it
cites. It is a **demo only** (out of scope for the baseline DoD): a thin HTTP
client over the REST API, kept in its own optional `demo` dependency group so
the core install and the production image stay lean.

**The baseline API must be running first** (see `make run`, and bring up the
database and model server per the sections below):

```bash
uv run --group demo streamlit run ui/app.py
```

The `--group demo` flag installs Streamlit on first use. Point the app at a
non-default API with the sidebar field or the `MEDCOPILOT_API_URL` environment
variable (default `http://localhost:8000`).

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
