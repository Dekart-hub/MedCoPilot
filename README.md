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
