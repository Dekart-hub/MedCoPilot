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
