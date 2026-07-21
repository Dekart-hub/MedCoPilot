.DEFAULT_GOAL := check

.PHONY: install run lint format typecheck test check mock-ehr-up mock-ehr-seed mock-ehr-down mock-ehr-live-test

install: ## Sync the uv-managed virtual environment
	uv sync

run: ## Run the FastAPI app with uvicorn
	PYTHONPATH=src uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

lint: ## Run ruff lint checks
	uv run ruff check src tests

format: ## Auto-format the code with ruff
	uv run ruff format src tests

typecheck: ## Run mypy static type checks
	uv run mypy

test: ## Run the test suite (an empty suite is treated as success)
	uv run pytest -q || [ $$? -eq 5 ]

check: lint typecheck test ## Run lint, typecheck and tests

mock-ehr-up: ## Start the pinned development HAPI FHIR R4 service
	docker compose -f compose.mock-ehr.yml up -d

mock-ehr-seed: ## Idempotently seed publication Patient/Encounter/Practitioner
	curl --fail --silent --show-error --retry 30 --retry-all-errors --retry-delay 2 -H 'Content-Type: application/fhir+json' --data-binary @mock_ehr/fixtures/publication-context.json http://localhost:$${MOCK_EHR_PORT:-8080}/fhir

mock-ehr-down: ## Stop and remove the development HAPI service
	docker compose -f compose.mock-ehr.yml down

mock-ehr-live-test: ## Validate and publish a document twice against live HAPI
	MOCK_EHR_BASE_URL=http://localhost:$${MOCK_EHR_PORT:-8080}/fhir uv run pytest -q tests/test_fhir_publication_live.py
