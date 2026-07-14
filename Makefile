.DEFAULT_GOAL := check

.PHONY: install run lint format typecheck test check

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
