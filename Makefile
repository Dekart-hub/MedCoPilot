.DEFAULT_GOAL := check

.PHONY: install lint format typecheck test check

install: ## Sync the uv-managed virtual environment
	uv sync

lint: ## Run ruff lint checks
	uv run ruff check src tests

format: ## Auto-format the code with ruff
	uv run ruff format src tests

typecheck: ## Run mypy static type checks
	uv run mypy

test: ## Run the test suite (an empty suite is treated as success)
	uv run pytest -q || [ $$? -eq 5 ]

check: lint typecheck test ## Run lint, typecheck and tests
