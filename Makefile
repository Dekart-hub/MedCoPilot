.DEFAULT_GOAL := help

ENV_FILE ?= .env
APP := app.main:app
HOST ?= 0.0.0.0
PORT ?= 8000
export PYTHONPATH := src

.PHONY: help install run dev test ui docker-build docker-run mock-ehr-up mock-ehr-seed mock-ehr-post-visit mock-ehr-live-test mock-ehr-down

help: ## Показать список целей
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Установить зависимости
	uv sync

run: ## Запустить приложение с .env
	uv run uvicorn $(APP) --env-file $(ENV_FILE) --host $(HOST) --port $(PORT)

dev: ## Запустить с автоперезагрузкой
	uv run uvicorn $(APP) --env-file $(ENV_FILE) --host $(HOST) --port $(PORT) --reload

test: ## Прогнать тесты
	uv run pytest -q

ui: ## Запустить Streamlit-демо (бэкенд поднимать отдельно через make dev)
	uv run --project ui streamlit run ui/app.py

docker-build: ## Build the Docker image
	docker build -t medcopilot:latest .

docker-run: ## Run the container with env from .env
	docker run --rm --env-file $(ENV_FILE) -p $(PORT):8000 medcopilot:latest

mock-ehr-up: ## Start the external HAPI FHIR mock service
	docker compose -f compose.mock-ehr.yml up -d

mock-ehr-seed: ## Load safe pre-visit FHIR fixtures
	uv run python scripts/seed_mock_ehr.py --phase pre-visit

mock-ehr-post-visit: ## Load the separate gold/post-visit fixture boundary
	uv run python scripts/seed_mock_ehr.py --phase post-visit

mock-ehr-live-test: ## Run the opt-in live HAPI read/write smoke test
	MOCK_EHR_BASE_URL=http://localhost:8080/fhir uv run pytest -q tests/test_fhir_live.py

mock-ehr-down: ## Stop and remove the external mock EHR
	docker compose -f compose.mock-ehr.yml down -v
