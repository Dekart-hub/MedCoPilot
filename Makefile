.DEFAULT_GOAL := help

ENV_FILE ?= .env
APP := app.main:app
HOST ?= 0.0.0.0
PORT ?= 8000
export PYTHONPATH := src

.PHONY: help install run dev test ui docker-build docker-run

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
