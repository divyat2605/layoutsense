.PHONY: help build up down logs test lint clean

## ── Help ─────────────────────────────────────────────────────────────────────
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## ── Docker ───────────────────────────────────────────────────────────────────
build:  ## Build Docker images
	docker compose build

up:  ## Start all services
	docker compose up

up-d:  ## Start all services (detached)
	docker compose up -d

down:  ## Stop all services
	docker compose down

logs:  ## Follow API logs
	docker compose logs -f api

## ── Development ──────────────────────────────────────────────────────────────
dev:  ## Start API in development mode (with hot reload)
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

## ── Testing ──────────────────────────────────────────────────────────────────
test:  ## Run unit tests (no model weights required)
	pytest -v

test-all:  ## Run all tests including integration (requires PaddleOCR weights)
	pytest -v -m ""

test-cov:  ## Run tests with coverage report
	pytest --cov=app --cov-report=term-missing --cov-report=html

## ── Code quality ─────────────────────────────────────────────────────────────
lint:  ## Run ruff linter
	ruff check app/ tests/

format:  ## Format code with ruff
	ruff format app/ tests/

typecheck:  ## Run mypy type checker (strict mode)
	mypy app/ --strict --ignore-missing-imports

## ── Cleanup ───────────────────────────────────────────────────────────────────
clean:  ## Remove Python caches and coverage artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/

clean-docker:  ## Remove Docker containers and volumes (DESTRUCTIVE)
	docker compose down -v --remove-orphans
	docker image rm docuparse:latest 2>/dev/null; true
