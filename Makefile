.PHONY: help sync format lint test cli

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Sync dependencies from lockfile
	uv sync

format: ## Format code with black and isort
	uv run black .
	uv run isort .

lint: ## Run linters
	uv run black --check .
	uv run isort --check-only .

test: ## Run tests for a specific function
	@echo "Usage: make test FUNC=aggregate"
	@if [ -z "$(FUNC)" ]; then \
		echo "Error: FUNC parameter is required"; \
		exit 1; \
	fi
	uv run python -m cli.cli run-tests -r functions/src/$(FUNC) -s py -fn $(FUNC)

cli: ## Run the CLI tool (usage: make cli ARGS="command args")
	uv run python -m cli.cli $(ARGS)

.DEFAULT_GOAL := help

