.PHONY: help sync format lint test cli

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Sync dependencies from lockfile
	uv sync

format: ## Format code with ruff
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Run linters
	uv run ruff format --check .
	uv run ruff check .

test: ## Run tests for a specific asset (usage: make test NAME=aggregate [TYPE=functions])
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter is required"; \
		echo "Usage: make test NAME=<asset_name> [TYPE=functions|modules|steps]"; \
		echo "Example: make test NAME=aggregate"; \
		echo "Example: make test NAME=mymodule TYPE=modules"; \
		exit 1; \
	fi
	@TYPE=$${TYPE:-functions}; \
	echo "Running tests for $$TYPE/src/$(NAME)"; \
	uv run python -m cli.cli run-tests -r $$TYPE/src/$(NAME) -s py -fn $(NAME)

cli: ## Run the CLI tool (usage: make cli ARGS="command args")
	uv run python -m cli.cli $(ARGS)

.DEFAULT_GOAL := help

