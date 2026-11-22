.PHONY: help setup-venv install install-dev test test-cov lint format format-check \
	type-check pre-commit-install pre-commit-run clean build

PYTHON ?= python3
SRC := clearview tests

help:
	@echo "ClearView - available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup-venv          Create a virtual environment in ./.venv"
	@echo "  make install             Install the package (editable)"
	@echo "  make install-dev         Install with dev, metrics and hub extras"
	@echo ""
	@echo "Development:"
	@echo "  make test                Run the test suite"
	@echo "  make test-cov            Run the test suite with coverage"
	@echo "  make lint                Lint with ruff"
	@echo "  make format              Format with ruff (and fix lint issues)"
	@echo "  make format-check        Check formatting without changing files"
	@echo "  make type-check          Type-check with mypy"
	@echo ""
	@echo "Pre-commit:"
	@echo "  make pre-commit-install  Install the git hooks"
	@echo "  make pre-commit-run      Run all hooks on all files"
	@echo ""
	@echo "Other:"
	@echo "  make build               Build sdist and wheel into ./dist"
	@echo "  make clean               Remove build artifacts and caches"

setup-venv:
	$(PYTHON) -m venv .venv
	@echo "Created .venv. Activate it with: source .venv/bin/activate"

install:
	pip install --upgrade pip
	pip install -e .

install-dev:
	pip install --upgrade pip
	pip install -e ".[dev,metrics,hub]"

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=clearview --cov-report=term-missing

lint:
	ruff check $(SRC)

format:
	ruff check --fix $(SRC)
	ruff format $(SRC)

format-check:
	ruff format --check $(SRC)

type-check:
	mypy clearview --config-file pyproject.toml

pre-commit-install:
	pip install pre-commit
	pre-commit install
	pre-commit install --hook-type commit-msg

pre-commit-run:
	pre-commit run --all-files

build:
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

clean:
	rm -rf build/ dist/ *.egg-info .eggs/ htmlcov/ .coverage
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
