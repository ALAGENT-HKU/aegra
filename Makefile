<<<<<<< HEAD
.PHONY: help install dev-install setup-hooks format lint type-check security test test-cov clean run ci-check
=======
.PHONY: help install dev-install setup-hooks format lint type-check security test test-api test-cli test-cov clean run ci-check openapi
>>>>>>> origin/dev_ALAGENT-HKU-merged

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
<<<<<<< HEAD
	@echo "  make dev-install   - Install dev dependencies + git hooks"
	@echo "  make setup-hooks   - Reinstall git hooks (if needed)"
	@echo "  make format        - Format code with ruff"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make type-check    - Run mypy type checking"
	@echo "  make security      - Run security checks with bandit"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage"
=======
	@echo "  make dev-install   - Install all dependencies + git hooks"
	@echo "  make setup-hooks   - Reinstall git hooks (if needed)"
	@echo "  make format        - Format code with ruff"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make type-check    - Run ty type checking"
	@echo "  make security      - Run security checks with bandit"
	@echo "  make test          - Run all tests"
	@echo "  make test-api      - Run aegra-api tests only"
	@echo "  make test-cli      - Run aegra-cli tests only"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make openapi       - Regenerate docs/openapi.json from code"
>>>>>>> origin/dev_ALAGENT-HKU-merged
	@echo "  make ci-check      - Run all CI checks locally"
	@echo "  make clean         - Clean cache files"
	@echo "  make run           - Run the server"

install:
<<<<<<< HEAD
	uv sync --no-dev

dev-install:
	uv sync
	@uv run pre-commit install
	@uv run pre-commit install --hook-type commit-msg
	@echo ""
	@echo "✅ Dependencies installed!"
	@echo "✅ Git hooks installed!"
	@echo "🚀 You're ready to develop!"
=======
	uv sync --all-packages --no-dev

dev-install:
	uv sync --all-packages
	@uv run pre-commit install
	@uv run pre-commit install --hook-type commit-msg
	@echo ""
	@echo "Done! Dependencies installed and git hooks set up."
>>>>>>> origin/dev_ALAGENT-HKU-merged

setup-hooks:
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg
	@echo ""
<<<<<<< HEAD
	@echo "✅ Git hooks reinstalled!"
	@echo "📝 Your commits will now be checked automatically"
=======
	@echo "Git hooks reinstalled!"
>>>>>>> origin/dev_ALAGENT-HKU-merged

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .

type-check:
<<<<<<< HEAD
	uv run mypy src/

security:
	uv run bandit -c pyproject.toml -r src/

test:
	uv run pytest

test-cov:
	uv run pytest --cov=src --cov-report=html --cov-report=term

ci-check: format lint type-check security test
	@echo ""
	@echo "✅ All CI checks passed!"
=======
	uv run ty check libs/aegra-api/src/ libs/aegra-cli/src/

security:
	uv run bandit -c pyproject.toml -r libs/aegra-api/src/ libs/aegra-cli/src/

test: test-api test-cli

test-api:
	uv run --package aegra-api pytest libs/aegra-api/tests/

test-cli:
	uv run --package aegra-cli pytest libs/aegra-cli/tests/

test-cov:
	uv run --package aegra-api pytest libs/aegra-api/tests/ --cov=libs/aegra-api/src --cov-report=html --cov-report=term
	uv run --package aegra-cli pytest libs/aegra-cli/tests/ --cov=libs/aegra-cli/src --cov-report=term

openapi:
	uv run --package aegra-api python scripts/export_openapi.py

ci-check: format lint
	-uv run ty check libs/aegra-api/src/ libs/aegra-cli/src/
	-uv run bandit -c pyproject.toml -r libs/aegra-api/src/ libs/aegra-cli/src/
	$(MAKE) test
	@echo ""
	@echo "All CI checks completed! (ty and bandit are non-blocking)"
>>>>>>> origin/dev_ALAGENT-HKU-merged

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
<<<<<<< HEAD
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov 2>/dev/null || true

run:
	uv run python run_server.py
=======
	rm -rf .pytest_cache .ty_cache .ruff_cache htmlcov 2>/dev/null || true

run:
	uv run --package aegra-api uvicorn aegra_api.main:app --reload
>>>>>>> origin/dev_ALAGENT-HKU-merged
