#!/bin/bash
set -e

echo "========================================"
echo "Running ruff linter..."
echo "========================================"
echo ""

uv run ruff check src/ tests/

echo ""
echo "========================================"
echo "Running mypy type checker..."
echo "========================================"
echo ""

uv run mypy src/ --ignore-missing-imports

echo ""
echo "========================================"
echo "Linting passed!"
echo "========================================"
