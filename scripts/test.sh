#!/bin/bash
set -e

echo "========================================"
echo "Running tests with coverage..."
echo "========================================"
echo ""

uv run pytest tests/ \
    -v \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html

echo ""
echo "========================================"
echo "Tests passed!"
echo "Coverage report generated in htmlcov/"
echo "========================================"
