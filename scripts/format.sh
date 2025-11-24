#!/bin/bash
set -e

echo "========================================"
echo "Formatting code with black..."
echo "========================================"
echo ""

uv run black src/ tests/

echo ""
echo "========================================"
echo "Sorting imports with ruff..."
echo "========================================"
echo ""

uv run ruff check --select I --fix src/ tests/

echo ""
echo "========================================"
echo "Code formatted successfully!"
echo "========================================"
