#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "========================================"
echo "Running all code quality checks"
echo "========================================"
echo ""

echo "Step 1/3: Formatting code..."
"$SCRIPT_DIR/format.sh"

echo ""
echo "Step 2/3: Linting code..."
"$SCRIPT_DIR/lint.sh"

echo ""
echo "Step 3/3: Running tests..."
"$SCRIPT_DIR/test.sh"

echo ""
echo "========================================"
echo "All checks passed!"
echo "========================================"
