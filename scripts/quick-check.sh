#!/bin/bash
# Quick local checks script
# Faster version for rapid development

set -e  # Exit on error

echo "============================================"
echo "Running Quick Checks (Fast)"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 passed${NC}"
    else
        echo -e "${RED}✗ $1 failed${NC}"
        exit 1
    fi
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must be run from project root directory${NC}"
    exit 1
fi

echo "1. Running black formatting check..."
echo "----------------------------------------"
black --check clearview tests
print_status "Black formatting"
echo ""

echo "2. Running ruff linting..."
echo "----------------------------------------"
ruff check clearview tests
print_status "Ruff linting"
echo ""

echo "3. Running unit tests only (no integration)..."
echo "----------------------------------------"
pytest tests/test_models.py tests/test_losses.py tests/test_utils.py -v --tb=short
print_status "Unit tests"
echo ""

echo "============================================"
echo -e "${GREEN}Quick checks passed! ✓${NC}"
echo "============================================"
echo ""
echo "For full CI checks, run: ./scripts/run-ci-checks.sh"
