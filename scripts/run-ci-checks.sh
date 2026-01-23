#!/bin/bash
# Local CI checks script
# Run this before pushing to ensure CI will pass

set -e  # Exit on error

echo "============================================"
echo "Running Local CI Checks"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

echo "1. Running pre-commit checks..."
echo "----------------------------------------"
pre-commit run --all-files
print_status "Pre-commit checks"
echo ""

echo "2. Running ruff linting..."
echo "----------------------------------------"
ruff check clearview tests
print_status "Ruff linting"
echo ""

echo "3. Running ruff format check..."
echo "----------------------------------------"
ruff format --check clearview tests
print_status "Ruff formatting"
echo ""

echo "4. Running type checking (mypy)..."
echo "----------------------------------------"
mypy clearview --config-file pyproject.toml || echo -e "${YELLOW}⚠ Type checking has warnings (non-blocking)${NC}"
echo ""

echo "5. Running test suite..."
echo "----------------------------------------"
pytest tests/ -v --cov=clearview --cov-report=term-missing
print_status "Test suite"
echo ""

echo "6. Building package..."
echo "----------------------------------------"
python -m build
print_status "Package build"
echo ""

echo "7. Checking package integrity..."
echo "----------------------------------------"
twine check dist/*
print_status "Package check"
echo ""

echo "============================================"
echo -e "${GREEN}All CI checks passed! ✓${NC}"
echo "============================================"
echo ""
echo "You can now push your changes with confidence!"
