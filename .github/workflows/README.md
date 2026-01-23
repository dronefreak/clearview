# GitHub Actions Workflows

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Workflows

### 1. CI Workflow (`ci.yml`)

**Trigger:** Push to `main`/`develop` branches, Pull Requests

**Jobs:**

#### Pre-commit Checks
- Runs all pre-commit hooks on all files
- Validates code formatting, linting, type hints
- Uses Python 3.11

#### Test Suite
- Runs on Python 3.8, 3.9, 3.10, and 3.11
- Installs PyTorch CPU-only version (no CUDA)
- Runs full test suite with coverage
- Uploads coverage report to Codecov (Python 3.11 only)

**Test Command:**
```bash
pytest tests/ -v --cov=clearview --cov-report=xml --cov-report=term-missing
```

#### Type Checking
- Runs mypy type checker
- Uses strict type checking configuration
- Python 3.11

#### Linting
- Runs ruff for code linting
- Checks code formatting with ruff format
- Python 3.11

#### Build Package
- Builds source distribution and wheel
- Validates package with twine
- Uploads build artifacts

### 2. Quick Check Workflow (`quick-check.yml`)

**Trigger:** Push and Pull Requests

**Purpose:** Fast feedback for developers

**Jobs:**

#### Quick Tests
- Runs only unit tests (excludes integration tests)
- Uses Python 3.11 only
- Minimal dependency installation
- Faster than full CI (~5 minutes vs ~15 minutes)

#### Lint Check
- Quick formatting check with black
- Quick linting with ruff

## PyTorch CPU Installation

All workflows use CPU-only PyTorch to:
- Reduce installation time
- Reduce storage requirements
- Keep CI costs low
- Tests don't require GPU

**Installation command:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Caching

The workflows use GitHub Actions caching for:
- pip packages
- Keyed by: OS, Python version, requirements hash

This speeds up subsequent runs by ~2-3 minutes.

## Adding CI Badge

Add this to your `README.md`:

```markdown
[![CI](https://github.com/dronefreak/clearview/workflows/CI/badge.svg)](https://github.com/dronefreak/clearview/actions/workflows/ci.yml)
```

## Local Testing

To run the same checks locally:

### Pre-commit
```bash
pre-commit run --all-files
```

### Tests
```bash
pytest tests/ -v --cov=clearview --cov-report=term-missing
```

### Type Checking
```bash
mypy clearview --config-file pyproject.toml
```

### Linting
```bash
ruff check clearview tests
ruff format --check clearview tests
```

### Build Package
```bash
python -m build
twine check dist/*
```

## Environment Variables

No special environment variables required. All workflows run with default GitHub Actions environment.

## Troubleshooting

### Tests Failing Locally But Passing in CI

- Ensure you're using the correct Python version
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Install package in editable mode: `pip install -e .`

### Pre-commit Hooks Failing

- Update hooks: `pre-commit autoupdate`
- Clear cache: `pre-commit clean`
- Run on all files: `pre-commit run --all-files`

### PyTorch Installation Issues

The workflows use CPU-only PyTorch. If you need GPU support locally:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Codecov Integration

Coverage reports are uploaded to Codecov from the Python 3.11 test run.

To set up Codecov:
1. Go to https://codecov.io
2. Connect your GitHub repository
3. Add the Codecov token to GitHub Secrets (if private repo)
4. Badge will appear after first successful upload

## Performance

| Workflow | Average Duration | When it Runs |
|----------|-----------------|--------------|
| Quick Check | ~5 minutes | Every push/PR |
| Full CI | ~15 minutes | Every push/PR |
| Pre-commit only | ~2 minutes | Part of Full CI |

## Maintenance

### Updating Python Versions

Edit the `matrix.python-version` in `ci.yml`:
```yaml
matrix:
  python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]
```

### Updating Dependencies

Dependencies are automatically installed from:
- `requirements.txt` - Core dependencies
- `pyproject.toml` - Package metadata and optional dependencies

### Updating Actions

GitHub Actions versions are pinned with major version (e.g., `@v4`).
To update:
```bash
# Check for updates
gh extension install actions/gh-actions-cache
```

## Contributing

When adding new code:
1. **Quick Check** will run first for fast feedback
2. **Full CI** will run all tests and checks
3. Ensure all checks pass before merging

The workflows are designed to catch issues early and provide fast feedback.
