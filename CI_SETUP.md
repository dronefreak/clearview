# CI/CD Setup - Quick Reference

## ✅ What's Included

### GitHub Actions Workflows

1. **Full CI Pipeline** (`.github/workflows/ci.yml`)
   - ✓ Pre-commit checks on all files
   - ✓ Tests on Python 3.8, 3.9, 3.10, 3.11
   - ✓ PyTorch CPU-only (no GPU downloads)
   - ✓ Type checking with mypy
   - ✓ Linting with ruff
   - ✓ Package build validation
   - ✓ Coverage upload to Codecov

2. **Quick Check Pipeline** (`.github/workflows/quick-check.yml`)
   - ✓ Fast unit tests (Python 3.11)
   - ✓ Quick lint checks
   - ✓ Runs on every push/PR

### Local Development Scripts

1. **Full CI Checks** (`scripts/run-ci-checks.sh`)
   ```bash
   ./scripts/run-ci-checks.sh
   ```
   Runs the same checks as GitHub Actions locally (~10 min)

2. **Quick Checks** (`scripts/quick-check.sh`)
   ```bash
   ./scripts/quick-check.sh
   ```
   Fast feedback during development (~2 min)

## 🚀 Quick Start

### First Time Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pre-commit pytest pytest-cov mypy ruff black build twine

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "import clearview; print(clearview.__version__)"
```

### Before Committing

```bash
# Quick check (recommended during development)
./scripts/quick-check.sh

# Full check (before pushing)
./scripts/run-ci-checks.sh
```

### Running Individual Checks

```bash
# Pre-commit on all files
pre-commit run --all-files

# Tests with coverage
pytest tests/ -v --cov=clearview --cov-report=term-missing

# Type checking
mypy clearview --config-file pyproject.toml

# Linting
ruff check clearview tests
ruff format --check clearview tests

# Build package
python -m build
twine check dist/*
```

## 📊 CI Pipeline Flow

```
Push/PR to main/develop
    ↓
┌───────────────────────────────────────┐
│  Quick Check (Fast Feedback)         │
│  - Unit tests (Python 3.11)          │
│  - Lint checks                        │
│  Duration: ~5 minutes                 │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  Full CI (Comprehensive)              │
│                                       │
│  Pre-commit ──────────────┐          │
│  Test (3.8, 3.9, 3.10, 3.11)─┐       │
│  Type Check ────────────────┼───→ ✓  │
│  Lint ──────────────────────┤       │
│  Build ─────────────────────┘       │
│                                       │
│  Duration: ~15 minutes                │
└───────────────────────────────────────┘
    ↓
  Merge ✓
```

## 🔧 PyTorch CPU Installation

The CI uses CPU-only PyTorch to:
- ✓ Speed up installation (no large CUDA files)
- ✓ Reduce CI costs
- ✓ Keep runners lightweight

**Installation command:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**For local GPU development:**
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 📈 Adding CI Badge to README

Add this to your `README.md`:

```markdown
[![CI](https://github.com/dronefreak/clearview/workflows/CI/badge.svg)](https://github.com/dronefreak/clearview/actions/workflows/ci.yml)
[![Quick Check](https://github.com/dronefreak/clearview/workflows/Quick%20Check/badge.svg)](https://github.com/dronefreak/clearview/actions/workflows/quick-check.yml)
```

## 🎯 Workflow Triggers

| Workflow | Trigger | When |
|----------|---------|------|
| Full CI | Push to main/develop | Every push |
| Full CI | Pull Request | Every PR |
| Quick Check | Push to any branch | Every push |
| Quick Check | Pull Request | Every PR |

## 📋 What Each Job Does

### Pre-commit Job
- Runs all hooks from `.pre-commit-config.yaml`
- Checks: formatting, linting, type hints, docstrings
- Duration: ~2 minutes

### Test Job (Matrix)
- Runs on Python 3.8, 3.9, 3.10, 3.11
- Installs PyTorch CPU
- Runs full test suite with coverage
- Uploads coverage (Python 3.11 only)
- Duration: ~5 minutes per Python version

### Type Check Job
- Runs mypy with strict configuration
- Validates type hints throughout codebase
- Duration: ~2 minutes

### Lint Job
- Runs ruff for code quality
- Checks formatting with ruff format
- Duration: ~1 minute

### Build Job
- Builds wheel and sdist
- Validates with twine
- Uploads artifacts
- Duration: ~2 minutes

## 🐛 Troubleshooting

### Local tests pass but CI fails

Check Python version:
```bash
python --version
```

Install exact dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

### Pre-commit hooks failing

Update and retry:
```bash
pre-commit autoupdate
pre-commit run --all-files
```

### PyTorch import errors

Reinstall PyTorch:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Tests running slowly

Use quick check script:
```bash
./scripts/quick-check.sh
```

## 📚 Further Reading

- GitHub Actions Workflows: `.github/workflows/README.md`
- Test Suite Documentation: `tests/README.md`
- Pre-commit Configuration: `.pre-commit-config.yaml`
- Package Configuration: `pyproject.toml`

## 💡 Tips

1. **During development:** Use `./scripts/quick-check.sh` for fast feedback
2. **Before pushing:** Use `./scripts/run-ci-checks.sh` to catch issues early
3. **Watch CI logs:** Check GitHub Actions tab for detailed output
4. **Use caching:** CI automatically caches pip packages
5. **Parallel jobs:** CI runs jobs in parallel for speed

## ⚡ Performance

| Check | Local | CI |
|-------|-------|-----|
| Quick Check | ~2 min | ~5 min |
| Full Check | ~10 min | ~15 min |
| Pre-commit Only | ~30 sec | ~2 min |
| Tests Only | ~5 min | ~5 min |

---

**Need help?** Check `.github/workflows/README.md` for detailed documentation.
