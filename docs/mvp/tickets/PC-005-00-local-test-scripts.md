# PC-005-00: Local Test Scripts

**Ticket ID:** PC-005-00  
**Epic:** [Epic-00: Project Foundation & Setup](../epics/Epic-00.md)  
**User Story:** US-00.5  
**Story Points:** 3  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Create executable shell scripts for running tests, linting, and code formatting to ensure code quality before committing changes to the repository.

**As a** developer  
**I want** local scripts to run tests, linting, and formatting  
**So that** I can ensure code quality before committing

---

## ✅ Acceptance Criteria

- [ ] pytest configured with sensible defaults
- [ ] Test discovery working for `tests/` directory
- [ ] ruff configured for linting (replacing flake8, isort)
- [ ] black configured for code formatting
- [ ] mypy configured for type checking
- [ ] Shell scripts created: `scripts/test.sh`, `scripts/lint.sh`, `scripts/format.sh`
- [ ] `scripts/check-all.sh` runs all checks in sequence
- [ ] All scripts executable and documented

---

## 🔧 Implementation Tasks

### 1. Configure pytest
- [ ] Add pytest configuration to `pyproject.toml`:
  - Test paths: `tests/`
  - Test file pattern: `test_*.py`
  - Coverage settings: `--cov=src --cov-report=term-missing --cov-report=html`
  - Verbosity: `-v`
- [ ] Create `tests/conftest.py` with common fixtures
- [ ] Create sample test `tests/test_sample.py`

### 2. Configure ruff
- [ ] Add ruff configuration to `pyproject.toml`:
  - Line length: 100
  - Target Python version: 3.11
  - Select rules: E, F, I, N, UP, B, A, C4, etc.
  - Ignore: E501 (line too long - handled by black)
- [ ] Configure import sorting
- [ ] Configure code complexity checks

### 3. Configure black
- [ ] Add black configuration to `pyproject.toml`:
  - Line length: 100
  - Target Python version: py311
  - Skip string normalization: false

### 4. Configure mypy
- [ ] Add mypy configuration to `pyproject.toml`:
  - Python version: 3.11
  - Strict mode settings
  - Disallow untyped definitions
  - Warn on return any
- [ ] Configure ignore missing imports for third-party libraries (if needed)

### 5. Create Test Script
- [ ] Create `scripts/test.sh`
- [ ] Add shebang: `#!/bin/bash`
- [ ] Add `set -e` to exit on error
- [ ] Run pytest with coverage
- [ ] Generate HTML coverage report
- [ ] Make script executable: `chmod +x scripts/test.sh`

### 6. Create Lint Script
- [ ] Create `scripts/lint.sh`
- [ ] Add shebang and `set -e`
- [ ] Run ruff linter on `src/` and `tests/`
- [ ] Run mypy type checker on `src/`
- [ ] Make script executable

### 7. Create Format Script
- [ ] Create `scripts/format.sh`
- [ ] Add shebang and `set -e`
- [ ] Run black formatter on `src/` and `tests/`
- [ ] Run ruff to sort imports
- [ ] Make script executable

### 8. Create Check-All Script
- [ ] Create `scripts/check-all.sh`
- [ ] Add shebang and `set -e`
- [ ] Call format script
- [ ] Call lint script
- [ ] Call test script
- [ ] Add progress messages between steps
- [ ] Make script executable

### 9. Create Sample Test
- [ ] Create `tests/test_sample.py` with:
  - Test for project initialization
  - Test for basic imports
  - Placeholder tests for future modules
- [ ] Ensure test passes

### 10. Documentation
- [ ] Update README.md with:
  - Testing section
  - Linting section
  - Formatting section
  - Pre-commit instructions
- [ ] Add script usage examples
- [ ] Document tool configuration

---

## 📄 Configuration in pyproject.toml

Add these sections to the existing `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=src --cov-report=term-missing --cov-report=html"

[tool.ruff]
line-length = 100
target-version = "py311"
select = [
    "E",   # pycodestyle errors
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "YTT", # flake8-2020
    "ASYNC", # flake8-async
    "S",   # flake8-bandit
    "B",   # flake8-bugbear
    "A",   # flake8-builtins
    "C4",  # flake8-comprehensions
    "DTZ", # flake8-datetimez
    "T10", # flake8-debugger
    "EM",  # flake8-errmsg
    "ISC", # flake8-implicit-str-concat
    "ICN", # flake8-import-conventions
    "PIE", # flake8-pie
    "PT",  # flake8-pytest-style
    "Q",   # flake8-quotes
    "RSE", # flake8-raise
    "RET", # flake8-return
    "SIM", # flake8-simplify
    "TID", # flake8-tidy-imports
    "ARG", # flake8-unused-arguments
    "PTH", # flake8-use-pathlib
    "PD",  # pandas-vet
    "PL",  # pylint
    "NPY", # numpy-specific rules
    "RUF", # ruff-specific rules
]
ignore = ["E501"]  # Line too long (handled by black)

[tool.black]
line-length = 100
target-version = ["py311"]
skip-string-normalization = false

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_unimported = false
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
check_untyped_defs = true
strict_equality = true
```

---

## 📜 Shell Scripts

### scripts/test.sh

```bash
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
echo "✅ Tests passed!"
echo "Coverage report generated in htmlcov/"
echo "========================================"
```

### scripts/lint.sh

```bash
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

uv run mypy src/

echo ""
echo "========================================"
echo "✅ Linting passed!"
echo "========================================"
```

### scripts/format.sh

```bash
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
echo "✅ Code formatted successfully!"
echo "========================================"
```

### scripts/check-all.sh

```bash
#!/bin/bash
set -e

echo ""
echo "========================================"
echo "Running all code quality checks"
echo "========================================"
echo ""

echo "Step 1/3: Formatting code..."
./scripts/format.sh

echo ""
echo "Step 2/3: Linting code..."
./scripts/lint.sh

echo ""
echo "Step 3/3: Running tests..."
./scripts/test.sh

echo ""
echo "========================================"
echo "✅ All checks passed!"
echo "========================================"
```

---

## 💻 Sample Test File

### tests/test_sample.py

```python
"""Sample test file to verify test infrastructure."""
import sys
from pathlib import Path


def test_project_initialized() -> None:
    """Verify project structure is initialized."""
    project_root = Path(__file__).parent.parent
    
    # Check key directories exist
    assert (project_root / "src").exists()
    assert (project_root / "tests").exists()
    assert (project_root / "config").exists()
    assert (project_root / "scripts").exists()


def test_python_version() -> None:
    """Verify Python version is 3.11+."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version}"


def test_core_imports() -> None:
    """Verify core packages can be imported."""
    try:
        import boto3
        import pandas
        import numpy
        import pydantic
        import yaml
    except ImportError as e:
        raise AssertionError(f"Failed to import core package: {e}")


def test_dev_imports() -> None:
    """Verify dev packages can be imported."""
    try:
        import pytest
        import black
        import mypy
    except ImportError as e:
        raise AssertionError(f"Failed to import dev package: {e}")
```

---

## 🧪 Testing & Validation

### Validation Commands

```bash
# Make all scripts executable
chmod +x scripts/*.sh

# Test each script individually
./scripts/format.sh
./scripts/lint.sh
./scripts/test.sh

# Run all checks
./scripts/check-all.sh

# Verify coverage report generated
ls -la htmlcov/
open htmlcov/index.html  # View coverage in browser

# Test that scripts exit on error
# (temporarily break a test and verify script fails)
```

### Success Criteria
- [ ] All scripts are executable
- [ ] `test.sh` runs pytest and generates coverage report
- [ ] `lint.sh` runs ruff and mypy without errors
- [ ] `format.sh` formats code consistently
- [ ] `check-all.sh` runs all checks in sequence
- [ ] Sample test passes
- [ ] Coverage report shows >80% coverage (for initial modules)
- [ ] Scripts exit with error code on failure

---

## 📝 Technical Notes

- Use `set -e` in scripts to exit immediately on error
- Use `uv run` prefix to ensure commands use virtual environment
- Scripts should be idempotent (safe to run multiple times)
- Consider adding pre-commit hooks in future (optional)
- HTML coverage report helps identify untested code
- mypy strict mode catches type errors early
- ruff is faster than flake8 + isort combined

---

## 🔗 Dependencies

**Depends On:**
- PC-001-00: Repository Structure Setup
- PC-002-00: Python Environment with uv

**Blocks:**
- All future tickets (should run checks before merging)

**Related:**
- PC-003-00: S3 Storage Configuration (tests for S3 client)
- PC-004-00: Structured Logging Framework (tests for logger)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] All implementation tasks completed
- [ ] pytest, ruff, black, mypy configured in `pyproject.toml`
- [ ] All 4 shell scripts created and executable
- [ ] Sample test created and passing
- [ ] All validation commands pass
- [ ] Code coverage >80% for existing modules
- [ ] Documentation updated in README.md
- [ ] Team trained on using scripts
- [ ] Epic-00 complete and ready for Epic-01

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Epic Status:** This is the final ticket for Epic-00. Upon completion, all foundation infrastructure will be ready for [Epic-01: Data Infrastructure Layer](../epics/Epic-01.md).
