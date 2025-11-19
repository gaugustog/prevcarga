# PC-002-00: Python Environment with uv

**Ticket ID:** PC-002-00  
**Epic:** [Epic-00: Project Foundation & Setup](../epics/Epic-00.md)  
**User Story:** US-00.2  
**Story Points:** 3  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Set up a modern Python package management system using uv for fast dependency installation and reproducible builds across development, testing, and production environments.

**As a** developer  
**I want** a modern Python package management system  
**So that** I can quickly install and manage dependencies reproducibly

---

## ✅ Acceptance Criteria

- [ ] Python 3.11+ configured
- [ ] uv installed and configured
- [ ] `pyproject.toml` created with project metadata
- [ ] Core dependencies defined (boto3, pandas, numpy, pydantic)
- [ ] Dev dependencies defined (pytest, ruff, black, mypy)
- [ ] `uv.lock` file generated for reproducible builds
- [ ] Virtual environment creation documented

---

## 🔧 Implementation Tasks

### 1. Install uv
- [ ] Install uv using: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] Verify installation: `uv --version`
- [ ] Add uv to PATH if necessary
- [ ] Document installation steps in README.md

### 2. Create pyproject.toml
- [ ] Create `pyproject.toml` in project root
- [ ] Add project metadata section:
  - name: "prevcarga"
  - version: "0.1.0"
  - description: "Unified Electric Load Forecasting System"
  - authors: [{name = "PrevCarga Team"}]
  - requires-python: ">=3.11"
  - readme: "README.md"
  - license: {text = "MIT"}
- [ ] Add core dependencies section
- [ ] Add optional dev dependencies section
- [ ] Add tool configurations (pytest, ruff, black, mypy)

### 3. Add Core Dependencies
- [ ] Add boto3>=1.34.0 via `uv add boto3>=1.34.0`
- [ ] Add pandas>=2.2.0 via `uv add pandas>=2.2.0`
- [ ] Add numpy>=1.26.0 via `uv add numpy>=1.26.0`
- [ ] Add pydantic>=2.5.0 via `uv add pydantic>=2.5.0`
- [ ] Add pyyaml>=6.0.1 via `uv add pyyaml>=6.0.1`
- [ ] Add python-json-logger for structured logging via `uv add python-json-logger`

### 4. Add Development Dependencies
- [ ] Add pytest>=7.4.0 via `uv add --dev pytest>=7.4.0`
- [ ] Add pytest-cov>=4.1.0 via `uv add --dev pytest-cov>=4.1.0`
- [ ] Add pytest-asyncio>=0.21.0 via `uv add --dev pytest-asyncio>=0.21.0`
- [ ] Add ruff>=0.1.0 via `uv add --dev ruff>=0.1.0`
- [ ] Add black>=23.12.0 via `uv add --dev black>=23.12.0`
- [ ] Add mypy>=1.7.0 via `uv add --dev mypy>=1.7.0`
- [ ] Add ipython>=8.18.0 via `uv add --dev ipython>=8.18.0`

### 5. Configure Tool Settings in pyproject.toml
- [ ] Configure pytest settings (testpaths, coverage)
- [ ] Configure ruff settings (line-length, target-version, select, ignore)
- [ ] Configure black settings (line-length, target-version)
- [ ] Configure mypy settings (python_version, strict settings)

### 6. Generate Lock File and Documentation
- [ ] Run `uv lock` to generate `uv.lock` file
- [ ] Verify lock file creation
- [ ] Update README.md with:
  - Prerequisites (Python 3.11+)
  - uv installation instructions
  - Virtual environment creation: `uv venv`
  - Environment activation: `source .venv/bin/activate`
  - Dependency installation: `uv sync`

---

## 📄 Configuration Files

### pyproject.toml (Core Sections)

```toml
[project]
name = "prevcarga"
version = "0.1.0"
description = "Unified Electric Load Forecasting System"
authors = [{name = "PrevCarga Team"}]
requires-python = ">=3.11"
readme = "README.md"
license = {text = "MIT"}

dependencies = [
    "boto3>=1.34.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "pydantic>=2.5.0",
    "pyyaml>=6.0.1",
    "python-json-logger>=2.0.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "black>=23.12.0",
    "mypy>=1.7.0",
    "ipython>=8.18.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=src --cov-report=term-missing"

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "N", "UP", "YTT", "ASYNC", "S", "B", "A", "C4", "DTZ", "T10", "EM", "ISC", "ICN", "PIE", "PT", "Q", "RSE", "RET", "SIM", "TID", "ARG", "PTH", "PD", "PL", "NPY", "RUF"]
ignore = ["E501"]

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

---

## 🧪 Testing & Validation

### Validation Commands
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv sync

# Verify Python version
python --version  # Should be 3.11+

# Verify core packages installed
python -c "import boto3; import pandas; import numpy; import pydantic; print('Core packages OK')"

# Verify dev packages installed
python -c "import pytest; import ruff; import black; import mypy; print('Dev packages OK')"

# Verify lock file exists
ls -la uv.lock
```

### Success Criteria
- [ ] uv installed and accessible
- [ ] Python 3.11+ available
- [ ] Virtual environment created successfully
- [ ] All dependencies install without errors
- [ ] `uv.lock` file exists and is valid
- [ ] Package imports work correctly

---

## 📝 Technical Notes

- uv is significantly faster than pip for package installation
- `uv.lock` ensures reproducible builds across environments
- Virtual environment should be in `.venv/` (already in .gitignore)
- Use `uv sync` to install dependencies from lock file
- Use `uv add <package>` to add new dependencies
- Use `uv add --dev <package>` for development dependencies

---

## 🔗 Dependencies

**Depends On:**
- PC-001-00: Repository Structure Setup

**Blocks:**
- PC-003-00: S3 Storage Configuration
- PC-004-00: Structured Logging Framework
- PC-005-00: Local Test Scripts

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] All implementation tasks completed
- [ ] Validation commands pass
- [ ] `pyproject.toml` committed to repository
- [ ] `uv.lock` committed to repository
- [ ] README.md updated with setup instructions
- [ ] Team can reproduce environment setup
- [ ] Ready for next ticket (PC-003-00)

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-18 | 1.0.0 | Initial ticket created | System |

---

**Next Ticket:** [PC-003-00: S3 Storage Configuration](PC-003-00-s3-storage-configuration.md)
