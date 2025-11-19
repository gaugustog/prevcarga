# Epic-00: Project Foundation & Setup

**Epic ID:** Epic-00  
**Epic Name:** Project Foundation & Setup  
**Phase:** Phase 0  
**Duration:** 1 week  
**Priority:** Critical  
**Dependencies:** None

---

## 🎯 Epic Goal

Establish the foundational project infrastructure and development environment for the PrevCarga Unified System, enabling efficient development of the electric load forecasting platform.

---

## 📋 User Stories

### US-00.1: Repository Structure Setup
**As a** developer  
**I want** a well-organized repository structure  
**So that** I can easily navigate and maintain the codebase

**Acceptance Criteria:**
- [ ] Root directory structure created following Python best practices
- [ ] Source code organized in `src/` directory with proper module structure
- [ ] Configuration files in `config/` directory
- [ ] Documentation in `docs/` directory
- [ ] Tests in `tests/` directory mirroring `src/` structure
- [ ] `.gitignore` configured for Python, IDE files, and data artifacts
- [ ] `README.md` with project overview and setup instructions

**Tasks:**
- [ ] Create directory structure
- [ ] Initialize Git repository
- [ ] Create initial README.md
- [ ] Configure .gitignore
- [ ] Create placeholder modules in src/

---

### US-00.2: Python Environment with uv
**As a** developer  
**I want** a modern Python package management system  
**So that** I can quickly install and manage dependencies reproducibly

**Acceptance Criteria:**
- [ ] Python 3.11+ configured
- [ ] uv installed and configured
- [ ] `pyproject.toml` created with project metadata
- [ ] Core dependencies defined (boto3, pandas, numpy, pydantic)
- [ ] Dev dependencies defined (pytest, ruff, black, mypy)
- [ ] `uv.lock` file generated for reproducible builds
- [ ] Virtual environment creation documented

**Tasks:**
- [ ] Install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [ ] Create pyproject.toml with project metadata
- [ ] Add core dependencies via `uv add`
- [ ] Add dev dependencies via `uv add --dev`
- [ ] Generate uv.lock file
- [ ] Document environment setup in README

---

### US-00.3: S3 Storage Configuration
**As a** developer  
**I want** S3 storage configured and accessible  
**So that** I can store and retrieve raw data, models, and results

**Acceptance Criteria:**
- [ ] AWS credentials configured (via environment variables or AWS CLI)
- [ ] S3 bucket created or identified: `prevcarga-bucket-sandbox`
- [ ] Bucket structure documented (raw_data/, features/, models/, results/, cache/)
- [ ] boto3 S3 client wrapper created with basic operations (list, get, put, delete)
- [ ] Connection test script created and passing
- [ ] S3 configuration module created in `src/storage/`

**Tasks:**
- [ ] Configure AWS credentials
- [ ] Create/verify S3 bucket existence
- [ ] Create `src/storage/s3_client.py` with basic operations
- [ ] Create `src/storage/config.py` for storage configuration
- [ ] Write connection test script
- [ ] Document S3 setup in README

---

### US-00.4: Structured Logging Framework
**As a** developer  
**I want** a structured logging system  
**So that** I can debug issues and monitor system behavior effectively

**Acceptance Criteria:**
- [ ] Logging configuration created using Python's `logging` module
- [ ] JSON structured logging format implemented
- [ ] Log levels configurable via environment variable or config file
- [ ] Logger includes: timestamp, level, module, function, message, context
- [ ] Console handler configured for development
- [ ] File handler configured with rotation
- [ ] Logging utility module created in `src/utils/`
- [ ] Example usage documented

**Tasks:**
- [ ] Create `src/utils/logger.py` with structured logging
- [ ] Create logging configuration in `config/logging.yaml`
- [ ] Implement JSON formatter
- [ ] Add console and file handlers
- [ ] Create logger factory function
- [ ] Write logging usage examples
- [ ] Document logging in README

---

### US-00.5: Local Test Scripts
**As a** developer  
**I want** local scripts to run tests, linting, and formatting  
**So that** I can ensure code quality before committing

**Acceptance Criteria:**
- [ ] pytest configured with sensible defaults
- [ ] Test discovery working for `tests/` directory
- [ ] ruff configured for linting (replacing flake8, isort)
- [ ] black configured for code formatting
- [ ] mypy configured for type checking
- [ ] Shell scripts created: `scripts/test.sh`, `scripts/lint.sh`, `scripts/format.sh`
- [ ] `scripts/check-all.sh` runs all checks in sequence
- [ ] All scripts executable and documented

**Tasks:**
- [ ] Create `pytest.ini` or configure in `pyproject.toml`
- [ ] Create `ruff.toml` configuration
- [ ] Configure black in `pyproject.toml`
- [ ] Configure mypy in `pyproject.toml`
- [ ] Create shell scripts in `scripts/` directory
- [ ] Write sample test in `tests/test_sample.py`
- [ ] Document script usage in README

---

## 🏗️ Technical Architecture

### Directory Structure
```
prevcarga/
├── .git/
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
├── config/
│   ├── logging.yaml
│   └── storage.yaml
├── docs/
│   ├── mvp/
│   │   ├── mvp-plan.md
│   │   └── epics/
│   │       ├── index.md
│   │       └── Epic-00.md
│   └── api/
├── scripts/
│   ├── test.sh
│   ├── lint.sh
│   ├── format.sh
│   └── check-all.sh
├── src/
│   ├── __init__.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── s3_client.py
│   │   └── config.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py
│   └── data/          # Placeholder for Epic-01
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_sample.py
│   └── storage/
│       └── test_s3_client.py
└── notebooks/         # For exploration (optional)
    └── .gitkeep
```

### Key Dependencies

**Core:**
```toml
[project]
name = "prevcarga"
version = "0.1.0"
description = "Unified Electric Load Forecasting System"
requires-python = ">=3.11"
dependencies = [
    "boto3>=1.34.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "pydantic>=2.5.0",
    "pyyaml>=6.0.1",
]
```

**Development:**
```toml
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
```

---

## 🧪 Testing Strategy

### Unit Tests
- **Coverage Target:** 80%+
- **Focus:** Utility functions, S3 client operations, logger initialization
- **Tools:** pytest, pytest-cov

### Integration Tests
- **Focus:** S3 connectivity, logging to file system
- **Tools:** pytest, moto (for S3 mocking)

### Test Examples
```python
# tests/test_sample.py
def test_project_initialized():
    """Verify project structure is initialized."""
    assert True

# tests/storage/test_s3_client.py
def test_s3_client_initialization():
    """Test S3 client can be initialized."""
    from src.storage.s3_client import S3Client
    client = S3Client(bucket_name="test-bucket")
    assert client is not None
```

---

## 📝 Configuration Files

### pyproject.toml (Core)
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
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "black>=23.12.0",
    "mypy>=1.7.0",
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

### config/logging.yaml
```yaml
version: 1
disable_existing_loggers: false

formatters:
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: "%(asctime)s %(name)s %(levelname)s %(message)s"
  
  standard:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: standard
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: logs/prevcarga.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  prevcarga:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console]
```

### config/storage.yaml
```yaml
s3:
  bucket: prevcarga-bucket-sandbox
  region: us-east-1
  
  paths:
    raw_data: raw_data/
    features: features/
    models: models/
    results: results/
    cache: cache/
  
  timeouts:
    connect: 5
    read: 30
  
  retry:
    max_attempts: 3
    mode: adaptive
```

---

## 📜 Shell Scripts

### scripts/test.sh
```bash
#!/bin/bash
set -e

echo "Running tests with coverage..."
uv run pytest tests/ \
    -v \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html

echo "Coverage report generated in htmlcov/"
```

### scripts/lint.sh
```bash
#!/bin/bash
set -e

echo "Running ruff linter..."
uv run ruff check src/ tests/

echo "Running mypy type checker..."
uv run mypy src/
```

### scripts/format.sh
```bash
#!/bin/bash
set -e

echo "Formatting code with black..."
uv run black src/ tests/

echo "Sorting imports with ruff..."
uv run ruff check --select I --fix src/ tests/

echo "Code formatted successfully!"
```

### scripts/check-all.sh
```bash
#!/bin/bash
set -e

echo "=== Running all checks ==="
echo ""

echo "1. Code formatting check..."
./scripts/format.sh

echo ""
echo "2. Linting..."
./scripts/lint.sh

echo ""
echo "3. Tests..."
./scripts/test.sh

echo ""
echo "=== All checks passed! ==="
```

---

## ✅ Success Criteria & Validation

### Checklist
- [ ] Repository structure matches specified layout
- [ ] uv environment creation works: `uv venv && source .venv/bin/activate`
- [ ] Dependencies install successfully: `uv sync`
- [ ] S3 connection test passes
- [ ] Logger creates log file with proper JSON format
- [ ] `scripts/test.sh` runs and all tests pass
- [ ] `scripts/lint.sh` runs without errors
- [ ] `scripts/format.sh` formats code correctly
- [ ] `scripts/check-all.sh` completes successfully
- [ ] README.md has clear setup instructions

### Validation Commands
```bash
# 1. Environment setup
uv venv
source .venv/bin/activate
uv sync

# 2. Run all checks
chmod +x scripts/*.sh
./scripts/check-all.sh

# 3. Verify S3 access
uv run python -c "from src.storage.s3_client import S3Client; print('S3 OK')"

# 4. Verify logging
uv run python -c "from src.utils.logger import get_logger; logger = get_logger(__name__); logger.info('Test'); print('Logging OK')"
```

---

## 📊 Definition of Done

- [ ] All user stories completed and accepted
- [ ] All acceptance criteria met
- [ ] All tasks checked off
- [ ] All validation commands pass
- [ ] Code review completed (if applicable)
- [ ] Documentation updated (README.md)
- [ ] No critical or high-priority issues open
- [ ] Ready for Epic-01 to start

---

## 🔗 Dependencies & Relationships

**Depends On:** None (this is the foundation epic)

**Blocks:**
- Epic-01: Data Infrastructure Layer
- All subsequent epics

**Related Documents:**
- [MVP Plan](../mvp-plan.md)
- [Epic Index](index.md)

---

## 📈 Metrics & KPIs

- **Time to Environment Setup:** < 10 minutes
- **Test Execution Time:** < 10 seconds (for initial tests)
- **Code Quality:** 0 linting errors, 100% type coverage in utils/storage
- **Documentation Completeness:** README covers all setup steps

---

## 🚧 Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| AWS credentials not available | High | Low | Use AWS SSO or IAM roles; document setup |
| S3 bucket permissions issues | Medium | Medium | Verify IAM policies; use sandbox bucket |
| Python version mismatch | Low | Low | Document Python 3.11+ requirement clearly |
| uv installation issues | Low | Low | Provide alternative pip-based fallback |

---

## 📝 Notes

- Keep this epic simple and focused on foundation only
- Don't over-engineer - Epic-01 will build on this
- Prioritize getting S3 and logging working reliably
- All configuration should be externalized (no hardcoded values)
- Use environment variables for sensitive data (AWS credentials)

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-17 | 1.0.0 | Initial Epic-00 created | System |

---

**Status:** 📝 Draft  
**Next Epic:** [Epic-01: Data Infrastructure Layer](Epic-01.md)
