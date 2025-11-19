# PC-100-11A: Production Dockerfile

**Ticket ID:** PC-100-11A  
**Epic:** [Epic-11A: Documentation & Infrastructure](../epics/Epic-11A.md)  
**User Story:** US-11A.2 - Production Docker Infrastructure  
**Story Points:** 5  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Create a multi-stage, production-optimized Dockerfile with security hardening, minimal image size, proper health checks, and efficient layer caching for deploying PrevCarga in containerized environments.

**As a** DevOps engineer  
**I want** a production-ready Docker image  
**So that** I can deploy PrevCarga reliably and securely

---

## ✅ Acceptance Criteria

- [ ] Multi-stage Dockerfile created
- [ ] Image size optimized (<2GB)
- [ ] Security hardening implemented (non-root user)
- [ ] Health check configured
- [ ] Graceful shutdown handling
- [ ] Environment variable configuration
- [ ] Layer caching optimized
- [ ] Build time < 10 minutes
- [ ] .dockerignore configured
- [ ] Image labels and metadata added

---

## 🔧 Implementation Tasks

### 1. Create Dockerfile
- [ ] Base stage with Python 3.11-slim
- [ ] Dependencies stage with uv
- [ ] Production stage with application
- [ ] Non-root user creation
- [ ] Working directory setup
- [ ] Health check implementation

### 2. Optimize Build
- [ ] Efficient layer ordering
- [ ] Multi-stage build for size reduction
- [ ] Build cache optimization
- [ ] Dependency caching
- [ ] Remove unnecessary files

### 3. Security Hardening
- [ ] Run as non-root user
- [ ] Minimal base image
- [ ] No secrets in layers
- [ ] Read-only filesystem where possible
- [ ] Security scanning integration

### 4. Configuration
- [ ] Environment variable support
- [ ] Volume mounts for config
- [ ] Logging to stdout/stderr
- [ ] Signal handling for graceful shutdown

### 5. Create .dockerignore
- [ ] Exclude unnecessary files
- [ ] Exclude development files
- [ ] Exclude test files
- [ ] Exclude documentation

---

## 📂 Files to Create

```
prevcarga/
├── Dockerfile
├── .dockerignore
└── docker/
    ├── entrypoint.sh
    └── healthcheck.sh
```

---

## 🔧 Technical Implementation

### Dockerfile

```dockerfile
# Multi-stage Dockerfile for PrevCarga

# ============================================================================
# Base Stage: System dependencies and uv installation
# ============================================================================
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# ============================================================================
# Dependencies Stage: Install Python dependencies
# ============================================================================
FROM base AS dependencies

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen for reproducibility)
RUN uv sync --frozen --no-dev

# ============================================================================
# Production Stage: Application runtime
# ============================================================================
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PREVCARGA_ENV=production

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 prevcarga

# Set working directory
WORKDIR /app

# Copy virtual environment from dependencies stage
COPY --from=dependencies /app/.venv /app/.venv

# Copy application code
COPY --chown=prevcarga:prevcarga src/ ./src/
COPY --chown=prevcarga:prevcarga config/ ./config/
COPY --chown=prevcarga:prevcarga pyproject.toml ./

# Install application in development mode
RUN /app/.venv/bin/pip install -e .

# Copy scripts
COPY --chown=prevcarga:prevcarga docker/entrypoint.sh /entrypoint.sh
COPY --chown=prevcarga:prevcarga docker/healthcheck.sh /healthcheck.sh
RUN chmod +x /entrypoint.sh /healthcheck.sh

# Create directories for logs and cache
RUN mkdir -p /app/logs /app/cache && \
    chown -R prevcarga:prevcarga /app

# Switch to non-root user
USER prevcarga

# Expose port (if running as service)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD ["/healthcheck.sh"]

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["prevcarga", "--help"]

# Labels for metadata
LABEL maintainer="prevcarga-team@company.com" \
      version="1.0.0" \
      description="PrevCarga Unified Electric Load Forecasting System" \
      org.opencontainers.image.title="PrevCarga" \
      org.opencontainers.image.description="Unified Electric Load Forecasting System" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="PrevCarga Team" \
      org.opencontainers.image.licenses="MIT"
```

### .dockerignore

```
# Git
.git
.gitignore
.gitattributes

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/
.eggs/
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.hypothesis/

# Documentation
docs/build/
*.md
!README.md

# Data files
*.parquet
*.csv
data/
models/
logs/
*.log

# Jupyter
.ipynb_checkpoints
*.ipynb

# CI/CD
.github/
.gitlab-ci.yml
.travis.yml

# Docker
Dockerfile*
docker-compose*.yml
.dockerignore

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.bak
*.swp
tmp/
temp/
```

### Entrypoint Script (`docker/entrypoint.sh`)

```bash
#!/bin/bash
set -e

# PrevCarga Docker Entrypoint Script

echo "Starting PrevCarga container..."
echo "Environment: ${PREVCARGA_ENV:-development}"

# Wait for dependencies (if needed)
# Example: wait for S3, databases, etc.

# Handle signals for graceful shutdown
trap 'echo "Received SIGTERM, shutting down gracefully..."; exit 0' SIGTERM
trap 'echo "Received SIGINT, shutting down gracefully..."; exit 0' SIGINT

# Check AWS credentials if required
if [ -n "$AWS_ACCESS_KEY_ID" ] || [ -n "$AWS_PROFILE" ]; then
    echo "AWS credentials configured"
else
    echo "Warning: No AWS credentials found"
fi

# Validate configuration
if [ -f "/app/config/prevcarga.yaml" ]; then
    echo "Configuration file found"
else
    echo "Warning: No configuration file found at /app/config/prevcarga.yaml"
fi

# Execute command
exec "$@"
```

### Health Check Script (`docker/healthcheck.sh`)

```bash
#!/bin/bash

# PrevCarga Health Check Script

# Check if prevcarga CLI is available
if ! command -v prevcarga &> /dev/null; then
    echo "ERROR: prevcarga command not found"
    exit 1
fi

# Check prevcarga status
if prevcarga status &> /dev/null; then
    echo "Health check passed"
    exit 0
else
    echo "Health check failed"
    exit 1
fi
```

---

## 🧪 Testing & Validation

### Build and Test Commands

```bash
# Build the image
docker build -t prevcarga:latest .

# Check image size
docker images prevcarga:latest

# Run container with health check
docker run --rm prevcarga:latest prevcarga --version

# Test with environment variables
docker run --rm \
    -e AWS_PROFILE=test \
    -e PREVCARGA_ENV=production \
    prevcarga:latest prevcarga status

# Test health check
docker run --rm --name prevcarga-test -d prevcarga:latest sleep 60
docker inspect --format='{{.State.Health.Status}}' prevcarga-test
docker stop prevcarga-test

# Test as non-root user
docker run --rm prevcarga:latest whoami  # Should output: prevcarga

# Test with volume mounts
docker run --rm \
    -v $(pwd)/config:/app/config:ro \
    -v $(pwd)/logs:/app/logs \
    prevcarga:latest prevcarga config show
```

### Security Scanning

```bash
# Scan with Trivy
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy:latest image prevcarga:latest

# Scan with Snyk
snyk container test prevcarga:latest

# Check for vulnerabilities
docker scan prevcarga:latest
```

### Performance Testing

```bash
# Measure build time
time docker build -t prevcarga:latest .

# Measure startup time
time docker run --rm prevcarga:latest prevcarga --version

# Check image layers
docker history prevcarga:latest

# Analyze image size
dive prevcarga:latest
```

---

## 📝 Technical Notes

- Use Python 3.11-slim (smaller than full Python image)
- Multi-stage builds reduce final image size significantly
- Non-root user improves security
- Health checks enable container orchestration
- Layer caching speeds up rebuilds
- .dockerignore reduces build context size
- Signal handling ensures graceful shutdown
- Environment variables for configuration flexibility

---

## 🔗 Dependencies

**Depends On:**
- Complete source code implementation
- pyproject.toml with dependencies

**Blocks:**
- PC-101-11A: Docker Compose for Development
- PC-102-11A: Container Security and Optimization
- PC-104-11A: Deployment and Validation

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Dockerfile builds successfully
- [ ] Image size < 2GB
- [ ] Security scan passes (no high-severity issues)
- [ ] Health check works correctly
- [ ] Runs as non-root user
- [ ] Entrypoint script handles signals
- [ ] Build time < 10 minutes
- [ ] Technical review approved
- [ ] Committed to repository

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-19 | 1.0.0 | Initial ticket created from Epic-11A | System |

---

**Next Ticket:** [PC-101-11A: Docker Compose Development Environment](PC-101-11A-docker-compose-dev.md)
