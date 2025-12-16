# PC-103-12: Dockerfile

**Epic:** [EPIC-12: Documentation & Deploy](../epics/EPIC-12-documentation-deploy.md)
**Task Reference:** T-12.6
**Priority:** High
**Estimated Effort:** 1 day

---

## Summary

Create production-ready Dockerfile with multi-stage build, R 4.3+, renv for reproducible dependencies, non-root user, and health checks.

---

## Acceptance Criteria

- [ ] Multi-stage build for smaller image
- [ ] R 4.3+ as base
- [ ] renv for reproducible dependencies
- [ ] Non-root user for security
- [ ] Health check endpoint
- [ ] Optimized layer caching
- [ ] Environment variable support
- [ ] Image size < 2GB

---

## Technical Specification

### Dockerfile

```dockerfile
# ══════════════════════════════════════════════════════════════════════════════
# PrevCargaONS Production Dockerfile
# Multi-stage build for optimized image size
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder - Install dependencies and build package
# ──────────────────────────────────────────────────────────────────────────────
FROM rocker/r-ver:4.3.2 AS builder

LABEL maintainer="ONS <prevcarga@ons.org.br>"
LABEL description="PrevCargaONS - Electric Load Forecasting System (Builder Stage)"

# Install system dependencies for R packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    gfortran \
    # For arrow/parquet
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    # For data.table
    zlib1g-dev \
    # For lightgbm
    cmake \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    # For arrow
    libarrow-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV RENV_PATHS_CACHE=/renv/cache
ENV RENV_CONFIG_REPOS_OVERRIDE=https://packagemanager.posit.co/cran/latest

# Create directories
RUN mkdir -p /app /renv/cache

WORKDIR /app

# Copy renv files first (for layer caching)
COPY renv.lock renv.lock
COPY .Rprofile .Rprofile
COPY renv/activate.R renv/activate.R

# Install renv and restore packages
RUN R -e "install.packages('renv', repos='https://cloud.r-project.org')" \
    && R -e "renv::restore()"

# Copy application code
COPY R/ R/
COPY DESCRIPTION DESCRIPTION
COPY NAMESPACE NAMESPACE
COPY inst/ inst/

# Build and install package
RUN R CMD INSTALL --no-multiarch --with-keep.source .

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime - Minimal production image
# ──────────────────────────────────────────────────────────────────────────────
FROM rocker/r-ver:4.3.2 AS runtime

LABEL maintainer="ONS <prevcarga@ons.org.br>"
LABEL description="PrevCargaONS - Electric Load Forecasting System"
LABEL version="1.0.0"

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Runtime libraries
    libcurl4 \
    libssl3 \
    libxml2 \
    zlib1g \
    libarrow1200 \
    # Utilities
    tini \
    curl \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r prevcarga && useradd -r -g prevcarga prevcarga

# Create application directories
RUN mkdir -p /app/{data,models,config,logs,reports} \
    && chown -R prevcarga:prevcarga /app

# Set environment variables
ENV PREVCARGA_HOME=/app
ENV PREVCARGA_CONFIG=/app/config/config.yaml
ENV PREVCARGA_LOG_LEVEL=INFO
ENV R_LIBS_USER=/app/renv/library
ENV TZ=America/Sao_Paulo

# Copy renv cache from builder (only needed libraries)
COPY --from=builder /renv/cache /renv/cache
COPY --from=builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library

# Copy installed package
COPY --from=builder /usr/local/lib/R/library/prevcargaons /usr/local/lib/R/library/prevcargaons

# Copy application files
COPY --chown=prevcarga:prevcarga inst/config/config.default.yaml /app/config/config.yaml
COPY --chown=prevcarga:prevcarga inst/bin/prevcarga /usr/local/bin/prevcarga

# Make entrypoint executable
RUN chmod +x /usr/local/bin/prevcarga

# Set working directory
WORKDIR /app

# Switch to non-root user
USER prevcarga

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD prevcarga --version || exit 1

# Expose no ports (CLI application)
# EXPOSE 8080

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/prevcarga"]

# Default command (interactive mode)
CMD ["interactive"]
```

### .dockerignore

```dockerignore
# .dockerignore

# Git
.git
.gitignore
.gitattributes

# Documentation
docs/
*.md
!README.md

# IDE
.vscode/
.idea/
*.Rproj
.Rproj.user/

# Test files
tests/
testthat/

# Development
.devcontainer/
.github/
Makefile

# Build artifacts
*.tar.gz
*.Rcheck/
*.o
*.so

# Environment
.env
.env.*
*.local

# Cache
.cache/
__pycache__/
*.pyc

# Logs
*.log
logs/

# Temporary
tmp/
temp/
*.tmp

# OS files
.DS_Store
Thumbs.db

# renv (we copy specific files)
renv/library/
renv/local/
renv/staging/
!renv/activate.R

# Data (mounted as volume)
data/
models/
reports/
```

### Build Script

```bash
#!/bin/bash
# scripts/docker-build.sh

set -euo pipefail

# Configuration
IMAGE_NAME="ons/prevcarga"
VERSION="${1:-$(cat VERSION 2>/dev/null || echo '1.0.0')}"
REGISTRY="${DOCKER_REGISTRY:-}"

echo "Building PrevCargaONS Docker image"
echo "Version: ${VERSION}"
echo "Image: ${IMAGE_NAME}"

# Build image
docker build \
    --tag "${IMAGE_NAME}:${VERSION}" \
    --tag "${IMAGE_NAME}:latest" \
    --build-arg VERSION="${VERSION}" \
    --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
    --file Dockerfile \
    .

# Show image size
echo ""
echo "Image size:"
docker images "${IMAGE_NAME}:${VERSION}" --format "{{.Size}}"

# Optional: push to registry
if [[ -n "${REGISTRY}" ]]; then
    echo "Pushing to ${REGISTRY}..."
    docker tag "${IMAGE_NAME}:${VERSION}" "${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    docker tag "${IMAGE_NAME}:latest" "${REGISTRY}/${IMAGE_NAME}:latest"
    docker push "${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    docker push "${REGISTRY}/${IMAGE_NAME}:latest"
fi

echo ""
echo "Build complete!"
echo "Run with: docker run -it --rm ${IMAGE_NAME}:${VERSION}"
```

### Entrypoint Script

```bash
#!/bin/bash
# inst/bin/prevcarga (Docker entrypoint)

set -euo pipefail

# PrevCargaONS CLI Launcher
PREVCARGA_HOME="${PREVCARGA_HOME:-/app}"
PREVCARGA_SHELL="${PREVCARGA_HOME}/R/prevcarga_shell.R"

# If no arguments, start interactive mode
if [[ $# -eq 0 ]]; then
    exec Rscript --vanilla -e "prevcargaons::interactive_shell()"
fi

# Handle special commands
case "${1:-}" in
    --version|-v)
        Rscript --vanilla -e "cat(paste0('PrevCargaONS v', packageVersion('prevcargaons'), '\n'))"
        ;;
    --help|-h)
        Rscript --vanilla -e "prevcargaons::cli_help()"
        ;;
    interactive)
        exec Rscript --vanilla -e "prevcargaons::interactive_shell()"
        ;;
    train|predict|backtest|eval-model|gen-features|combine|reconcile|report|status)
        exec Rscript --vanilla -e "prevcargaons::cli_main()" -- "$@"
        ;;
    bash|sh)
        # Allow shell access for debugging
        exec /bin/bash
        ;;
    *)
        # Pass all arguments to CLI
        exec Rscript --vanilla -e "prevcargaons::cli_main()" -- "$@"
        ;;
esac
```

### Default Configuration

```yaml
# inst/config/config.default.yaml
# Default configuration for Docker deployment

project:
  name: "PrevCargaONS"
  version: "1.0.0"
  environment: "production"
  seed: 42
  timezone: "America/Sao_Paulo"

storage:
  backend: local
  local:
    base_path: /app/data
    models_path: /app/models

logging:
  level: "${PREVCARGA_LOG_LEVEL:-INFO}"
  format: json
  handlers:
    - type: console
      level: INFO

models:
  enabled:
    - lgbm
    - rf
    - hw
  default_horizons: [0, 1, 2, 3, 4, 5, 6, 7, 8]

parallel:
  enabled: true
  workers: auto
```

### Security Scanning

```bash
#!/bin/bash
# scripts/docker-scan.sh

set -euo pipefail

IMAGE="${1:-ons/prevcarga:latest}"

echo "Scanning ${IMAGE} for vulnerabilities..."

# Trivy scan
if command -v trivy &> /dev/null; then
    echo "Running Trivy..."
    trivy image --severity HIGH,CRITICAL "${IMAGE}"
fi

# Docker Scout (if available)
if docker scout version &> /dev/null 2>&1; then
    echo "Running Docker Scout..."
    docker scout cves "${IMAGE}"
fi

# Hadolint for Dockerfile
if command -v hadolint &> /dev/null; then
    echo "Running Hadolint..."
    hadolint Dockerfile
fi

echo "Scan complete!"
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | Build succeeds | Image created |
| TC-002 | Image size | < 2GB |
| TC-003 | Non-root user | Not running as root |
| TC-004 | Health check | Passes |
| TC-005 | --version works | Shows version |
| TC-006 | Interactive mode | Shell starts |
| TC-007 | Commands work | All CLI commands |
| TC-008 | Volume mounts | Data persists |
| TC-009 | Env vars | Config applied |
| TC-010 | Security scan | No critical vulns |

### Build and Test Commands

```bash
# Build image
./scripts/docker-build.sh 1.0.0

# Test basic functionality
docker run --rm ons/prevcarga:1.0.0 --version
docker run --rm ons/prevcarga:1.0.0 --help

# Test interactive mode
docker run -it --rm ons/prevcarga:1.0.0

# Test with volumes
docker run -it --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    ons/prevcarga:1.0.0 predict --area RJ

# Check user
docker run --rm ons/prevcarga:1.0.0 bash -c "whoami"
# Expected: prevcarga

# Check image size
docker images ons/prevcarga:1.0.0 --format "{{.Size}}"

# Security scan
./scripts/docker-scan.sh ons/prevcarga:1.0.0
```

---

## Dependencies

- None (infrastructure component)

---

## Definition of Done

- [ ] Dockerfile created and tested
- [ ] Multi-stage build working
- [ ] Image size < 2GB
- [ ] Non-root user configured
- [ ] Health check passing
- [ ] All CLI commands work
- [ ] Security scan clean
- [ ] Build script created
- [ ] .dockerignore optimized
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use rocker/r-ver for reliable R base image
- Multi-stage build significantly reduces size
- renv ensures reproducible dependencies
- Non-root user is security best practice
- Health check enables orchestration integration
- Tini as init system handles signals properly
