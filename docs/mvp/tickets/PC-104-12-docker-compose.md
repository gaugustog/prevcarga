# PC-104-12: Docker Compose

**Epic:** [EPIC-12: Documentation & Deploy](../epics/EPIC-12-documentation-deploy.md)
**Task Reference:** T-12.7
**Priority:** High
**Estimated Effort:** 0.5 days

---

## Summary

Create Docker Compose configuration for local development and testing with service definitions, volume mounts, environment configuration, and optional services (LocalStack for S3 emulation).

---

## Acceptance Criteria

- [ ] `docker-compose.yml` for production use
- [ ] `docker-compose.dev.yml` for development
- [ ] Volume mounts for persistence
- [ ] Environment variable configuration
- [ ] Optional LocalStack service for S3 emulation
- [ ] Health checks configured
- [ ] Override file support

---

## Technical Specification

### Production Docker Compose

```yaml
# docker-compose.yml
# PrevCargaONS Production Docker Compose

version: '3.8'

services:
  # ────────────────────────────────────────────────────────────────────────────
  # Main Application Service
  # ────────────────────────────────────────────────────────────────────────────
  prevcarga:
    image: ons/prevcarga:${PREVCARGA_VERSION:-latest}
    container_name: prevcarga
    hostname: prevcarga

    # Interactive mode
    stdin_open: true
    tty: true

    # Environment
    environment:
      - PREVCARGA_LOG_LEVEL=${PREVCARGA_LOG_LEVEL:-INFO}
      - PREVCARGA_CONFIG=/app/config/config.yaml
      - TZ=America/Sao_Paulo
      # AWS credentials (if using S3)
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-}
      - AWS_REGION=${AWS_REGION:-sa-east-1}

    # Volumes for data persistence
    volumes:
      - prevcarga-data:/app/data
      - prevcarga-models:/app/models
      - prevcarga-logs:/app/logs
      - prevcarga-reports:/app/reports
      # Custom config (optional)
      - ./config:/app/config:ro

    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

    # Health check
    healthcheck:
      test: ["CMD", "prevcarga", "--version"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

    # Restart policy
    restart: unless-stopped

    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

# ──────────────────────────────────────────────────────────────────────────────
# Named Volumes
# ──────────────────────────────────────────────────────────────────────────────
volumes:
  prevcarga-data:
    driver: local
  prevcarga-models:
    driver: local
  prevcarga-logs:
    driver: local
  prevcarga-reports:
    driver: local

# ──────────────────────────────────────────────────────────────────────────────
# Networks
# ──────────────────────────────────────────────────────────────────────────────
networks:
  default:
    name: prevcarga-network
    driver: bridge
```

### Development Docker Compose

```yaml
# docker-compose.dev.yml
# PrevCargaONS Development Docker Compose
# Usage: docker compose -f docker-compose.yml -f docker-compose.dev.yml up

version: '3.8'

services:
  # ────────────────────────────────────────────────────────────────────────────
  # Development Application Service
  # ────────────────────────────────────────────────────────────────────────────
  prevcarga:
    build:
      context: .
      dockerfile: Dockerfile
      target: builder  # Use builder stage for development

    environment:
      - PREVCARGA_LOG_LEVEL=DEBUG
      - R_LIBS_USER=/app/renv/library

    # Mount source code for hot reload
    volumes:
      - .:/app/src:cached
      - ./R:/app/R:cached
      - ./tests:/app/tests:cached
      - ./data:/app/data
      - ./models:/app/models
      - ./config:/app/config
      - ./logs:/app/logs
      - ./reports:/app/reports
      # Preserve renv cache
      - renv-cache:/renv/cache

    # Override command for development
    command: ["bash"]

    # Development resource limits (higher for testing)
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G

  # ────────────────────────────────────────────────────────────────────────────
  # LocalStack - S3 Emulation for Local Development
  # ────────────────────────────────────────────────────────────────────────────
  localstack:
    image: localstack/localstack:3.0
    container_name: localstack
    ports:
      - "4566:4566"            # LocalStack Gateway
      - "4510-4559:4510-4559"  # External services
    environment:
      - SERVICES=s3
      - DEBUG=0
      - DATA_DIR=/var/lib/localstack/data
      - DOCKER_HOST=unix:///var/run/docker.sock
    volumes:
      - localstack-data:/var/lib/localstack
      - /var/run/docker.sock:/var/run/docker.sock
      - ./scripts/localstack-init.sh:/etc/localstack/init/ready.d/init.sh:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:4566/_localstack/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ────────────────────────────────────────────────────────────────────────────
  # RStudio Server (Optional - for interactive development)
  # ────────────────────────────────────────────────────────────────────────────
  rstudio:
    image: rocker/rstudio:4.3.2
    container_name: rstudio
    ports:
      - "8787:8787"
    environment:
      - DISABLE_AUTH=true
      - ROOT=TRUE
    volumes:
      - .:/home/rstudio/prevcarga
      - renv-cache:/renv/cache
    profiles:
      - dev-full  # Only start with: docker compose --profile dev-full up

volumes:
  renv-cache:
    driver: local
  localstack-data:
    driver: local
```

### LocalStack Initialization Script

```bash
#!/bin/bash
# scripts/localstack-init.sh
# Initialize LocalStack with test buckets

set -e

echo "Initializing LocalStack S3..."

# Create test buckets
awslocal s3 mb s3://prevcarga-dev-data
awslocal s3 mb s3://prevcarga-dev-models

# Create sample directory structure
awslocal s3api put-object --bucket prevcarga-dev-data --key carga/
awslocal s3api put-object --bucket prevcarga-dev-data --key previsao/
awslocal s3api put-object --bucket prevcarga-dev-models --key lgbm/
awslocal s3api put-object --bucket prevcarga-dev-models --key rf/

echo "LocalStack S3 initialized!"
awslocal s3 ls
```

### Environment File Template

```bash
# .env.example
# PrevCargaONS Docker Compose Environment Variables
# Copy to .env and customize

# ──────────────────────────────────────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────────────────────────────────────
PREVCARGA_VERSION=latest
PREVCARGA_LOG_LEVEL=INFO

# ──────────────────────────────────────────────────────────────────────────────
# AWS Credentials (for S3 storage backend)
# ──────────────────────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=sa-east-1

# ──────────────────────────────────────────────────────────────────────────────
# LocalStack (for development)
# ──────────────────────────────────────────────────────────────────────────────
# Use these when testing with LocalStack
# AWS_ACCESS_KEY_ID=test
# AWS_SECRET_ACCESS_KEY=test
# AWS_ENDPOINT_URL=http://localstack:4566

# ──────────────────────────────────────────────────────────────────────────────
# Resource Limits
# ──────────────────────────────────────────────────────────────────────────────
# Uncomment to override default limits
# COMPOSE_CPU_LIMIT=4
# COMPOSE_MEMORY_LIMIT=8G
```

### Makefile for Docker Operations

```makefile
# Makefile
# Docker Compose helpers for PrevCargaONS

.PHONY: help build up down logs shell test clean

# Default target
help:
	@echo "PrevCargaONS Docker Commands"
	@echo ""
	@echo "  make build      Build Docker image"
	@echo "  make up         Start services"
	@echo "  make down       Stop services"
	@echo "  make logs       View logs"
	@echo "  make shell      Open shell in container"
	@echo "  make test       Run tests in container"
	@echo "  make dev        Start development environment"
	@echo "  make clean      Remove containers and volumes"
	@echo ""

# Build image
build:
	docker compose build

# Start production services
up:
	docker compose up -d

# Start development services (with LocalStack)
dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Start development with RStudio
dev-full:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile dev-full up -d

# Stop services
down:
	docker compose down

# View logs
logs:
	docker compose logs -f prevcarga

# Open shell
shell:
	docker compose exec prevcarga bash

# Run interactive mode
interactive:
	docker compose run --rm prevcarga interactive

# Run prediction
predict:
	docker compose run --rm prevcarga predict $(ARGS)

# Run training
train:
	docker compose run --rm prevcarga train $(ARGS)

# Run backtest
backtest:
	docker compose run --rm prevcarga backtest $(ARGS)

# Run tests
test:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm prevcarga \
		R -e "testthat::test_local()"

# Clean up
clean:
	docker compose down -v --remove-orphans
	docker system prune -f

# Remove all (including images)
clean-all: clean
	docker rmi ons/prevcarga:latest || true
```

### Usage Examples

```bash
# ──────────────────────────────────────────────────────────────────────────────
# Production Usage
# ──────────────────────────────────────────────────────────────────────────────

# Start service
docker compose up -d

# Run interactive shell
docker compose run --rm prevcarga

# Run specific command
docker compose run --rm prevcarga predict --area RJ --horizon 1

# View logs
docker compose logs -f

# Stop services
docker compose down

# ──────────────────────────────────────────────────────────────────────────────
# Development Usage
# ──────────────────────────────────────────────────────────────────────────────

# Start with LocalStack (S3 emulation)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Start with RStudio
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile dev-full up -d
# Access RStudio at http://localhost:8787

# Run tests
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm prevcarga \
    R -e "testthat::test_local()"

# Use LocalStack S3
docker compose exec prevcarga bash
# Inside container:
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_ENDPOINT_URL=http://localstack:4566
prevcarga predict --area RJ --storage s3

# ──────────────────────────────────────────────────────────────────────────────
# Using Makefile
# ──────────────────────────────────────────────────────────────────────────────

make build          # Build image
make up             # Start production
make dev            # Start development
make shell          # Open shell
make test           # Run tests
make clean          # Clean up
```

---

## Test Cases

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| TC-001 | `docker compose up` | Service starts |
| TC-002 | Volumes persist | Data survives restart |
| TC-003 | Environment vars | Config applied |
| TC-004 | Health check | Reports healthy |
| TC-005 | LocalStack starts | S3 accessible |
| TC-006 | RStudio starts | Web UI available |
| TC-007 | Makefile works | All targets execute |
| TC-008 | Resource limits | Applied correctly |
| TC-009 | Logs captured | JSON format |
| TC-010 | Network isolation | Services communicate |

---

## Dependencies

- PC-103-12: Dockerfile

---

## Definition of Done

- [ ] docker-compose.yml created
- [ ] docker-compose.dev.yml created
- [ ] .env.example documented
- [ ] Makefile created
- [ ] LocalStack integration working
- [ ] All services start correctly
- [ ] Health checks pass
- [ ] Documentation complete
- [ ] Code reviewed and approved
- [ ] Merged to develop branch

---

## Notes

- Use named volumes for data persistence
- LocalStack enables offline S3 development
- RStudio service is optional for GUI development
- Makefile simplifies common operations
- Resource limits prevent system overload
