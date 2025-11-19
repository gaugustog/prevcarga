# PC-101-11A: Docker Compose Development Environment

**Ticket ID:** PC-101-11A  
**Epic:** [Epic-11A: Documentation & Infrastructure](../epics/Epic-11A.md)  
**User Story:** US-11A.2 - Production Docker Infrastructure  
**Story Points:** 3  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Create Docker Compose configuration for local development environment with hot-reload, volume mounts, and optional monitoring services for efficient local development and testing.

**As a** developer  
**I want** a Docker Compose development environment  
**So that** I can develop and test PrevCarga locally with minimal setup

---

## ✅ Acceptance Criteria

- [ ] docker-compose.yml created for development
- [ ] docker-compose.prod.yml created for production-like testing
- [ ] Volume mounts for live code changes
- [ ] Environment variable configuration
- [ ] Optional monitoring services (Prometheus, Grafana)
- [ ] Network configuration for service communication
- [ ] Development database/storage if needed
- [ ] Documentation for usage

---

## 🔧 Implementation Tasks

### 1. Create Development Compose File
- [ ] Main prevcarga service configuration
- [ ] Volume mounts for source code
- [ ] Volume mounts for config and logs
- [ ] Environment variables
- [ ] Port mappings

### 2. Add Optional Services
- [ ] Prometheus for metrics
- [ ] Grafana for visualization
- [ ] LocalStack for AWS S3 simulation (optional)

### 3. Create Production Compose File
- [ ] Production-like configuration
- [ ] No volume mounts for code
- [ ] Production environment variables
- [ ] Health checks enabled

### 4. Documentation
- [ ] Usage instructions
- [ ] Service descriptions
- [ ] Environment variable reference

---

## 📂 Files to Create

```
prevcarga/
├── docker-compose.yml          # Development compose
├── docker-compose.prod.yml     # Production-like compose
├── docker/
│   └── prometheus.yml          # Prometheus configuration
└── .env.example                # Environment variable template
```

---

## 🔧 Technical Implementation

### docker-compose.yml (Development)

```yaml
version: '3.8'

services:
  prevcarga:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    container_name: prevcarga-dev
    ports:
      - "8000:8000"
    environment:
      - PREVCARGA_ENV=development
      - PREVCARGA_LOG_LEVEL=DEBUG
      - AWS_PROFILE=${AWS_PROFILE:-default}
      - AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
      - PYTHONUNBUFFERED=1
    volumes:
      # Mount source code for live reload
      - ./src:/app/src:rw
      - ./config:/app/config:ro
      - ./logs:/app/logs:rw
      - ~/.aws:/home/prevcarga/.aws:ro  # AWS credentials
    networks:
      - prevcarga-network
    command: prevcarga --help
    restart: unless-stopped
  
  # Optional: Prometheus for metrics collection
  prometheus:
    image: prom/prometheus:latest
    container_name: prevcarga-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - prevcarga-network
    restart: unless-stopped
    profiles:
      - monitoring
  
  # Optional: Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: prevcarga-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - prevcarga-network
    restart: unless-stopped
    profiles:
      - monitoring
  
  # Optional: LocalStack for AWS S3 simulation
  localstack:
    image: localstack/localstack:latest
    container_name: prevcarga-localstack
    ports:
      - "4566:4566"
    environment:
      - SERVICES=s3
      - DEBUG=1
      - DATA_DIR=/tmp/localstack/data
    volumes:
      - localstack-data:/tmp/localstack
    networks:
      - prevcarga-network
    restart: unless-stopped
    profiles:
      - aws-local

networks:
  prevcarga-network:
    driver: bridge
    name: prevcarga-net

volumes:
  prometheus-data:
    name: prevcarga-prometheus-data
  grafana-data:
    name: prevcarga-grafana-data
  localstack-data:
    name: prevcarga-localstack-data
```

### docker-compose.prod.yml (Production-like)

```yaml
version: '3.8'

services:
  prevcarga:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    container_name: prevcarga-prod
    ports:
      - "8000:8000"
    environment:
      - PREVCARGA_ENV=production
      - PREVCARGA_LOG_LEVEL=INFO
      - AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
    volumes:
      # Only mount config and logs (no source code)
      - ./config:/app/config:ro
      - ./logs:/app/logs:rw
    networks:
      - prevcarga-network
    healthcheck:
      test: ["/healthcheck.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: always
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

networks:
  prevcarga-network:
    driver: bridge
    name: prevcarga-prod-net
```

### .env.example

```bash
# PrevCarga Environment Variables

# Environment
PREVCARGA_ENV=development
PREVCARGA_LOG_LEVEL=INFO

# AWS Configuration
AWS_PROFILE=default
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# S3 Storage
PREVCARGA_S3_BUCKET=prevcarga-data-production
PREVCARGA_MODELS_BUCKET=prevcarga-models-production

# Application
PREVCARGA_PARALLEL_WORKERS=4
PREVCARGA_CACHE_DIR=/tmp/prevcarga_cache
```

### Prometheus Configuration (`docker/prometheus.yml`)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prevcarga'
    static_configs:
      - targets: ['prevcarga:8000']
    metrics_path: '/metrics'
```

---

## 🧪 Testing & Validation

### Validation Commands

```bash
# Start development environment
docker-compose up -d

# View logs
docker-compose logs -f prevcarga

# Check service status
docker-compose ps

# Run a command in the container
docker-compose exec prevcarga prevcarga --version

# Start with monitoring
docker-compose --profile monitoring up -d

# Start with local AWS
docker-compose --profile aws-local up -d

# Stop all services
docker-compose down

# Start production-like environment
docker-compose -f docker-compose.prod.yml up -d

# Clean up everything
docker-compose down -v
```

### Test Live Reload

```bash
# Start dev environment
docker-compose up -d

# Make a change to source code
echo "# Test change" >> src/prevcarga/__init__.py

# Verify change is reflected (depends on implementation)
docker-compose exec prevcarga cat /app/src/prevcarga/__init__.py
```

---

## 📝 Technical Notes

- Use `profiles` for optional services
- Volume mounts enable live code changes in development
- Read-only mounts for config files prevent accidental modification
- Bridge network allows inter-service communication
- Named volumes persist data across container restarts
- Resource limits in production compose prevent resource exhaustion

---

## 🔗 Dependencies

**Depends On:**
- PC-100-11A: Production Dockerfile

**Blocks:**
- Local development and testing

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] docker-compose.yml created and tested
- [ ] docker-compose.prod.yml created and tested
- [ ] Services start successfully
- [ ] Volume mounts work correctly
- [ ] Network communication verified
- [ ] Documentation updated
- [ ] Technical review approved
- [ ] Committed to repository

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-19 | 1.0.0 | Initial ticket created from Epic-11A | System |

---

**Next Ticket:** [PC-102-11A: Container Security and Optimization](PC-102-11A-container-security-optimization.md)
