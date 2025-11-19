# PC-102-11A: Container Security and Optimization

**Ticket ID:** PC-102-11A  
**Epic:** [Epic-11A: Documentation & Infrastructure](../epics/Epic-11A.md)  
**User Story:** US-11A.2 - Production Docker Infrastructure  
**Story Points:** 5  
**Priority:** High  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement container security scanning, vulnerability remediation, image optimization, and establish CI/CD pipeline for automated security checks and image building.

**As a** security engineer  
**I want** secure and optimized container images  
**So that** production deployments are safe and efficient

---

## ✅ Acceptance Criteria

- [ ] Container security scanning integrated (Trivy/Snyk)
- [ ] No high-severity vulnerabilities in final image
- [ ] Image size < 2GB (optimized)
- [ ] CI/CD pipeline for automated builds
- [ ] Automated security scanning in CI
- [ ] Image layer optimization completed
- [ ] Security best practices documented
- [ ] Vulnerability remediation process documented
- [ ] Container signing/verification (optional)

---

## 🔧 Implementation Tasks

### 1. Security Scanning Setup
- [ ] Integrate Trivy scanner
- [ ] Integrate Snyk scanner (optional)
- [ ] Configure scanning in CI/CD
- [ ] Set up vulnerability reporting
- [ ] Define severity thresholds

### 2. Vulnerability Remediation
- [ ] Scan base image for vulnerabilities
- [ ] Update dependencies to secure versions
- [ ] Remove unnecessary packages
- [ ] Apply security patches
- [ ] Document remediation actions

### 3. Image Optimization
- [ ] Analyze layer sizes
- [ ] Optimize layer ordering
- [ ] Remove build artifacts
- [ ] Compress image layers
- [ ] Verify size reduction

### 4. CI/CD Pipeline
- [ ] Create GitHub Actions workflow
- [ ] Automated build on commit
- [ ] Automated security scanning
- [ ] Image tagging strategy
- [ ] Registry push automation

### 5. Security Best Practices
- [ ] Document security guidelines
- [ ] Create security checklist
- [ ] Establish update schedule
- [ ] Define incident response

---

## 📂 Files to Create

```
prevcarga/
├── .github/
│   └── workflows/
│       ├── docker-build.yml
│       └── security-scan.yml
├── docker/
│   └── trivy.yaml
└── docs/
    └── security/
        ├── container-security.md
        └── vulnerability-response.md
```

---

## 🔧 Technical Implementation

### GitHub Actions - Docker Build (`.github/workflows/docker-build.yml`)

```yaml
name: Docker Build and Push

on:
  push:
    branches: [main, develop]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      security-events: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ steps.meta.outputs.version }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Check vulnerability threshold
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ steps.meta.outputs.version }}
          exit-code: '1'
          severity: 'CRITICAL,HIGH'
          ignore-unfixed: true
```

### Security Scan Workflow (`.github/workflows/security-scan.yml`)

```yaml
name: Security Scan

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  scan-image:
    runs-on: ubuntu-latest
    permissions:
      security-events: write

    steps:
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'ghcr.io/${{ github.repository }}:latest'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Generate vulnerability report
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'ghcr.io/${{ github.repository }}:latest'
          format: 'table'
          output: 'vulnerability-report.txt'

      - name: Upload report artifact
        uses: actions/upload-artifact@v4
        with:
          name: vulnerability-report
          path: vulnerability-report.txt
```

### Trivy Configuration (`docker/trivy.yaml`)

```yaml
# Trivy configuration file

severity:
  - CRITICAL
  - HIGH
  - MEDIUM

vulnerability:
  type:
    - os
    - library

ignore-unfixed: false

secret:
  config: trivy-secret.yaml

output: table

exit-code: 1
```

### Security Documentation (`docs/security/container-security.md`)

```markdown
# Container Security

## Security Scanning

### Trivy Scanner

Run security scan locally:

```bash
# Scan image for vulnerabilities
trivy image prevcarga:latest

# Scan with specific severity
trivy image --severity HIGH,CRITICAL prevcarga:latest

# Generate JSON report
trivy image --format json --output report.json prevcarga:latest

# Scan for secrets
trivy image --scanners secret prevcarga:latest
```

### Snyk Scanner

```bash
# Authenticate
snyk auth

# Scan image
snyk container test prevcarga:latest

# Monitor image
snyk container monitor prevcarga:latest
```

## Security Best Practices

### 1. Use Minimal Base Images
- ✅ Use `python:3.11-slim` instead of full image
- ✅ Remove unnecessary packages
- ✅ Use multi-stage builds

### 2. Run as Non-Root User
- ✅ Create dedicated user account
- ✅ Set appropriate file permissions
- ✅ Use USER directive in Dockerfile

### 3. Keep Dependencies Updated
- 🔄 Regular dependency updates
- 🔄 Monitor security advisories
- 🔄 Automated scanning in CI/CD

### 4. Scan for Vulnerabilities
- ✅ Pre-deployment scanning
- ✅ Scheduled security scans
- ✅ Fail builds on high-severity issues

### 5. Minimize Attack Surface
- ✅ Remove build tools from production image
- ✅ No secrets in image layers
- ✅ Read-only filesystem where possible

### 6. Image Signing
- ⏳ Sign images with Cosign
- ⏳ Verify signatures before deployment
- ⏳ Maintain signing key security

## Vulnerability Response

### High-Severity Vulnerabilities

1. **Identify**: Automated scanning detects issue
2. **Assess**: Evaluate impact and exploitability
3. **Remediate**: Update dependencies or apply patches
4. **Test**: Verify fix doesn't break functionality
5. **Deploy**: Push updated image to production
6. **Document**: Record issue and resolution

### Update Schedule

- **Critical**: Immediate response (<24 hours)
- **High**: Within 1 week
- **Medium**: Within 1 month
- **Low**: Quarterly review

## Compliance

### CIS Docker Benchmark

Key compliance areas:
- Image provenance and integrity
- Container runtime security
- Network isolation
- Resource limits
- Logging and monitoring
```

---

## 🧪 Testing & Validation

### Manual Security Scan

```bash
# Build image
docker build -t prevcarga:latest .

# Scan with Trivy
trivy image prevcarga:latest

# Scan with Snyk
snyk container test prevcarga:latest

# Check image size
docker images prevcarga:latest

# Analyze layers
dive prevcarga:latest

# Verify non-root user
docker run --rm prevcarga:latest whoami

# Check for secrets
trivy image --scanners secret prevcarga:latest
```

### Optimization Verification

```bash
# Before optimization
docker build -t prevcarga:before .
docker images prevcarga:before

# After optimization
docker build -t prevcarga:after .
docker images prevcarga:after

# Compare sizes
docker images | grep prevcarga
```

---

## 📝 Technical Notes

- Trivy scans for OS and library vulnerabilities
- Snyk provides detailed remediation advice
- GitHub Container Registry supports vulnerability scanning
- Image signing with Cosign adds supply chain security
- Regular automated scans catch new vulnerabilities
- CI/CD integration prevents vulnerable images from deployment

---

## 🔗 Dependencies

**Depends On:**
- PC-100-11A: Production Dockerfile

**Blocks:**
- PC-104-11A: Deployment and Validation

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] Security scanning integrated in CI/CD
- [ ] No high-severity vulnerabilities
- [ ] Image size < 2GB
- [ ] Automated builds working
- [ ] Documentation complete
- [ ] Security team review approved
- [ ] Committed to repository

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-19 | 1.0.0 | Initial ticket created from Epic-11A | System |

---

**Next Ticket:** [PC-103-11A: AWS Infrastructure as Code](PC-103-11A-aws-infrastructure-iac.md)
