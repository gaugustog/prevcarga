# PC-096-11A: README and Installation Guide

**Ticket ID:** PC-096-11A  
**Epic:** [Epic-11A: Documentation & Infrastructure](../epics/Epic-11A.md)  
**User Story:** US-11A.1 - Comprehensive Documentation Suite  
**Story Points:** 5  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Create comprehensive README.md and detailed installation guide covering project overview, quick start, and installation procedures for various environments (development, Docker, AWS deployment).

**As a** new user or developer  
**I want** clear installation instructions and project overview  
**So that** I can quickly understand the project and get it running in my environment

---

## ✅ Acceptance Criteria

- [ ] README.md created with complete project overview
- [ ] Quick start guide with minimal steps to get running
- [ ] Detailed installation guide for multiple environments
- [ ] Prerequisites documented (Python version, AWS credentials, etc.)
- [ ] Development environment setup instructions
- [ ] Docker-based installation instructions
- [ ] AWS deployment prerequisites documented
- [ ] Troubleshooting section for common installation issues
- [ ] Links to detailed documentation sections
- [ ] Badges for build status, coverage, and version

---

## 🔧 Implementation Tasks

### 1. Create Root README.md
- [ ] Add project title and tagline
- [ ] Add project badges (build status, coverage, version, license)
- [ ] Write project overview (2-3 paragraphs)
- [ ] List key features and capabilities
- [ ] Add quick start section (3-5 steps)
- [ ] Add table of contents
- [ ] Link to detailed documentation
- [ ] Add contributing guidelines section
- [ ] Add license and support information
- [ ] Add acknowledgments and credits

### 2. Create Installation Guide
- [ ] Create `docs/source/guides/installation.md`
- [ ] Document system prerequisites
- [ ] Python version requirements (3.11+)
- [ ] System dependencies (build tools, etc.)
- [ ] AWS credentials setup
- [ ] Write development environment installation
- [ ] Write Docker installation instructions
- [ ] Write production deployment prerequisites
- [ ] Add environment variable configuration
- [ ] Document optional dependencies

### 3. Add Quick Start Examples
- [ ] Basic prediction example
- [ ] Model training example
- [ ] Configuration example
- [ ] Data loading example
- [ ] CLI usage examples

### 4. Create Environment-Specific Guides
- [ ] Local development with uv
- [ ] Virtual environment setup
- [ ] Docker Compose development
- [ ] AWS credentials configuration
- [ ] S3 bucket access setup

### 5. Add Troubleshooting Section
- [ ] Common installation errors
- [ ] Python version issues
- [ ] AWS credentials problems
- [ ] Dependency conflicts
- [ ] Permission issues
- [ ] Network connectivity problems

---

## 📂 Files to Create/Modify

```
prevcarga/
├── README.md                           # Root README (to create)
└── docs/source/guides/
    ├── installation.md                 # Detailed installation guide
    ├── quick-start.md                  # Quick start tutorial
    └── troubleshooting.md              # Troubleshooting guide
```

---

## 🔧 Technical Implementation

### Root README.md Template

```markdown
# 🔌 PrevCarga - Unified Electric Load Forecasting System

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-85%25-green)]()
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

> Advanced machine learning system for short-term and medium-term electric load forecasting with hierarchical prediction strategies.

---

## 📖 Overview

PrevCarga is a production-ready forecasting system that provides accurate electric load predictions using:

- **Multiple ML Models**: LightGBM, Random Forest, Hierarchical models (REGDIN-SVM, Holt-Winters)
- **Advanced Feature Engineering**: Temporal, calendar, lag, cyclical, and signal processing features
- **Hierarchical Forecasting**: Bottom-up reconciliation with sophisticated aggregation
- **Production-Ready**: Docker containerization, AWS integration, comprehensive monitoring
- **Extensible Architecture**: Plugin-based system for easy model and feature additions

### Key Features

✨ **Multiple Forecasting Models**
- Base ML models (LightGBM, Random Forest)
- Hierarchical strategies (ARIMA+SVM, Holt-Winters)
- Ensemble combinations with intelligent selection

⚙️ **Rich Feature Engineering**
- Automated temporal and calendar features
- Lag and rolling window computations
- Signal processing (LOESS, Wavelets)
- Cyclical encoding for periodic patterns

🎯 **Production-Ready**
- Comprehensive CLI interface
- Docker containerization
- AWS S3 integration
- Structured logging and monitoring
- Health checks and validation

🔌 **Extensible Architecture**
- Plugin system for features and models
- Configuration-driven pipelines
- Easy integration of new strategies

---

## 🚀 Quick Start

Get started with PrevCarga in 5 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/your-org/prevcarga.git
cd prevcarga

# 2. Install with uv (recommended)
pip install uv
uv sync

# 3. Configure AWS credentials
export AWS_PROFILE=your-profile
export AWS_DEFAULT_REGION=us-east-1

# 4. Run a prediction
prevcarga predict --model lgbm --area area001 --date 2024-01-01

# 5. Train a model
prevcarga train model --model lgbm --area area001 \
    --start-date 2023-01-01 --end-date 2023-12-31
```

---

## 📦 Installation

### Prerequisites

- **Python**: 3.11 or higher
- **AWS Account**: For S3 data access
- **System**: Linux, macOS, or Windows (with WSL recommended)
- **Memory**: Minimum 8GB RAM (16GB recommended for training)

### Development Installation

```bash
# Install uv package manager
pip install uv

# Clone and install
git clone https://github.com/your-org/prevcarga.git
cd prevcarga
uv sync

# Verify installation
prevcarga --version
```

### Docker Installation

```bash
# Build Docker image
docker build -t prevcarga:latest .

# Run container
docker run -p 8000:8000 \
    -e AWS_PROFILE=your-profile \
    prevcarga:latest
```

📚 **Detailed Installation Guide**: See [Installation Documentation](docs/source/guides/installation.md)

---

## 📖 Documentation

- **[Installation Guide](docs/source/guides/installation.md)**: Detailed setup instructions
- **[Usage Guide](docs/source/guides/usage.md)**: CLI commands and examples
- **[API Reference](docs/source/api/index.rst)**: Complete API documentation
- **[Architecture](docs/source/architecture/overview.md)**: System design and decisions
- **[Extension Guide](docs/source/guides/extensions.md)**: Adding plugins and models

---

## 💻 Usage Examples

### Generate Predictions

```bash
# Single model prediction
prevcarga predict --model lgbm --area area001 --date 2024-01-01

# Ensemble prediction with multiple models
prevcarga predict --model all --area area001 --date 2024-01-01

# Batch prediction for multiple areas
prevcarga predict batch --config predictions.yaml
```

### Train Models

```bash
# Train single model
prevcarga train model --model lgbm --area area001 \
    --start-date 2023-01-01 --end-date 2023-12-31

# Batch training across areas
prevcarga train batch --config training.yaml --parallel 4
```

### Data Management

```bash
# Validate data quality
prevcarga data validate --area area001 --date 2024-01-01

# Generate feature engineering pipeline
prevcarga feature build --area area001 --start-date 2024-01-01
```

---

## 🏗️ Project Structure

```
prevcarga/
├── src/prevcarga/          # Main source code
│   ├── data/               # Data loading and validation
│   ├── features/           # Feature engineering plugins
│   ├── models/             # ML model implementations
│   ├── hierarchical/       # Hierarchical forecasting
│   ├── ensemble/           # Model combination strategies
│   ├── workflows/          # Training and prediction workflows
│   └── cli/                # Command-line interface
├── config/                 # Configuration files
├── docs/                   # Documentation
├── tests/                  # Test suite
└── scripts/                # Utility scripts
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=prevcarga --cov-report=html

# Run specific test suite
pytest tests/models/
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development workflow

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Support

- **Documentation**: [https://prevcarga.readthedocs.io](https://prevcarga.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-org/prevcarga/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/prevcarga/discussions)

---

## 🙏 Acknowledgments

- LightGBM and scikit-learn teams for excellent ML libraries
- statsmodels team for time series modeling
- The open-source community for inspiration and support

---

**Built with ❤️ by the PrevCarga Team**
```

### Installation Guide (`docs/source/guides/installation.md`)

```markdown
# Installation Guide

Complete installation instructions for PrevCarga across different environments.

## Table of Contents

- [System Requirements](#system-requirements)
- [Development Installation](#development-installation)
- [Docker Installation](#docker-installation)
- [AWS Configuration](#aws-configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Requirements

- **CPU**: 4+ cores recommended for parallel training
- **RAM**: 
  - Minimum: 8GB
  - Recommended: 16GB
  - Production: 32GB+
- **Storage**: 
  - Minimum: 10GB free space
  - Recommended: 50GB+ for data and models

### Software Requirements

- **Operating System**: 
  - Linux (Ubuntu 20.04+, CentOS 8+)
  - macOS (12.0+)
  - Windows (with WSL2)
- **Python**: 3.11 or higher
- **Git**: 2.30+
- **AWS CLI**: 2.0+ (for S3 access)

---

## Development Installation

### Step 1: Install System Dependencies

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3.11-dev \
    build-essential \
    git \
    curl
```

#### macOS

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 git
```

#### Windows (WSL2)

```bash
# Install WSL2 with Ubuntu
wsl --install -d Ubuntu-22.04

# Inside WSL, run Ubuntu commands above
```

### Step 2: Install uv Package Manager

```bash
# Install uv (fast Python package manager)
pip install uv

# Verify installation
uv --version
```

### Step 3: Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-org/prevcarga.git
cd prevcarga

# Checkout desired branch (optional)
git checkout main
```

### Step 4: Install Python Dependencies

```bash
# Install all dependencies with uv
uv sync

# This creates a virtual environment and installs:
# - All required packages
# - Development dependencies
# - Testing tools
```

### Step 5: Install PrevCarga

```bash
# Install in development mode
uv pip install -e .

# Verify installation
prevcarga --version
prevcarga --help
```

---

## Docker Installation

### Step 1: Install Docker

#### Ubuntu

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### macOS

```bash
# Download and install Docker Desktop
# Visit: https://www.docker.com/products/docker-desktop
```

### Step 2: Build Docker Image

```bash
# Build production image
docker build -t prevcarga:latest .

# Verify image
docker images | grep prevcarga
```

### Step 3: Run Container

```bash
# Run with default settings
docker run -p 8000:8000 prevcarga:latest

# Run with AWS credentials
docker run -p 8000:8000 \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_DEFAULT_REGION=us-east-1 \
    prevcarga:latest

# Run with volume mounts
docker run -p 8000:8000 \
    -v $(pwd)/config:/app/config \
    -v $(pwd)/logs:/app/logs \
    prevcarga:latest
```

### Step 4: Docker Compose (Development)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f prevcarga

# Stop services
docker-compose down
```

---

## AWS Configuration

### Step 1: Create AWS Account and IAM User

1. Sign in to AWS Console
2. Navigate to IAM → Users → Create User
3. Attach policies: `AmazonS3ReadOnlyAccess` or custom policy
4. Generate access keys

### Step 2: Configure AWS CLI

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., us-east-1)
# - Default output format (json)
```

### Step 3: Verify S3 Access

```bash
# List S3 buckets
aws s3 ls

# Test access to PrevCarga buckets
aws s3 ls s3://prevcarga-data-production/
```

### Step 4: Set Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc
export AWS_PROFILE=prevcarga
export AWS_DEFAULT_REGION=us-east-1
export PREVCARGA_S3_BUCKET=prevcarga-data-production

# Reload shell configuration
source ~/.bashrc
```

---

## Verification

### Verify Installation

```bash
# Check version
prevcarga --version

# Check help
prevcarga --help

# Check configuration
prevcarga config show

# Run system check
prevcarga status
```

### Run Sample Prediction

```bash
# Quick prediction test
prevcarga predict \
    --model lgbm \
    --area area001 \
    --date 2024-01-01 \
    --dry-run
```

### Run Tests

```bash
# Install test dependencies
uv sync --group dev

# Run test suite
pytest

# Run with coverage
pytest --cov=prevcarga
```

---

## Troubleshooting

### Python Version Issues

**Problem**: Wrong Python version

```bash
# Check Python version
python --version

# Install specific version with pyenv
curl https://pyenv.run | bash
pyenv install 3.11.7
pyenv global 3.11.7
```

### AWS Credentials Issues

**Problem**: Unable to access S3

```bash
# Check credentials
aws sts get-caller-identity

# Test bucket access
aws s3 ls s3://prevcarga-data-production/

# Verify IAM permissions
aws iam get-user
```

### Dependency Installation Issues

**Problem**: Package installation fails

```bash
# Clear cache
uv cache clean

# Reinstall dependencies
uv sync --reinstall

# Install with verbose output
uv sync -v
```

### Docker Issues

**Problem**: Container fails to start

```bash
# Check logs
docker logs <container-id>

# Check container status
docker ps -a

# Rebuild image
docker build --no-cache -t prevcarga:latest .
```

### Memory Issues

**Problem**: Out of memory during training

```bash
# Reduce parallel workers
prevcarga train model --parallel 1

# Use smaller batch size
prevcarga train model --batch-size 1000

# Monitor memory usage
htop  # or top
```

---

## Next Steps

- Review [Usage Guide](usage.md) for CLI commands
- Explore [API Reference](../api/index.rst) for programmatic usage
- Read [Architecture Documentation](../architecture/overview.md) for system design
- Check [Extension Guide](extensions.md) for customization

---

**Need Help?** [Open an issue](https://github.com/your-org/prevcarga/issues)
```

---

## 🧪 Testing & Validation

### Validation Commands
```bash
# Verify README renders correctly on GitHub
# (Use GitHub's Markdown preview or grip tool)
grip README.md

# Check for broken links
markdown-link-check README.md docs/source/guides/installation.md

# Verify installation instructions
# Create clean test environment
python -m venv test_env
source test_env/bin/activate
# Follow installation guide step-by-step
```

### Success Criteria
- [ ] README.md renders correctly with all badges
- [ ] All links work (no 404 errors)
- [ ] Installation guide can be followed by new user
- [ ] Quick start commands execute successfully
- [ ] Documentation links point to correct locations
- [ ] Code examples are syntactically correct

---

## 📝 Technical Notes

- Use GitHub-flavored Markdown for README.md
- Use MyST Markdown for Sphinx documentation
- Include badges from shields.io or similar
- Keep quick start under 5 commands
- Use code blocks with syntax highlighting
- Include visual hierarchy with emojis (sparingly)
- Test installation guide on clean VM/container

---

## 🔗 Dependencies

**Depends On:**
- PC-095-11A: Documentation Structure Setup

**Blocks:**
- PC-097-11A: Usage Guide and Examples
- All other documentation tickets (provides entry point)

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] All implementation tasks completed
- [ ] README.md created and reviewed
- [ ] Installation guide tested on clean environment
- [ ] Quick start verified to work
- [ ] All links validated
- [ ] Markdown linting passed
- [ ] Technical review approved
- [ ] Committed to repository

---

## 🔄 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-11-19 | 1.0.0 | Initial ticket created from Epic-11A | System |

---

**Next Ticket:** [PC-097-11A: Usage Guide and Examples](PC-097-11A-usage-guide-examples.md)
