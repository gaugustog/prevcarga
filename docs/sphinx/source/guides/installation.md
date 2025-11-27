# Installation Guide

Complete installation instructions for PrevCarga across different environments.

## Table of Contents

- [System Requirements](#system-requirements)
- [Development Installation](#development-installation)
- [Docker Installation](#docker-installation)
- [AWS Configuration](#aws-configuration)
- [Verification](#verification)
- [Optional Dependencies](#optional-dependencies)

---

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended | Production |
|-----------|---------|-------------|------------|
| **CPU** | 2 cores | 4+ cores | 8+ cores |
| **RAM** | 8GB | 16GB | 32GB+ |
| **Storage** | 10GB free | 50GB | 100GB+ |

```{note}
Training large models or running parallel backtests requires significantly more resources.
For multi-area parallel training, allocate at least 2GB RAM per concurrent training job.
```

### Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| **Python** | 3.11+ | Python 3.12 also supported |
| **Git** | 2.30+ | For cloning repository |
| **AWS CLI** | 2.0+ | For S3 backend access |
| **Docker** | 20.10+ | Optional, for containerized deployment |

### Supported Operating Systems

- **Linux**: Ubuntu 20.04+, CentOS 8+, Debian 11+
- **macOS**: 12.0 (Monterey) or higher
- **Windows**: Windows 10+ with WSL2 (native Windows not recommended)

---

## Development Installation

### Step 1: Install System Dependencies

#### Ubuntu/Debian

```bash
# Update package lists
sudo apt-get update

# Install required packages
sudo apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    build-essential \
    git \
    curl \
    libffi-dev \
    libssl-dev
```

#### macOS

```bash
# Install Homebrew (if not present)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python@3.11 git

# Add Python 3.11 to PATH
echo 'export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### Windows (WSL2)

```powershell
# Install WSL2 (PowerShell as Administrator)
wsl --install -d Ubuntu-22.04

# Restart your computer, then open Ubuntu terminal
# Follow Ubuntu/Debian instructions above
```

### Step 2: Install uv Package Manager

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver, recommended for PrevCarga development.

```bash
# Install uv via pip
pip install uv

# Or via curl (standalone installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### Step 3: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/gaugustog/prevcarga.git
cd prevcarga

# Optional: checkout a specific version
git checkout v1.0.0

# Or stay on main for latest development
git checkout main
```

### Step 4: Install Dependencies

```bash
# Install all dependencies (creates .venv automatically)
uv sync

# This installs:
# - Core dependencies
# - Development tools (pytest, black, ruff, mypy)
# - Documentation dependencies
```

```{tip}
Use `uv sync --no-dev` to install only production dependencies without development tools.
```

### Step 5: Install PrevCarga

```bash
# Install in development mode
uv pip install -e .

# Verify installation
prevcarga --version
prevcarga --help
```

### Step 6: Activate the Virtual Environment

The virtual environment is created automatically by `uv sync`. Activate it for development:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows (WSL)
source .venv/bin/activate

# Verify activation
which python  # Should show .venv/bin/python
```

---

## Docker Installation

### Step 1: Install Docker

#### Ubuntu

```bash
# Install Docker via convenience script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add current user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker run hello-world
```

#### macOS

Download and install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop).

After installation:
```bash
# Verify installation
docker --version
docker run hello-world
```

#### Windows

Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop).

Ensure WSL2 backend is enabled in Docker Desktop settings.

### Step 2: Build Docker Image

```bash
# Build production image
docker build -t prevcarga:latest .

# Build with specific tag
docker build -t prevcarga:v1.0.0 .

# Verify image
docker images | grep prevcarga
```

### Step 3: Run Container

#### Basic Run

```bash
# Run with default settings
docker run --rm prevcarga:latest --help
```

#### With AWS Credentials

```bash
# Using environment variables
docker run --rm \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e AWS_DEFAULT_REGION=us-east-1 \
    prevcarga:latest predict --config config.yaml

# Using AWS credentials file
docker run --rm \
    -v ~/.aws:/root/.aws:ro \
    prevcarga:latest predict --config config.yaml
```

#### With Volume Mounts

```bash
# Mount config and data directories
docker run --rm \
    -v $(pwd)/config:/app/config:ro \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    prevcarga:latest train --config /app/config/config.yaml
```

### Step 4: Docker Compose (Development)

Create or use the existing `docker-compose.yml`:

```yaml
version: '3.8'

services:
  prevcarga:
    build: .
    image: prevcarga:latest
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
      - AWS_DEFAULT_REGION=us-east-1
      - PREVCARGA_LOG_LEVEL=INFO
    command: ["predict", "--config", "/app/config/config.yaml"]
```

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f prevcarga

# Stop services
docker-compose down
```

---

## AWS Configuration

PrevCarga uses AWS S3 for production data storage. Follow these steps to configure AWS access.

### Step 1: Install AWS CLI

```bash
# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify installation
aws --version
```

### Step 2: Configure AWS Credentials

#### Option A: AWS Configure (Recommended for Development)

```bash
# Interactive configuration
aws configure

# Enter when prompted:
# AWS Access Key ID: your-access-key-id
# AWS Secret Access Key: your-secret-access-key
# Default region name: us-east-1
# Default output format: json
```

#### Option B: Environment Variables

```bash
# Add to ~/.bashrc or ~/.zshrc
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="us-east-1"

# Reload shell configuration
source ~/.bashrc
```

#### Option C: AWS Profiles (Multiple Accounts)

```bash
# Configure named profile
aws configure --profile prevcarga

# Use profile
export AWS_PROFILE=prevcarga

# Or specify per command
aws s3 ls --profile prevcarga
```

### Step 3: IAM Permissions

Ensure your IAM user/role has the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::your-prevcarga-bucket",
                "arn:aws:s3:::your-prevcarga-bucket/*"
            ]
        }
    ]
}
```

### Step 4: Verify S3 Access

```bash
# Check identity
aws sts get-caller-identity

# List buckets
aws s3 ls

# Test access to PrevCarga bucket
aws s3 ls s3://your-prevcarga-bucket/
```

### Step 5: Configure PrevCarga for S3

Update your configuration file:

```yaml
# config/config.yaml
storage:
  backend: s3
  s3:
    bucket: your-prevcarga-bucket
    region: us-east-1
    prefix: prevcarga/  # Optional prefix for all paths
```

---

## Verification

### Verify Python Installation

```bash
# Check Python version
python --version  # Should be 3.11+

# Check pip
pip --version
```

### Verify PrevCarga Installation

```bash
# Check version
prevcarga --version

# View available commands
prevcarga --help

# Check configuration
prevcarga config show
```

### Run Smoke Test

```bash
# Verify imports work
python -c "from src.models import LightGBMModel; print('Models OK')"
python -c "from src.features import FeatureRegistry; print('Features OK')"
python -c "from src.storage import StorageFactory; print('Storage OK')"

# Run unit tests
pytest tests/unit/ -v --tb=short
```

### Test Prediction (Dry Run)

```bash
# Dry run prediction (no actual data needed)
prevcarga predict \
    --config config/config_dev.yaml \
    --date 2024-01-01 \
    --areas SECO_RJ \
    --dry-run
```

---

## Optional Dependencies

### GPU Support (CUDA)

For GPU-accelerated training with LightGBM:

```bash
# Install CUDA toolkit (Ubuntu)
sudo apt-get install -y nvidia-cuda-toolkit

# Reinstall LightGBM with GPU support
pip uninstall lightgbm
pip install lightgbm --install-option=--gpu

# Verify GPU support
python -c "import lightgbm; print(lightgbm.LGBMClassifier().get_params())"
```

### Development Tools

```bash
# Install development dependencies
uv sync --group dev

# This includes:
# - pytest (testing)
# - black (formatting)
# - ruff (linting)
# - mypy (type checking)
# - pre-commit (git hooks)
```

### Documentation Tools

```bash
# Install documentation dependencies
uv sync --group docs

# Build documentation locally
cd docs/sphinx
make html

# View documentation
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
```

---

## Next Steps

After successful installation:

1. **[Quick Start Guide](quick-start.md)**: Run your first prediction in minutes
2. **[API Reference](../api/index.rst)**: For programmatic usage
3. **[Tutorials](../tutorials/index.rst)**: Step-by-step tutorials
4. **[Architecture Overview](../architecture/index.rst)**: Understand the system design

---

## Getting Help

- **[Troubleshooting Guide](troubleshooting.md)**: Common issues and solutions
- **[GitHub Issues](https://github.com/gaugustog/prevcarga/issues)**: Report bugs or request features
- **[GitHub Discussions](https://github.com/gaugustog/prevcarga/discussions)**: Ask questions

---

**Need help?** Open an issue on [GitHub](https://github.com/gaugustog/prevcarga/issues)
