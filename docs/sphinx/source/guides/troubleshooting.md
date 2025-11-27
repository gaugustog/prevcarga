# Troubleshooting Guide

Solutions for common issues when installing and using PrevCarga.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Issues](#configuration-issues)
- [AWS and S3 Issues](#aws-and-s3-issues)
- [Runtime Issues](#runtime-issues)
- [Performance Issues](#performance-issues)
- [Docker Issues](#docker-issues)
- [Getting Help](#getting-help)

---

## Installation Issues

### Python Version Errors

**Problem**: PrevCarga requires Python 3.11+, but wrong version is detected.

```bash
# Error message:
# ERROR: This project requires Python >=3.11
```

**Solution**:

```bash
# Check current Python version
python --version

# If wrong version, install Python 3.11
# Ubuntu/Debian
sudo apt-get install python3.11 python3.11-venv

# macOS
brew install python@3.11

# Use pyenv for version management
curl https://pyenv.run | bash
pyenv install 3.11.7
pyenv global 3.11.7

# Verify
python --version  # Should show 3.11.x
```

### uv Installation Fails

**Problem**: Cannot install or run uv package manager.

**Solution**:

```bash
# Try alternative installation methods
# Method 1: pip
pip install uv

# Method 2: curl (standalone)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Method 3: pipx
pipx install uv

# Add to PATH if needed
export PATH="$HOME/.local/bin:$PATH"

# Verify
uv --version
```

### Dependency Installation Fails

**Problem**: `uv sync` or `pip install` fails with dependency errors.

```bash
# Error examples:
# ERROR: Could not find a version that satisfies the requirement...
# ERROR: ResolutionImpossible...
```

**Solutions**:

```bash
# Clear uv cache
uv cache clean

# Reinstall with verbose output
uv sync -v

# Try with fresh virtual environment
rm -rf .venv
uv sync

# If specific package fails, install separately
uv pip install problematic-package
uv sync
```

### Build Tools Missing

**Problem**: Compilation fails for packages with C extensions.

```bash
# Error message:
# error: Microsoft Visual C++ 14.0 or greater is required
# OR
# error: command 'gcc' failed
```

**Solution**:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3.11-dev libffi-dev libssl-dev

# macOS
xcode-select --install
brew install openssl

# Windows (WSL recommended)
# Use WSL2 with Ubuntu instead of native Windows
```

### Import Errors After Installation

**Problem**: `ModuleNotFoundError` when importing PrevCarga modules.

```bash
# Error:
# ModuleNotFoundError: No module named 'src'
```

**Solution**:

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Verify you're in project root
pwd  # Should be in prevcarga/

# Reinstall in development mode
uv pip install -e .

# Verify installation
python -c "import src; print('OK')"
```

---

## Configuration Issues

### Configuration File Not Found

**Problem**: PrevCarga cannot find the configuration file.

```bash
# Error:
# FileNotFoundError: Configuration file not found: config.yaml
```

**Solution**:

```bash
# Check if file exists
ls -la config/

# Specify full path
prevcarga predict --config /full/path/to/config.yaml

# Create default config
prevcarga config create --output config/config.yaml
```

### Invalid Configuration Format

**Problem**: YAML parsing errors in configuration file.

```bash
# Error:
# yaml.scanner.ScannerError: mapping values are not allowed here
```

**Solution**:

```yaml
# Check YAML syntax - common issues:

# WRONG: Missing space after colon
storage:backend: s3

# CORRECT: Space after colon
storage:
  backend: s3

# WRONG: Inconsistent indentation
storage:
  backend: s3
   bucket: my-bucket  # Wrong indent

# CORRECT: Consistent indentation (2 spaces)
storage:
  backend: s3
  bucket: my-bucket
```

Use a YAML validator:

```bash
# Install yamllint
pip install yamllint

# Validate config
yamllint config/config.yaml
```

### Missing Required Configuration

**Problem**: Required configuration options are missing.

```bash
# Error:
# ValidationError: 'storage.backend' is required
```

**Solution**:

Ensure your configuration includes all required fields:

```yaml
# Minimum required configuration
storage:
  backend: local  # or 's3'
  local:
    base_path: ./data  # Required for local backend
  # OR
  s3:
    bucket: your-bucket  # Required for s3 backend
    region: us-east-1

logging:
  level: INFO

models:
  default: lgbm
```

---

## AWS and S3 Issues

### AWS Credentials Not Found

**Problem**: Cannot authenticate to AWS.

```bash
# Error:
# botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Solution**:

```bash
# Option 1: Configure AWS CLI
aws configure
# Enter Access Key ID, Secret Access Key, Region

# Option 2: Set environment variables
export AWS_ACCESS_KEY_ID="your-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# Option 3: Use AWS profile
export AWS_PROFILE=prevcarga

# Verify credentials
aws sts get-caller-identity
```

### S3 Access Denied

**Problem**: Cannot read/write to S3 bucket.

```bash
# Error:
# botocore.exceptions.ClientError: An error occurred (AccessDenied)
```

**Solution**:

```bash
# Check bucket policy and IAM permissions
aws iam list-attached-user-policies --user-name your-user

# Test access manually
aws s3 ls s3://your-bucket/
aws s3 cp test.txt s3://your-bucket/test.txt

# Required IAM permissions:
# - s3:GetObject
# - s3:PutObject
# - s3:ListBucket
# - s3:DeleteObject (optional)
```

### S3 Bucket Not Found

**Problem**: Specified S3 bucket does not exist.

```bash
# Error:
# botocore.exceptions.ClientError: The specified bucket does not exist
```

**Solution**:

```bash
# List available buckets
aws s3 ls

# Check bucket name in config
cat config/config.yaml | grep bucket

# Create bucket if needed
aws s3 mb s3://your-bucket-name --region us-east-1
```

### Region Mismatch

**Problem**: S3 operations fail due to region configuration.

```bash
# Error:
# The bucket you are attempting to access must be addressed using the specified endpoint
```

**Solution**:

```bash
# Check bucket region
aws s3api get-bucket-location --bucket your-bucket

# Update configuration to match
# config/config.yaml
storage:
  s3:
    bucket: your-bucket
    region: us-west-2  # Match actual bucket region

# Or set environment variable
export AWS_DEFAULT_REGION=us-west-2
```

---

## Runtime Issues

### Out of Memory

**Problem**: Process killed due to insufficient memory.

```bash
# Error:
# Killed
# OR
# MemoryError: Unable to allocate...
```

**Solution**:

```bash
# Reduce parallel workers
prevcarga train --parallel 1

# Use smaller batch size
prevcarga train --batch-size 1000

# Limit data range
prevcarga train --start-date 2024-01-01 --end-date 2024-03-31

# Monitor memory usage
htop  # or top

# Increase swap (Linux)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Model Not Found

**Problem**: Cannot load a trained model.

```bash
# Error:
# FileNotFoundError: Model not found for area SECO_RJ
```

**Solution**:

```bash
# List available models
prevcarga model list --storage-backend local

# Check model path structure
ls -la data/models/

# Train the model if not exists
prevcarga train \
    --areas SECO_RJ \
    --models lgbm \
    --start-date 2023-01-01 \
    --end-date 2023-12-31
```

### Data Not Found

**Problem**: Input data files are missing.

```bash
# Error:
# FileNotFoundError: Data not found for area SECO_RJ, date 2024-01-15
```

**Solution**:

```bash
# Check data availability
prevcarga data list --area SECO_RJ

# Validate data directory
ls -la data/raw/

# For local backend, ensure data structure:
# data/
#   raw/
#     SECO_RJ/
#       2024-01-15.parquet
```

### Invalid Date Format

**Problem**: Date parsing errors.

```bash
# Error:
# ValueError: time data '01-15-2024' does not match format '%Y-%m-%d'
```

**Solution**:

```bash
# Use ISO format: YYYY-MM-DD
prevcarga predict --date 2024-01-15  # Correct

# Not these formats:
# --date 01-15-2024  # Wrong
# --date 15/01/2024  # Wrong
# --date "January 15, 2024"  # Wrong
```

---

## Performance Issues

### Slow Training

**Problem**: Model training takes too long.

**Solutions**:

```bash
# Enable parallel training
prevcarga train --parallel 4

# Use GPU if available (LightGBM)
# Requires LightGBM compiled with GPU support
prevcarga train --gpu

# Reduce hyperparameter search space
# In config:
models:
  lgbm:
    n_estimators: 500  # Reduce from 1000
    early_stopping_rounds: 50
```

### Slow Predictions

**Problem**: Prediction generation is slow.

**Solutions**:

```bash
# Enable batch processing
prevcarga predict --batch-size 10000

# Use parallel prediction
prevcarga predict --parallel 4

# Profile to find bottleneck
prevcarga predict --profile --output profile.json
```

### High Memory Usage

**Problem**: Memory usage grows excessively.

**Solutions**:

```python
# Use chunked processing in Python API
from src.data import DataLoader

loader = DataLoader(chunk_size=10000)
for chunk in loader.iter_chunks(start_date, end_date):
    process(chunk)
    del chunk  # Explicit cleanup
```

```bash
# Monitor memory during execution
watch -n 1 free -h

# Use memory-efficient data types
# In config:
data:
  optimize_memory: true
  float_precision: float32  # Instead of float64
```

---

## Docker Issues

### Container Fails to Start

**Problem**: Docker container exits immediately.

```bash
# Check logs
docker logs <container-id>

# Run with interactive terminal to see errors
docker run -it prevcarga:latest /bin/bash

# Check container status
docker ps -a
```

### Build Fails

**Problem**: Docker image build fails.

```bash
# Rebuild without cache
docker build --no-cache -t prevcarga:latest .

# Build with verbose output
docker build --progress=plain -t prevcarga:latest .

# Check disk space
df -h
docker system df
```

### Volume Mount Issues

**Problem**: Container cannot access mounted volumes.

```bash
# Check mount permissions
ls -la ./data/

# Use absolute paths
docker run -v /full/path/to/data:/app/data prevcarga:latest

# On macOS, ensure Docker has file sharing permissions
# Docker Desktop > Preferences > Resources > File Sharing

# On Linux, check SELinux labels
docker run -v ./data:/app/data:Z prevcarga:latest  # Add :Z suffix
```

### Network Issues in Container

**Problem**: Container cannot access external resources.

```bash
# Check network mode
docker run --network host prevcarga:latest

# DNS issues - use Google DNS
docker run --dns 8.8.8.8 prevcarga:latest

# Proxy configuration
docker run \
    -e HTTP_PROXY=$HTTP_PROXY \
    -e HTTPS_PROXY=$HTTPS_PROXY \
    prevcarga:latest
```

---

## Getting Help

If you cannot resolve your issue:

### 1. Collect Diagnostic Information

```bash
# System information
python --version
uv --version
prevcarga --version

# Environment
env | grep -E "(AWS|PREVCARGA|PYTHON)"

# Configuration
prevcarga config show

# Logs
cat logs/prevcarga.log | tail -100
```

### 2. Search Existing Issues

Check [GitHub Issues](https://github.com/gaugustog/prevcarga/issues) for similar problems.

### 3. Create a New Issue

If your issue is new, create a GitHub issue with:

1. **Description**: What you're trying to do
2. **Error message**: Complete error output
3. **Environment**: OS, Python version, PrevCarga version
4. **Steps to reproduce**: Minimal commands to trigger the issue
5. **Configuration**: Relevant config (redact sensitive data)

### 4. Community Support

- **GitHub Discussions**: [Ask questions](https://github.com/gaugustog/prevcarga/discussions)
- **Email**: Contact the development team for urgent production issues

---

## Quick Reference: Common Error Messages

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `ModuleNotFoundError` | Virtual env not activated | `source .venv/bin/activate` |
| `NoCredentialsError` | AWS not configured | `aws configure` |
| `AccessDenied` | IAM permissions | Check bucket policy |
| `MemoryError` | Insufficient RAM | Reduce `--parallel` or `--batch-size` |
| `FileNotFoundError: Model` | Model not trained | Run `prevcarga train` first |
| `YAML syntax error` | Config formatting | Validate with `yamllint` |
| `Connection timeout` | Network issues | Check firewall/proxy |

---

**Still stuck?** Open an issue on [GitHub](https://github.com/gaugustog/prevcarga/issues)
