# PC-073-09A: Core CLI Application Structure

**Ticket ID:** PC-073-09A  
**Epic:** Epic-09A - Core CLI Commands  
**Parent Epic:** Epic-09 - CLI & User Interface  
**Story Points:** 5  
**Priority:** High  
**Status:** Not Started  

---

## 📋 Description

Implement the core CLI application structure using the Click framework, establishing the foundation for all command-line interactions in the PrevCarga unified electric load forecasting system.

---

## 🎯 Acceptance Criteria

- [ ] Click-based CLI application with entry point configured
- [ ] Global options implemented: `--config`, `--verbose`, `--log-level`
- [ ] Command context management with `click.Context`
- [ ] Version command displays system information
- [ ] Status command checks system connectivity (S3, model registry, config, data)
- [ ] Error handling framework with user-friendly messages
- [ ] Auto-completion support configured
- [ ] Logging system initialized based on global options
- [ ] Configuration loading mechanism implemented
- [ ] Help texts are clear and complete

---

## 🔧 Technical Implementation

### **Files to Create/Modify**

```
prevcarga/
├── cli/
│   ├── __init__.py          # CLI package initialization
│   ├── main.py              # Main CLI application with @click.group()
│   ├── commands/            # Command modules (created in other tickets)
│   └── utils.py             # CLI utilities (logging, config loading)
└── setup.py                 # Entry point configuration
```

### **Core CLI Structure**

```python
# prevcarga/cli/main.py
import click
from pathlib import Path
from typing import Optional
import sys
import platform

@click.group()
@click.option('--config', '-c', type=Path, help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='INFO', help='Set logging level')
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], verbose: bool, log_level: str):
    """PrevCarga Unified Electric Load Forecasting System
    
    A comprehensive CLI for training, prediction, and evaluation of
    electric load forecasting models across 26 time series.
    
    Examples:
        prevcarga --config custom.yaml train model --model lgbm
        prevcarga --verbose status
        prevcarga --log-level DEBUG predict batch --date 2024-01-15
    """
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose
    ctx.obj['log_level'] = log_level
    
    # Initialize logging
    from prevcarga.cli.utils import setup_logging
    setup_logging(log_level, verbose)
    
    # Load configuration
    from prevcarga.cli.utils import load_configuration
    ctx.obj['system_config'] = load_configuration(config)


@cli.command()
def version():
    """Display version and system information."""
    from prevcarga import __version__
    
    click.echo(f"PrevCarga CLI v{__version__}")
    click.echo(f"Python: {sys.version}")
    click.echo(f"Platform: {platform.platform()}")
    click.echo(f"Architecture: {platform.machine()}")


@cli.command()
@click.pass_context
def status(ctx):
    """Check system status and connectivity.
    
    Verifies:
    - S3 connection and bucket accessibility
    - Model registry availability
    - Configuration validity
    - Data availability in S3
    - Required dependencies
    """
    config = ctx.obj.get('system_config')
    
    checks = [
        ('S3 Connection', check_s3_connection),
        ('Model Registry', check_model_registry),
        ('Configuration', check_configuration),
        ('Data Availability', check_data_availability),
        ('Dependencies', check_dependencies)
    ]
    
    click.echo("System Status Check")
    click.echo("=" * 50)
    
    with click.progressbar(checks, label='Checking system status', 
                          show_eta=False, show_percent=False) as bar:
        results = []
        for check_name, check_func in bar:
            try:
                check_func(config)
                results.append((check_name, '✓', 'OK'))
            except Exception as e:
                results.append((check_name, '✗', str(e)))
    
    click.echo()
    for check_name, status_icon, message in results:
        click.echo(f"{status_icon} {check_name:20} {message}")
    
    # Overall status
    all_passed = all(status == '✓' for _, status, _ in results)
    click.echo()
    if all_passed:
        click.secho("✓ System operational", fg='green', bold=True)
    else:
        click.secho("✗ System has issues", fg='red', bold=True)
        sys.exit(1)


def check_s3_connection(config):
    """Check S3 connectivity."""
    import boto3
    from botocore.exceptions import ClientError
    
    try:
        s3_client = boto3.client('s3')
        s3_client.head_bucket(Bucket=config.s3.bucket)
    except ClientError as e:
        raise Exception(f"Cannot access S3 bucket: {e}")


def check_model_registry(config):
    """Check model registry availability."""
    # Placeholder for model registry check
    pass


def check_configuration(config):
    """Validate configuration."""
    from prevcarga.cli.utils import validate_config
    validate_config(config)


def check_data_availability(config):
    """Check data availability in S3."""
    # Placeholder for data availability check
    pass


def check_dependencies(config):
    """Check required dependencies."""
    required = ['pandas', 'numpy', 'lightgbm', 'scikit-learn', 'click']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        raise Exception(f"Missing dependencies: {', '.join(missing)}")


if __name__ == '__main__':
    cli()
```

### **CLI Utilities**

```python
# prevcarga/cli/utils.py
import logging
from pathlib import Path
from typing import Optional
import yaml
import click

def setup_logging(log_level: str, verbose: bool):
    """Setup logging configuration."""
    level = getattr(logging, log_level)
    
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if verbose:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('prevcarga.log')
        ]
    )


def load_configuration(config_path: Optional[Path]):
    """Load system configuration."""
    from prevcarga.config import Config
    
    if config_path:
        if not config_path.exists():
            raise click.ClickException(f"Configuration file not found: {config_path}")
        
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        
        return Config.from_dict(config_dict)
    else:
        # Load default configuration
        return Config.load_default()


def validate_config(config):
    """Validate configuration object."""
    required_sections = ['data', 'models', 's3', 'features']
    
    for section in required_sections:
        if not hasattr(config, section):
            raise Exception(f"Missing configuration section: {section}")
```

### **Entry Point Configuration**

```python
# setup.py (add entry point)
from setuptools import setup, find_packages

setup(
    name='prevcarga',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'click>=8.0',
        'pyyaml>=6.0',
        'boto3>=1.26',
        # ... other dependencies
    ],
    entry_points={
        'console_scripts': [
            'prevcarga=prevcarga.cli.main:cli',
        ],
    },
)
```

### **Package Initialization**

```python
# prevcarga/cli/__init__.py
"""PrevCarga CLI package."""

from prevcarga.cli.main import cli

__all__ = ['cli']
```

---

## 🧪 Testing Requirements

### **Unit Tests**

```python
# tests/cli/test_main.py
import pytest
from click.testing import CliRunner
from prevcarga.cli.main import cli, version, status

def test_cli_help():
    """Test CLI help output."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'PrevCarga' in result.output


def test_version_command():
    """Test version command."""
    runner = CliRunner()
    result = runner.invoke(version)
    assert result.exit_code == 0
    assert 'PrevCarga CLI' in result.output
    assert 'Python:' in result.output


def test_global_options():
    """Test global options parsing."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--verbose', '--log-level', 'DEBUG', '--help'])
    assert result.exit_code == 0


def test_config_file_not_found():
    """Test error handling for missing config file."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--config', 'nonexistent.yaml', 'version'])
    assert result.exit_code != 0
    assert 'not found' in result.output.lower()


@pytest.mark.integration
def test_status_command(mock_config):
    """Test status command with mocked dependencies."""
    runner = CliRunner()
    result = runner.invoke(status)
    assert 'System Status Check' in result.output
```

### **Integration Tests**

- Test CLI application initialization
- Test configuration loading from file
- Test logging setup with different levels
- Test error handling for invalid commands
- Test auto-completion script generation

---

## 📦 Dependencies

### **New Dependencies**
- `click>=8.0` - CLI framework
- `pyyaml>=6.0` - Configuration file parsing

### **Integration Points**
- Configuration system (to be implemented)
- Logging framework
- S3 connectivity (boto3)

---

## 📚 Documentation

### **CLI Help Text**
- Main command help with examples
- Global options documentation
- Version and status command descriptions

### **User Guide Section**
```markdown
## Getting Started with PrevCarga CLI

### Installation
```bash
pip install prevcarga
```

### Basic Usage
```bash
# Check version
prevcarga version

# Check system status
prevcarga status

# Get help
prevcarga --help

# Use custom configuration
prevcarga --config my-config.yaml status
```

### Global Options
- `--config, -c`: Specify custom configuration file
- `--verbose, -v`: Enable verbose output
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
```

---

## ⚡ Performance Targets

- CLI startup time: <500ms
- Version command: <100ms
- Status command: <5 seconds (with all checks)
- Memory usage: <50MB for CLI core

---

## 🔗 Related Tickets

### **Depends On**
- Configuration system implementation

### **Blocks**
- PC-074-09A: Training Commands
- PC-075-09A: Prediction Commands
- PC-076-09A: Feature Commands
- PC-077-09A: Input Validation
- PC-078-09A: Progress Monitoring

### **Related Epics**
- Epic-09A: Core CLI Commands
- Epic-09B: Interactive Mode & Advanced Features

---

## ✅ Definition of Done

- [ ] Click CLI application structure implemented
- [ ] Global options working correctly
- [ ] Version command displays all required information
- [ ] Status command checks all system components
- [ ] Error handling provides clear user feedback
- [ ] Entry point configured in setup.py
- [ ] Auto-completion support enabled
- [ ] Unit tests pass with >80% coverage
- [ ] Integration tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved
- [ ] Ready for command module integration

---

**Assignee:** Backend Team  
**Reviewer:** Tech Lead  
**Estimated Hours:** 8-12 hours  
**Created:** 2024-11-19  
**Target Completion:** Week 24, Day 1
