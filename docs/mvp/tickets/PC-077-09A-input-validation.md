# PC-077-09A: Input Validation Framework

**Ticket ID:** PC-077-09A  
**Epic:** Epic-09A - Core CLI Commands  
**Parent Epic:** Epic-09 - CLI & User Interface  
**Story Points:** 5  
**Priority:** High  
**Status:** Not Started  

---

## 📋 Description

Implement comprehensive input validation framework (CLIValidators class) to prevent configuration errors and ensure data quality across all CLI commands.

---

## 🎯 Acceptance Criteria

- [ ] `CLIValidators` class with static validation methods
- [ ] Date range validation (max 2 years with user confirmation)
- [ ] Area selection validation against configuration
- [ ] Model compatibility checks (e.g., intraday requires LGBM)
- [ ] Output path validation with overwrite confirmation
- [ ] Horizon range validation (0-8)
- [ ] Parallel workers validation (positive integer, reasonable max)
- [ ] Configuration file validation
- [ ] Comprehensive error messages with actionable guidance
- [ ] Click parameter types for common validations

---

## 🔧 Technical Implementation

```python
# prevcarga/cli/validators.py
import click
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional


class CLIValidators:
    """Input validation utilities for CLI commands."""
    
    MAX_DATE_RANGE_DAYS = 730  # 2 years
    MAX_PARALLEL_WORKERS = 32
    VALID_HORIZONS = range(0, 9)  # D+0 to D+8
    
    @staticmethod
    def validate_date_range(start_date: datetime, end_date: datetime, 
                           max_days: Optional[int] = None):
        """Validate date range parameters.
        
        Args:
            start_date: Start date
            end_date: End date
            max_days: Maximum allowed days (default: 730)
        
        Raises:
            click.BadParameter: If validation fails
        """
        if start_date >= end_date:
            raise click.BadParameter(
                "Start date must be before end date. "
                f"Received: {start_date.date()} >= {end_date.date()}"
            )
        
        days_diff = (end_date - start_date).days
        max_allowed = max_days or CLIValidators.MAX_DATE_RANGE_DAYS
        
        if days_diff > max_allowed:
            message = (
                f"Date range exceeds maximum of {max_allowed} days "
                f"({max_allowed // 365} years). "
                f"Requested range: {days_diff} days. "
            )
            
            if click.confirm(f"{message}\nContinue anyway?", abort=True):
                click.echo("Warning: Large date range may take significant time")
        
        # Warn if date range is in the future
        if start_date > datetime.now():
            click.confirm(
                f"Start date is in the future ({start_date.date()}). Continue?",
                abort=True
            )
    
    @staticmethod
    def validate_area_selection(areas: List[str], config):
        """Validate area selection against configuration.
        
        Args:
            areas: List of area codes to validate
            config: System configuration object
        
        Raises:
            click.BadParameter: If invalid areas found
        """
        valid_areas = set(config.data.areas)
        invalid_areas = set(areas) - valid_areas
        
        if invalid_areas:
            raise click.BadParameter(
                f"Invalid area(s): {', '.join(sorted(invalid_areas))}. "
                f"Valid areas: {', '.join(sorted(valid_areas))}"
            )
        
        if not areas:
            raise click.BadParameter("At least one area must be specified")
    
    @staticmethod
    def validate_model_selection(models: List[str], config):
        """Validate model selection against configuration.
        
        Args:
            models: List of model types to validate
            config: System configuration object
        
        Raises:
            click.BadParameter: If invalid models found
        """
        valid_models = set(config.models.types)
        invalid_models = set(models) - valid_models
        
        if invalid_models:
            raise click.BadParameter(
                f"Invalid model(s): {', '.join(sorted(invalid_models))}. "
                f"Valid models: {', '.join(sorted(valid_models))}"
            )
    
    @staticmethod
    def validate_model_compatibility(models: List[str], operation: str):
        """Validate model compatibility for specific operations.
        
        Args:
            models: List of model types
            operation: Operation type (e.g., 'intraday', 'ensemble')
        
        Raises:
            click.BadParameter: If incompatible model configuration
        """
        if operation == 'intraday':
            if 'lgbm' not in models:
                raise click.BadParameter(
                    "Intraday predictions require LGBM model. "
                    "Add --model lgbm to your command."
                )
        
        elif operation == 'ensemble':
            if len(models) < 2:
                raise click.BadParameter(
                    f"Ensemble methods require at least 2 models. "
                    f"Received: {len(models)} model(s)"
                )
    
    @staticmethod
    def validate_output_path(path: Path, format: str, overwrite: bool = False):
        """Validate output path and format compatibility.
        
        Args:
            path: Output file path
            format: Expected file format
            overwrite: Whether to allow overwrite without confirmation
        
        Raises:
            click.BadParameter: If validation fails
        """
        # Check file extension matches format
        expected_ext = f".{format}"
        if not path.suffix:
            raise click.BadParameter(
                f"Output path must include file extension ({expected_ext}). "
                f"Received: {path}"
            )
        
        if path.suffix.lower() != expected_ext.lower():
            raise click.BadParameter(
                f"File extension '{path.suffix}' does not match format '{format}'. "
                f"Expected: {expected_ext}"
            )
        
        # Check if file exists
        if path.exists() and not overwrite:
            if not click.confirm(f"File {path} exists. Overwrite?", abort=True):
                raise click.ClickException("Operation cancelled by user")
        
        # Check if directory exists
        if not path.parent.exists():
            if click.confirm(f"Directory {path.parent} does not exist. Create?"):
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                raise click.BadParameter(f"Output directory does not exist: {path.parent}")
    
    @staticmethod
    def validate_horizon_selection(horizons: List[int]):
        """Validate forecast horizon selection.
        
        Args:
            horizons: List of horizon values
        
        Raises:
            click.BadParameter: If invalid horizons found
        """
        invalid_horizons = [h for h in horizons if h not in CLIValidators.VALID_HORIZONS]
        
        if invalid_horizons:
            raise click.BadParameter(
                f"Invalid horizon(s): {invalid_horizons}. "
                f"Valid horizons: 0-8 (D+0 to D+8)"
            )
        
        # Check for duplicates
        if len(horizons) != len(set(horizons)):
            duplicates = [h for h in set(horizons) if horizons.count(h) > 1]
            raise click.BadParameter(f"Duplicate horizons: {duplicates}")
    
    @staticmethod
    def validate_parallel_workers(workers: int):
        """Validate parallel worker configuration.
        
        Args:
            workers: Number of parallel workers
        
        Raises:
            click.BadParameter: If invalid worker count
        """
        if workers < 1:
            raise click.BadParameter(
                f"Parallel workers must be positive. Received: {workers}"
            )
        
        if workers > CLIValidators.MAX_PARALLEL_WORKERS:
            raise click.BadParameter(
                f"Parallel workers exceeds maximum ({CLIValidators.MAX_PARALLEL_WORKERS}). "
                f"Received: {workers}"
            )
        
        # Warn if too many workers
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if workers > cpu_count * 2:
            click.echo(
                f"Warning: {workers} workers requested but only {cpu_count} CPUs available. "
                f"Performance may be degraded."
            )
    
    @staticmethod
    def validate_config_file(config_path: Path):
        """Validate configuration file exists and is readable.
        
        Args:
            config_path: Path to configuration file
        
        Raises:
            click.BadParameter: If file not found or unreadable
        """
        if not config_path.exists():
            raise click.BadParameter(
                f"Configuration file not found: {config_path}. "
                f"Use 'prevcarga config init' to create a new configuration."
            )
        
        if not config_path.is_file():
            raise click.BadParameter(f"Path is not a file: {config_path}")
        
        # Check file is readable
        try:
            with open(config_path) as f:
                f.read(1)
        except PermissionError:
            raise click.BadParameter(f"Permission denied reading: {config_path}")
        except Exception as e:
            raise click.BadParameter(f"Cannot read configuration file: {e}")
    
    @staticmethod
    def validate_combination_method(method: str, models: List[str]):
        """Validate combination method compatibility.
        
        Args:
            method: Combination method
            models: List of models
        
        Raises:
            click.BadParameter: If incompatible configuration
        """
        if method in ['weighted_avg', 'stacking', 'markov']:
            if len(models) < 2:
                raise click.BadParameter(
                    f"Combination method '{method}' requires multiple models. "
                    f"Received: {len(models)} model(s)"
                )


# Custom Click parameter types for common validations
class DateRangeParamType(click.ParamType):
    """Custom parameter type for date range validation."""
    
    name = "daterange"
    
    def convert(self, value, param, ctx):
        try:
            date = datetime.strptime(value, '%Y-%m-%d')
            return date
        except ValueError:
            self.fail(
                f"Invalid date format: {value}. Expected: YYYY-MM-DD",
                param,
                ctx
            )


class AreaListParamType(click.ParamType):
    """Custom parameter type for area list validation."""
    
    name = "arealist"
    
    def __init__(self, valid_areas: List[str]):
        self.valid_areas = set(valid_areas)
    
    def convert(self, value, param, ctx):
        if value not in self.valid_areas:
            self.fail(
                f"Invalid area: {value}. "
                f"Valid areas: {', '.join(sorted(self.valid_areas))}",
                param,
                ctx
            )
        return value
```

---

## 🧪 Testing Requirements

```python
# tests/cli/test_validators.py
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from click.testing import CliRunner
import click

from prevcarga.cli.validators import CLIValidators


def test_validate_date_range_valid():
    """Test valid date range."""
    start = datetime(2023, 1, 1)
    end = datetime(2023, 12, 31)
    # Should not raise
    CLIValidators.validate_date_range(start, end)


def test_validate_date_range_invalid_order():
    """Test date range with start after end."""
    start = datetime(2023, 12, 31)
    end = datetime(2023, 1, 1)
    
    with pytest.raises(click.BadParameter, match="Start date must be before"):
        CLIValidators.validate_date_range(start, end)


def test_validate_date_range_too_long(monkeypatch):
    """Test date range exceeding maximum."""
    start = datetime(2020, 1, 1)
    end = datetime(2023, 1, 1)  # 3 years
    
    # Mock user confirmation to decline
    monkeypatch.setattr('click.confirm', lambda *args, **kwargs: False)
    
    with pytest.raises(click.Abort):
        CLIValidators.validate_date_range(start, end)


def test_validate_area_selection_valid(mock_config):
    """Test valid area selection."""
    areas = ['SE', 'S']
    CLIValidators.validate_area_selection(areas, mock_config)


def test_validate_area_selection_invalid(mock_config):
    """Test invalid area selection."""
    areas = ['SE', 'INVALID']
    
    with pytest.raises(click.BadParameter, match="Invalid area"):
        CLIValidators.validate_area_selection(areas, mock_config)


def test_validate_model_compatibility_intraday():
    """Test model compatibility for intraday."""
    with pytest.raises(click.BadParameter, match="require LGBM"):
        CLIValidators.validate_model_compatibility(['rf'], 'intraday')


def test_validate_output_path_no_extension():
    """Test output path without extension."""
    path = Path('output')
    
    with pytest.raises(click.BadParameter, match="must include file extension"):
        CLIValidators.validate_output_path(path, 'csv')


def test_validate_output_path_wrong_extension():
    """Test output path with wrong extension."""
    path = Path('output.json')
    
    with pytest.raises(click.BadParameter, match="does not match format"):
        CLIValidators.validate_output_path(path, 'csv')


def test_validate_horizon_selection_invalid():
    """Test invalid horizon selection."""
    horizons = [0, 1, 10]  # 10 is invalid
    
    with pytest.raises(click.BadParameter, match="Invalid horizon"):
        CLIValidators.validate_horizon_selection(horizons)


def test_validate_parallel_workers_negative():
    """Test negative parallel workers."""
    with pytest.raises(click.BadParameter, match="must be positive"):
        CLIValidators.validate_parallel_workers(-1)


def test_validate_parallel_workers_too_many():
    """Test too many parallel workers."""
    with pytest.raises(click.BadParameter, match="exceeds maximum"):
        CLIValidators.validate_parallel_workers(100)
```

---

## 📚 Documentation

### **Error Message Guidelines**
```python
# Good error messages:
# ✓ Clear problem statement
# ✓ Expected vs actual values
# ✓ Actionable guidance

"Invalid area(s): INVALID. Valid areas: SE, S, NE, N, CO"

# Bad error messages:
# ✗ Vague
# ✗ No context
# ✗ No solution

"Invalid input"
```

---

## ✅ Definition of Done

- [ ] CLIValidators class implemented with all methods
- [ ] Custom Click parameter types created
- [ ] Error messages are clear and actionable
- [ ] Unit tests pass (>90% coverage)
- [ ] Integration tests with commands pass
- [ ] Documentation complete
- [ ] Code reviewed

---

**Assignee:** Backend Team  
**Estimated Hours:** 8-10 hours  
**Target Completion:** Week 24, Day 4
