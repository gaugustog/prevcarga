"""Configuration management CLI commands.

This module provides CLI commands for managing PrevCarga system configuration,
including showing, validating, initializing, comparing, and exporting
configuration files.

Key Commands:
- config show: Display current configuration
- config validate: Validate configuration for completeness
- config init: Initialize configuration from template
- config diff: Compare two configuration files
- config export: Export configuration to various formats

Example:
    ```bash
    # Show current configuration
    prevcarga config show

    # Show specific section
    prevcarga config show --section training

    # Validate configuration
    prevcarga config validate --strict

    # Initialize from template
    prevcarga config init --template production

    # Compare configurations
    prevcarga config diff --config1 dev.yaml --config2 prod.yaml

    # Export to JSON
    prevcarga config export --format json --output config.json
    ```
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence


class ConfigFormat(Enum):
    """Configuration output format."""

    YAML = "yaml"
    JSON = "json"
    TABLE = "table"
    ENV = "env"


class ConfigTemplate(Enum):
    """Configuration template types."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class ValidationSeverity(Enum):
    """Validation issue severity."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Single validation issue."""

    severity: ValidationSeverity
    field: str
    message: str
    suggestion: str | None = None


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    checks_performed: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0


@dataclass
class DiffItem:
    """Single diff item between configurations."""

    path: str
    change_type: str  # "added", "removed", "changed"
    old_value: Any = None
    new_value: Any = None


@dataclass
class ConfigDiffResult:
    """Result of configuration comparison."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    identical_sections: list[str] = field(default_factory=list)

    @property
    def has_differences(self) -> bool:
        """Check if configurations differ."""
        return bool(self.added or self.removed or self.changed)

    @property
    def total_changes(self) -> int:
        """Total number of changes."""
        return len(self.added) + len(self.removed) + len(self.changed)


class ConfigValidator:
    """Validate configuration for completeness and correctness."""

    # Required fields for each section
    REQUIRED_FIELDS = {
        "system": ["environment", "log_level"],
        "training": ["parallel_workers", "timeout_seconds"],
        "prediction": ["output_format"],
        "backtesting": ["step_days"],
    }

    # Valid values for certain fields
    VALID_VALUES = {
        "system.environment": ["development", "staging", "production"],
        "system.log_level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        "prediction.output_format": ["csv", "json", "parquet"],
    }

    # Range constraints
    RANGE_CONSTRAINTS = {
        "training.parallel_workers": (1, 64),
        "training.timeout_seconds": (60, 86400),
        "training.cv_folds": (2, 20),
        "training.optimization_trials": (10, 1000),
        "backtesting.step_days": (1, 90),
    }

    def __init__(self, strict: bool = False) -> None:
        """Initialize validator.

        Args:
            strict: Enable strict validation mode.
        """
        self.strict = strict
        self.checks_performed = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.issues: list[ValidationIssue] = []

    def validate(self, config: dict[str, Any]) -> ValidationResult:
        """Validate configuration.

        Args:
            config: Configuration dictionary to validate.

        Returns:
            ValidationResult with errors and warnings.
        """
        self.errors = []
        self.warnings = []
        self.issues = []
        self.checks_performed = 0

        # Run all validation checks
        self._check_required_fields(config)
        self._check_value_types(config)
        self._check_value_ranges(config)
        self._check_valid_values(config)
        self._check_cross_field_consistency(config)

        if self.strict:
            self._check_strict_mode(config)

        is_valid = len(self.errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            checks_performed=self.checks_performed,
            errors=self.errors,
            warnings=self.warnings,
            issues=self.issues,
        )

    def _check_required_fields(self, config: dict[str, Any]) -> None:
        """Check for required fields."""
        for section, fields in self.REQUIRED_FIELDS.items():
            self.checks_performed += 1
            if section not in config:
                self.errors.append(f"Missing required section: {section}")
                self.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field=section,
                        message=f"Missing required section: {section}",
                        suggestion=f"Add '{section}' section to configuration",
                    )
                )
                continue

            section_config = config[section]
            if not isinstance(section_config, dict):
                self.errors.append(f"Section '{section}' must be a dictionary")
                continue

            for field_name in fields:
                self.checks_performed += 1
                if field_name not in section_config:
                    self.warnings.append(
                        f"Missing recommended field: {section}.{field_name}"
                    )
                    self.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            field=f"{section}.{field_name}",
                            message=f"Missing recommended field",
                            suggestion=f"Consider adding {field_name} to {section}",
                        )
                    )

    def _check_value_types(self, config: dict[str, Any]) -> None:
        """Check value types."""
        type_checks = {
            "training.parallel_workers": int,
            "training.timeout_seconds": int,
            "training.cv_folds": int,
            "training.optimize_hyperparameters": bool,
            "prediction.apply_reconciliation": bool,
            "backtesting.step_days": int,
            "backtesting.generate_report": bool,
        }

        for path, expected_type in type_checks.items():
            self.checks_performed += 1
            value = self._get_nested_value(config, path)
            if value is not None and not isinstance(value, expected_type):
                self.errors.append(
                    f"Invalid type for {path}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
                self.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field=path,
                        message=f"Expected {expected_type.__name__}, got {type(value).__name__}",
                        suggestion=f"Change value to {expected_type.__name__}",
                    )
                )

    def _check_value_ranges(self, config: dict[str, Any]) -> None:
        """Check value ranges."""
        for path, (min_val, max_val) in self.RANGE_CONSTRAINTS.items():
            self.checks_performed += 1
            value = self._get_nested_value(config, path)
            if value is not None and isinstance(value, (int, float)):
                if value < min_val or value > max_val:
                    self.errors.append(
                        f"Value for {path} out of range: {value} "
                        f"(expected {min_val}-{max_val})"
                    )
                    self.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            field=path,
                            message=f"Value {value} out of range [{min_val}, {max_val}]",
                            suggestion=f"Set value between {min_val} and {max_val}",
                        )
                    )

    def _check_valid_values(self, config: dict[str, Any]) -> None:
        """Check values against valid options."""
        for path, valid_options in self.VALID_VALUES.items():
            self.checks_performed += 1
            value = self._get_nested_value(config, path)
            if value is not None:
                # Handle enum values
                if hasattr(value, "value"):
                    value = value.value
                if str(value).lower() not in [v.lower() for v in valid_options]:
                    self.errors.append(
                        f"Invalid value for {path}: '{value}'. "
                        f"Valid options: {', '.join(valid_options)}"
                    )
                    self.issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            field=path,
                            message=f"Invalid value: {value}",
                            suggestion=f"Use one of: {', '.join(valid_options)}",
                        )
                    )

    def _check_cross_field_consistency(self, config: dict[str, Any]) -> None:
        """Check cross-field consistency."""
        self.checks_performed += 1

        # Check training workers vs timeout
        workers = self._get_nested_value(config, "training.parallel_workers")
        timeout = self._get_nested_value(config, "training.timeout_seconds")

        # Only compare if both are integers
        if (workers and timeout and
            isinstance(workers, int) and isinstance(timeout, int) and
            workers > 16 and timeout < 300):
            self.warnings.append(
                "High parallel workers with low timeout may cause issues"
            )
            self.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="training",
                    message="High workers with low timeout may cause issues",
                    suggestion="Increase timeout_seconds for more workers",
                )
            )

        # Check backtesting dates
        start_date = self._get_nested_value(config, "backtesting.start_date")
        end_date = self._get_nested_value(config, "backtesting.end_date")

        if start_date and end_date:
            try:
                if isinstance(start_date, str):
                    start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                else:
                    start = start_date
                if isinstance(end_date, str):
                    end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                else:
                    end = end_date

                if start >= end:
                    self.errors.append("Backtesting start_date must be before end_date")
            except (ValueError, TypeError):
                pass  # Invalid date format will be caught by type check

    def _check_strict_mode(self, config: dict[str, Any]) -> None:
        """Additional strict mode checks."""
        self.checks_performed += 1

        # Check for empty sections
        for section_name, section_config in config.items():
            if isinstance(section_config, dict) and not section_config:
                self.errors.append(f"Empty section not allowed in strict mode: {section_name}")
                self.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        field=section_name,
                        message="Empty section not allowed in strict mode",
                        suggestion="Add configuration values or remove section",
                    )
                )

        # Check for unknown sections
        known_sections = {"system", "training", "prediction", "backtesting",
                         "reconciliation", "features", "storage", "models"}
        for section_name in config.keys():
            self.checks_performed += 1
            if section_name not in known_sections:
                self.warnings.append(f"Unknown section in strict mode: {section_name}")

    def _get_nested_value(self, config: dict[str, Any], path: str) -> Any:
        """Get nested configuration value by dot-separated path."""
        parts = path.split(".")
        value = config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value


class ConfigComparator:
    """Compare two configuration files."""

    def compare(
        self,
        config1: dict[str, Any],
        config2: dict[str, Any],
        prefix: str = "",
    ) -> ConfigDiffResult:
        """Compare two configurations.

        Args:
            config1: First configuration.
            config2: Second configuration.
            prefix: Key prefix for nested comparison.

        Returns:
            ConfigDiffResult with differences.
        """
        result = ConfigDiffResult()
        self._compare_dicts(config1, config2, prefix, result)
        return result

    def _compare_dicts(
        self,
        dict1: dict[str, Any],
        dict2: dict[str, Any],
        prefix: str,
        result: ConfigDiffResult,
    ) -> None:
        """Recursively compare dictionaries."""
        all_keys = set(dict1.keys()) | set(dict2.keys())

        for key in sorted(all_keys):
            path = f"{prefix}.{key}" if prefix else key

            if key not in dict1:
                result.added.append(path)
            elif key not in dict2:
                result.removed.append(path)
            else:
                val1 = self._normalize_value(dict1[key])
                val2 = self._normalize_value(dict2[key])

                if isinstance(val1, dict) and isinstance(val2, dict):
                    self._compare_dicts(val1, val2, path, result)
                elif val1 != val2:
                    result.changed[path] = (val1, val2)
                else:
                    # Track identical top-level sections
                    if not prefix:
                        result.identical_sections.append(key)

    def _normalize_value(self, value: Any) -> Any:
        """Normalize value for comparison."""
        if hasattr(value, "value"):  # Enum
            return value.value
        if hasattr(value, "model_dump"):  # Pydantic model
            return value.model_dump()
        if hasattr(value, "__dict__") and not isinstance(value, (dict, list, str, int, float, bool)):
            return vars(value)
        return value


class ConfigTemplateGenerator:
    """Generate configuration templates."""

    TEMPLATES = {
        ConfigTemplate.DEVELOPMENT: {
            "system": {
                "environment": "development",
                "log_level": "DEBUG",
                "log_format": "text",
                "timezone": "America/Sao_Paulo",
                "temp_dir": "/tmp/prevcarga",
                "cache_enabled": True,
            },
            "training": {
                "parallel_workers": 4,
                "timeout_seconds": 3600,
                "optimize_hyperparameters": True,
                "optimization_trials": 50,
                "cv_folds": 5,
                "model_output_dir": "models/",
            },
            "prediction": {
                "model_dir": "models/",
                "output_format": "csv",
                "output_dir": "predictions/",
                "apply_reconciliation": True,
                "horizons": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            },
            "backtesting": {
                "start_date": "2024-01-01",
                "end_date": "2024-03-31",
                "step_days": 7,
                "retrain_interval_days": 30,
                "generate_report": True,
            },
            "reconciliation": {
                "enabled": True,
                "method": "mint",
                "covariance_method": "shrinkage",
            },
            "features": {
                "plugins": ["temporal", "calendar", "lag", "rolling"],
                "cache_enabled": True,
            },
        },
        ConfigTemplate.PRODUCTION: {
            "system": {
                "environment": "production",
                "log_level": "INFO",
                "log_format": "json",
                "timezone": "America/Sao_Paulo",
                "temp_dir": "/tmp/prevcarga",
                "cache_enabled": True,
            },
            "training": {
                "parallel_workers": 8,
                "timeout_seconds": 7200,
                "optimize_hyperparameters": False,
                "optimization_trials": 100,
                "cv_folds": 5,
                "model_output_dir": "s3://prevcarga-models/",
            },
            "prediction": {
                "model_dir": "s3://prevcarga-models/",
                "output_format": "parquet",
                "output_dir": "s3://prevcarga-predictions/",
                "apply_reconciliation": True,
                "horizons": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            },
            "backtesting": {
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
                "step_days": 7,
                "retrain_interval_days": 14,
                "generate_report": True,
            },
            "reconciliation": {
                "enabled": True,
                "method": "mint",
                "covariance_method": "shrinkage",
            },
            "features": {
                "plugins": [
                    "temporal", "calendar", "lag", "rolling",
                    "wavelet", "seasonality", "blf",
                ],
                "cache_enabled": True,
            },
            "storage": {
                "backend": "s3",
                "s3_bucket": "prevcarga-data",
                "s3_region": "us-east-1",
            },
        },
        ConfigTemplate.TESTING: {
            "system": {
                "environment": "development",
                "log_level": "WARNING",
                "log_format": "text",
                "timezone": "America/Sao_Paulo",
                "temp_dir": "/tmp/prevcarga-test",
                "cache_enabled": False,
            },
            "training": {
                "parallel_workers": 2,
                "timeout_seconds": 600,
                "optimize_hyperparameters": False,
                "optimization_trials": 10,
                "cv_folds": 3,
                "model_output_dir": "test_models/",
            },
            "prediction": {
                "model_dir": "test_models/",
                "output_format": "csv",
                "output_dir": "test_predictions/",
                "apply_reconciliation": False,
                "horizons": [0, 1],
            },
            "backtesting": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "step_days": 7,
                "retrain_interval_days": 7,
                "generate_report": False,
            },
            "reconciliation": {
                "enabled": False,
                "method": "ols",
            },
            "features": {
                "plugins": ["temporal", "calendar"],
                "cache_enabled": False,
            },
        },
    }

    @classmethod
    def get_template(cls, template: ConfigTemplate | str) -> dict[str, Any]:
        """Get configuration template.

        Args:
            template: Template name or enum.

        Returns:
            Template configuration dictionary.
        """
        if isinstance(template, str):
            template = ConfigTemplate(template.lower())
        return cls.TEMPLATES.get(template, cls.TEMPLATES[ConfigTemplate.DEVELOPMENT])


def load_config_file(filepath: Path) -> dict[str, Any]:
    """Load configuration from file.

    Args:
        filepath: Path to configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        click.ClickException: If file cannot be loaded.
    """
    if not filepath.exists():
        raise click.ClickException(f"Configuration file not found: {filepath}")

    try:
        with open(filepath, encoding="utf-8") as f:
            if filepath.suffix in (".yaml", ".yml"):
                return yaml.safe_load(f) or {}
            elif filepath.suffix == ".json":
                return json.load(f)
            else:
                raise click.ClickException(f"Unsupported file format: {filepath.suffix}")
    except yaml.YAMLError as e:
        raise click.ClickException(f"Invalid YAML in {filepath}: {e}")
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in {filepath}: {e}")


def config_to_yaml(config: dict[str, Any]) -> str:
    """Convert configuration to YAML string."""

    def serialize(obj: Any) -> Any:
        """Convert non-serializable types."""
        if hasattr(obj, "value"):  # Enum
            return obj.value
        if hasattr(obj, "model_dump"):  # Pydantic
            return obj.model_dump()
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(item) for item in obj]
        return obj

    serialized = serialize(config)
    return yaml.dump(serialized, default_flow_style=False, sort_keys=False)


def config_to_json(config: dict[str, Any]) -> str:
    """Convert configuration to JSON string."""

    def serialize(obj: Any) -> Any:
        """Convert non-serializable types."""
        if hasattr(obj, "value"):  # Enum
            return obj.value
        if hasattr(obj, "model_dump"):  # Pydantic
            return obj.model_dump()
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(item) for item in obj]
        return obj

    serialized = serialize(config)
    return json.dumps(serialized, indent=2)


def config_to_env_vars(config: dict[str, Any], prefix: str = "PREVCARGA") -> str:
    """Convert configuration to environment variable format.

    Args:
        config: Configuration dictionary.
        prefix: Environment variable prefix.

    Returns:
        String with environment variable definitions.
    """
    lines = []

    def flatten(obj: Any, path: str) -> None:
        """Recursively flatten configuration to env vars."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}_{key.upper()}" if path else key.upper()
                flatten(value, new_path)
        elif isinstance(obj, list):
            # Convert lists to comma-separated strings
            str_values = [str(v.value if hasattr(v, "value") else v) for v in obj]
            lines.append(f'{prefix}_{path}="{",".join(str_values)}"')
        elif isinstance(obj, bool):
            lines.append(f"{prefix}_{path}={'true' if obj else 'false'}")
        elif obj is not None:
            value = obj.value if hasattr(obj, "value") else obj
            if isinstance(value, str) and " " in value:
                lines.append(f'{prefix}_{path}="{value}"')
            else:
                lines.append(f"{prefix}_{path}={value}")

    for section, section_config in config.items():
        if isinstance(section_config, dict):
            for key, value in section_config.items():
                flatten(value, f"{section.upper()}_{key.upper()}")
        else:
            flatten(section_config, section.upper())

    return "\n".join(sorted(lines))


def display_config_table(config: dict[str, Any], section: str | None = None) -> None:
    """Display configuration as a formatted table.

    Args:
        config: Configuration dictionary.
        section: Optional section to display.
    """
    if section:
        if section not in config:
            click.echo(f"Section '{section}' not found")
            return
        click.echo(f"\n{'=' * 60}")
        click.secho(f" {section.upper()} ", fg="cyan", bold=True)
        click.echo(f"{'=' * 60}")
        _display_section_table(config[section])
    else:
        for section_name, section_config in config.items():
            click.echo(f"\n{'=' * 60}")
            click.secho(f" {section_name.upper()} ", fg="cyan", bold=True)
            click.echo(f"{'=' * 60}")
            if isinstance(section_config, dict):
                _display_section_table(section_config)
            else:
                click.echo(f"  {section_config}")


def _display_section_table(section: dict[str, Any], indent: int = 2) -> None:
    """Display section as table."""
    prefix = " " * indent
    for key, value in section.items():
        if isinstance(value, dict):
            click.echo(f"{prefix}{key}:")
            _display_section_table(value, indent + 2)
        elif isinstance(value, list):
            click.echo(f"{prefix}{key}: {', '.join(str(v) for v in value)}")
        elif hasattr(value, "value"):  # Enum
            click.echo(f"{prefix}{key}: {value.value}")
        else:
            click.echo(f"{prefix}{key}: {value}")


# Click Command Group
@click.group()
def config_cmd() -> None:
    """Configuration management commands.

    Manage PrevCarga system configuration including showing, validating,
    initializing, comparing, and exporting configuration files.
    """
    pass


@config_cmd.command("show")
@click.option(
    "--section",
    "-s",
    type=str,
    help="Configuration section to display (e.g., training, prediction)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["yaml", "json", "table"]),
    default="yaml",
    help="Output format",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file path",
)
def show(
    section: str | None,
    output_format: str,
    config_file: Path | None,
) -> None:
    """Display current system configuration.

    Shows the current configuration in the specified format.
    Use --section to filter to a specific configuration section.

    Examples:
        prevcarga config show
        prevcarga config show --section training
        prevcarga config show --format json
        prevcarga config show --config-file custom.yaml
    """
    # Load configuration
    if config_file:
        config = load_config_file(config_file)
    else:
        # Try to find default config file
        default_paths = [
            Path("prevcarga.yaml"),
            Path("prevcarga.yml"),
            Path("config/prevcarga.yaml"),
        ]
        config = None
        for path in default_paths:
            if path.exists():
                config = load_config_file(path)
                break

        if config is None:
            # Use default configuration template
            config = ConfigTemplateGenerator.get_template(ConfigTemplate.DEVELOPMENT)
            click.secho("No configuration file found. Showing default configuration.", fg="yellow")

    # Filter by section if specified
    if section:
        if section not in config:
            raise click.ClickException(f"Unknown section: {section}")
        config = {section: config[section]}

    # Display based on format
    if output_format == "yaml":
        click.echo(config_to_yaml(config))
    elif output_format == "json":
        click.echo(config_to_json(config))
    elif output_format == "table":
        display_config_table(config, section)


@config_cmd.command("validate")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file to validate",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Enable strict validation mode",
)
def validate(config_file: Path | None, strict: bool) -> None:
    """Validate configuration for completeness and correctness.

    Performs comprehensive validation checks including:
    - Required fields presence
    - Value type correctness
    - Value range validity
    - Cross-field consistency
    - Resource availability (in strict mode)

    Examples:
        prevcarga config validate
        prevcarga config validate --strict
        prevcarga config validate --config-file custom.yaml
    """
    # Load configuration
    if config_file:
        config = load_config_file(config_file)
        source = str(config_file)
    else:
        # Try to find default config file
        default_paths = [
            Path("prevcarga.yaml"),
            Path("prevcarga.yml"),
            Path("config/prevcarga.yaml"),
        ]
        config = None
        source = "default"
        for path in default_paths:
            if path.exists():
                config = load_config_file(path)
                source = str(path)
                break

        if config is None:
            click.echo("No configuration file found to validate.")
            return

    # Validate
    validator = ConfigValidator(strict=strict)
    result = validator.validate(config)

    # Display results
    click.echo()
    click.secho("=" * 60, fg="cyan")
    click.secho("  Configuration Validation Results", fg="cyan", bold=True)
    click.secho("=" * 60, fg="cyan")
    click.echo()
    click.echo(f"Source: {source}")
    click.echo(f"Strict mode: {'enabled' if strict else 'disabled'}")
    click.echo(f"Checks performed: {result.checks_performed}")
    click.echo()

    if result.errors:
        click.secho(f"Errors ({len(result.errors)}):", fg="red", bold=True)
        for error in result.errors:
            click.echo(f"  {error}")
        click.echo()

    if result.warnings:
        click.secho(f"Warnings ({len(result.warnings)}):", fg="yellow", bold=True)
        for warning in result.warnings:
            click.echo(f"  {warning}")
        click.echo()

    if result.is_valid:
        click.secho("Configuration is valid", fg="green", bold=True)
    else:
        click.secho("Configuration validation failed", fg="red", bold=True)
        raise SystemExit(1)


@config_cmd.command("init")
@click.option(
    "--template",
    "-t",
    type=click.Choice(["development", "production", "testing"]),
    default="development",
    help="Configuration template to use",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output configuration file path",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing file without confirmation",
)
def init(template: str, output: Path | None, force: bool) -> None:
    """Initialize configuration from template.

    Creates a new configuration file with appropriate defaults
    for the specified environment.

    Templates:
    - development: Local development with debug logging
    - production: Production settings with S3 storage
    - testing: Minimal configuration for tests

    Examples:
        prevcarga config init
        prevcarga config init --template production
        prevcarga config init --output custom-config.yaml
        prevcarga config init --template production --force
    """
    config_path = output or Path("prevcarga.yaml")

    # Check if file exists
    if config_path.exists() and not force:
        if not click.confirm(f"Configuration file {config_path} exists. Overwrite?"):
            click.echo("Operation cancelled")
            return

    # Generate configuration
    template_enum = ConfigTemplate(template)
    config_data = ConfigTemplateGenerator.get_template(template_enum)

    # Write configuration
    click.echo(f"Creating configuration from '{template}' template...")

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    click.echo()
    click.secho(f"Configuration initialized: {config_path}", fg="green", bold=True)
    click.echo(f"  Template: {template}")
    click.echo()

    # Show next steps
    click.secho("=" * 60, fg="cyan")
    click.secho("  Next Steps", fg="cyan", bold=True)
    click.secho("=" * 60, fg="cyan")
    click.echo()
    click.echo("  1. Review and customize the configuration")
    click.echo(f"  2. Validate: prevcarga config validate --config-file {config_path}")
    click.echo(f"  3. Use: prevcarga --config {config_path} <command>")
    click.echo()


@config_cmd.command("diff")
@click.option(
    "--config1",
    "-a",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="First configuration file",
)
@click.option(
    "--config2",
    "-b",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Second configuration file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Save diff to file",
)
def diff(config1: Path, config2: Path, output: Path | None) -> None:
    """Compare two configuration files.

    Shows differences between configuration files,
    useful for comparing environments or tracking changes.

    Examples:
        prevcarga config diff --config1 dev.yaml --config2 prod.yaml
        prevcarga config diff -a old.yaml -b new.yaml --output changes.txt
    """
    # Load configurations
    cfg1 = load_config_file(config1)
    cfg2 = load_config_file(config2)

    # Compare
    comparator = ConfigComparator()
    result = comparator.compare(cfg1, cfg2)

    # Display results
    click.echo()
    click.secho("=" * 60, fg="cyan")
    click.secho("  Configuration Diff", fg="cyan", bold=True)
    click.secho("=" * 60, fg="cyan")
    click.echo()
    click.echo(f"Comparing: {config1.name} vs {config2.name}")
    click.echo()

    output_lines = []

    if result.added:
        click.secho(f"Added ({len(result.added)}):", fg="green", bold=True)
        output_lines.append(f"Added ({len(result.added)}):")
        for item in sorted(result.added):
            click.echo(f"  + {item}")
            output_lines.append(f"  + {item}")
        click.echo()

    if result.removed:
        click.secho(f"Removed ({len(result.removed)}):", fg="red", bold=True)
        output_lines.append(f"Removed ({len(result.removed)}):")
        for item in sorted(result.removed):
            click.echo(f"  - {item}")
            output_lines.append(f"  - {item}")
        click.echo()

    if result.changed:
        click.secho(f"Changed ({len(result.changed)}):", fg="yellow", bold=True)
        output_lines.append(f"Changed ({len(result.changed)}):")
        for item, (old, new) in sorted(result.changed.items()):
            click.echo(f"  ~ {item}: {old} -> {new}")
            output_lines.append(f"  ~ {item}: {old} -> {new}")
        click.echo()

    if not result.has_differences:
        click.secho("Configurations are identical", fg="green", bold=True)
        output_lines.append("Configurations are identical")
    else:
        click.echo(f"Total changes: {result.total_changes}")
        output_lines.append(f"Total changes: {result.total_changes}")

    # Save to file if requested
    if output:
        output.write_text("\n".join(output_lines))
        click.echo()
        click.secho(f"Diff saved to: {output}", fg="green")


@config_cmd.command("export")
@click.option(
    "--format",
    "-f",
    "export_format",
    type=click.Choice(["yaml", "json", "env"]),
    default="yaml",
    help="Export format",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Source configuration file",
)
def export(export_format: str, output: Path | None, config_file: Path | None) -> None:
    """Export configuration to specified format.

    Useful for documentation, backups, or conversion between formats.

    Formats:
    - yaml: YAML format (default)
    - json: JSON format
    - env: Environment variable format (.env file)

    Examples:
        prevcarga config export
        prevcarga config export --format json
        prevcarga config export --format env --output .env
    """
    # Load configuration
    if config_file:
        config = load_config_file(config_file)
    else:
        # Try to find default config file
        default_paths = [
            Path("prevcarga.yaml"),
            Path("prevcarga.yml"),
            Path("config/prevcarga.yaml"),
        ]
        config = None
        for path in default_paths:
            if path.exists():
                config = load_config_file(path)
                break

        if config is None:
            config = ConfigTemplateGenerator.get_template(ConfigTemplate.DEVELOPMENT)
            click.secho("No configuration file found. Exporting default configuration.", fg="yellow")

    # Convert to requested format
    if export_format == "env":
        content = config_to_env_vars(config)
    elif export_format == "json":
        content = config_to_json(config)
    else:
        content = config_to_yaml(config)

    # Output
    if output:
        output.write_text(content)
        click.secho(f"Configuration exported to {output}", fg="green")
    else:
        click.echo(content)


# Export the command group
config = config_cmd
