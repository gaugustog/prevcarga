"""Input validation framework for CLI commands.

This module provides the CLIValidators class with comprehensive validation
methods for all CLI input parameters, ensuring data quality and providing
clear error messages.
"""

from __future__ import annotations

import multiprocessing
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from collections.abc import Sequence


class CLIValidators:
    """Input validation utilities for CLI commands.

    Provides static validation methods for common CLI parameters
    including dates, areas, models, paths, and horizons.

    Attributes:
        MAX_DATE_RANGE_DAYS: Maximum allowed date range (2 years).
        MAX_PARALLEL_WORKERS: Maximum allowed parallel workers.
        VALID_HORIZONS: Valid forecast horizon values (D+0 to D+8).
        VALID_AREAS: Valid area codes for the Brazilian power grid.
        VALID_MODEL_TYPES: Valid model types for training/prediction.
        VALID_OUTPUT_FORMATS: Valid output file formats.
    """

    MAX_DATE_RANGE_DAYS = 730  # 2 years
    MAX_PARALLEL_WORKERS = 32
    VALID_HORIZONS = tuple(range(9))  # D+0 to D+8

    # Valid areas in the Brazilian interconnected grid
    VALID_AREAS = frozenset(
        {
            "nacional",
            "SE",
            "S",
            "NE",
            "N",
            "SECO",
            "SE_CO",
            "ONS_SECO",
            "ONS_S",
            "ONS_NE",
            "ONS_N",
        }
    )

    # Valid model types
    VALID_MODEL_TYPES = frozenset(
        {
            "lgbm",
            "rf",
            "regdin_svm",
            "holt_winters",
            "xgboost",
        }
    )

    # Valid output formats
    VALID_OUTPUT_FORMATS = frozenset(
        {
            "parquet",
            "csv",
            "feather",
            "json",
            "html",
        }
    )

    # Valid combination methods
    VALID_COMBINATION_METHODS = frozenset(
        {
            "simple_avg",
            "weighted_avg",
            "stacking",
            "markov",
        }
    )

    @staticmethod
    def validate_date_range(
        start_date: datetime,
        end_date: datetime,
        max_days: int | None = None,
        allow_future: bool = False,
        confirm_large_range: bool = True,
    ) -> None:
        """Validate date range parameters.

        Args:
            start_date: Start date of the range.
            end_date: End date of the range.
            max_days: Maximum allowed days (default: 730 days / 2 years).
            allow_future: Whether to allow future dates without warning.
            confirm_large_range: Whether to prompt for confirmation on large ranges.

        Raises:
            click.BadParameter: If validation fails.
            click.Abort: If user declines confirmation.
        """
        if start_date >= end_date:
            raise click.BadParameter(
                f"Start date must be before end date. "
                f"Received: {start_date.date()} >= {end_date.date()}"
            )

        days_diff = (end_date - start_date).days
        max_allowed = max_days or CLIValidators.MAX_DATE_RANGE_DAYS

        if days_diff > max_allowed:
            years = max_allowed // 365
            message = (
                f"Date range exceeds maximum of {max_allowed} days "
                f"({years} years). Requested range: {days_diff} days."
            )

            if confirm_large_range:
                if not click.confirm(f"{message}\nContinue anyway?"):
                    raise click.Abort()
                click.echo("Warning: Large date range may take significant time")
            else:
                raise click.BadParameter(message)

        # Warn if date range is in the future
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        if start_date > now and not allow_future:
            if not click.confirm(f"Start date is in the future ({start_date.date()}). Continue?"):
                raise click.Abort()

    @staticmethod
    def validate_area_selection(
        areas: Sequence[str],
        valid_areas: frozenset[str] | None = None,
        require_at_least_one: bool = True,
    ) -> list[str]:
        """Validate area selection against configuration.

        Args:
            areas: List of area codes to validate.
            valid_areas: Set of valid areas (default: VALID_AREAS).
            require_at_least_one: Whether at least one area is required.

        Returns:
            List of validated area codes.

        Raises:
            click.BadParameter: If invalid areas found.
        """
        valid = valid_areas or CLIValidators.VALID_AREAS
        area_list = list(areas)

        if require_at_least_one and not area_list:
            raise click.BadParameter("At least one area must be specified")

        invalid_areas = set(area_list) - valid
        if invalid_areas:
            raise click.BadParameter(
                f"Invalid area(s): {', '.join(sorted(invalid_areas))}. "
                f"Valid areas: {', '.join(sorted(valid))}"
            )

        return area_list

    @staticmethod
    def validate_model_selection(
        models: Sequence[str],
        valid_models: frozenset[str] | None = None,
        require_at_least_one: bool = True,
    ) -> list[str]:
        """Validate model selection against configuration.

        Args:
            models: List of model types to validate.
            valid_models: Set of valid models (default: VALID_MODEL_TYPES).
            require_at_least_one: Whether at least one model is required.

        Returns:
            List of validated model types.

        Raises:
            click.BadParameter: If invalid models found.
        """
        valid = valid_models or CLIValidators.VALID_MODEL_TYPES
        model_list = list(models)

        if require_at_least_one and not model_list:
            raise click.BadParameter("At least one model must be specified")

        invalid_models = set(model_list) - valid
        if invalid_models:
            raise click.BadParameter(
                f"Invalid model(s): {', '.join(sorted(invalid_models))}. "
                f"Valid models: {', '.join(sorted(valid))}"
            )

        return model_list

    @staticmethod
    def validate_model_compatibility(
        models: Sequence[str],
        operation: str,
    ) -> None:
        """Validate model compatibility for specific operations.

        Args:
            models: List of model types.
            operation: Operation type (e.g., 'intraday', 'ensemble').

        Raises:
            click.BadParameter: If incompatible model configuration.
        """
        model_set = set(models)
        min_models_for_ensemble = 2

        if operation == "intraday":
            if "lgbm" not in model_set:
                raise click.BadParameter(
                    "Intraday predictions require LGBM model. " "Add --model lgbm to your command."
                )

        elif operation == "ensemble":
            if len(models) < min_models_for_ensemble:
                raise click.BadParameter(
                    f"Ensemble methods require at least {min_models_for_ensemble} models. "
                    f"Received: {len(models)} model(s)"
                )

    @staticmethod
    def validate_output_path(
        path: Path,
        expected_format: str | None = None,
        overwrite: bool = False,
        create_parents: bool = True,
    ) -> Path:
        """Validate output path and format compatibility.

        Args:
            path: Output file path.
            expected_format: Expected file format (checks extension if provided).
            overwrite: Whether to allow overwrite without confirmation.
            create_parents: Whether to create parent directories if missing.

        Returns:
            Validated path.

        Raises:
            click.BadParameter: If validation fails.
            click.Abort: If user declines confirmation.
        """
        # Check file extension matches format if specified
        if expected_format:
            expected_ext = f".{expected_format}"
            if not path.suffix:
                raise click.BadParameter(
                    f"Output path must include file extension ({expected_ext}). "
                    f"Received: {path}"
                )

            if path.suffix.lower() != expected_ext.lower():
                raise click.BadParameter(
                    f"File extension '{path.suffix}' does not match format '{expected_format}'. "
                    f"Expected: {expected_ext}"
                )

        # Check if file exists
        if path.exists() and not overwrite:
            if not click.confirm(f"File {path} exists. Overwrite?"):
                raise click.Abort()

        # Check if directory exists
        if not path.parent.exists():
            if create_parents:
                if click.confirm(f"Directory {path.parent} does not exist. Create?"):
                    path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    raise click.BadParameter(f"Output directory does not exist: {path.parent}")
            else:
                raise click.BadParameter(f"Output directory does not exist: {path.parent}")

        return path

    @staticmethod
    def validate_horizon_selection(
        horizons: Sequence[int],
        valid_horizons: tuple[int, ...] | None = None,
        require_at_least_one: bool = True,
    ) -> list[int]:
        """Validate forecast horizon selection.

        Args:
            horizons: List of horizon values.
            valid_horizons: Tuple of valid horizons (default: 0-8).
            require_at_least_one: Whether at least one horizon is required.

        Returns:
            List of validated horizons.

        Raises:
            click.BadParameter: If invalid horizons found.
        """
        valid = valid_horizons or CLIValidators.VALID_HORIZONS
        horizon_list = list(horizons)

        if require_at_least_one and not horizon_list:
            raise click.BadParameter("At least one horizon must be specified")

        invalid_horizons = [h for h in horizon_list if h not in valid]
        if invalid_horizons:
            raise click.BadParameter(
                f"Invalid horizon(s): {invalid_horizons}. " f"Valid horizons: 0-8 (D+0 to D+8)"
            )

        # Check for duplicates
        if len(horizon_list) != len(set(horizon_list)):
            duplicates = [h for h in set(horizon_list) if horizon_list.count(h) > 1]
            raise click.BadParameter(f"Duplicate horizons: {duplicates}")

        return sorted(horizon_list)

    @staticmethod
    def validate_parallel_workers(
        workers: int,
        max_workers: int | None = None,
        warn_if_excessive: bool = True,
    ) -> int:
        """Validate parallel worker configuration.

        Args:
            workers: Number of parallel workers.
            max_workers: Maximum allowed workers (default: 32).
            warn_if_excessive: Whether to warn if workers exceed CPU count.

        Returns:
            Validated worker count.

        Raises:
            click.BadParameter: If invalid worker count.
        """
        max_allowed = max_workers or CLIValidators.MAX_PARALLEL_WORKERS

        if workers < 1:
            raise click.BadParameter(f"Parallel workers must be positive. Received: {workers}")

        if workers > max_allowed:
            raise click.BadParameter(
                f"Parallel workers exceeds maximum ({max_allowed}). " f"Received: {workers}"
            )

        # Warn if too many workers relative to CPU count
        if warn_if_excessive:
            cpu_count = multiprocessing.cpu_count()
            excessive_threshold = cpu_count * 2
            if workers > excessive_threshold:
                click.echo(
                    f"Warning: {workers} workers requested but only {cpu_count} CPUs available. "
                    "Performance may be degraded."
                )

        return workers

    @staticmethod
    def validate_config_file(config_path: Path) -> Path:
        """Validate configuration file exists and is readable.

        Args:
            config_path: Path to configuration file.

        Returns:
            Validated path.

        Raises:
            click.BadParameter: If file not found or unreadable.
        """
        if not config_path.exists():
            raise click.BadParameter(
                f"Configuration file not found: {config_path}. "
                "Use 'prevcarga config init' to create a new configuration."
            )

        if not config_path.is_file():
            raise click.BadParameter(f"Path is not a file: {config_path}")

        # Check file is readable
        try:
            with open(config_path) as f:
                f.read(1)
        except PermissionError as e:
            raise click.BadParameter(f"Permission denied reading: {config_path}") from e
        except OSError as e:
            raise click.BadParameter(f"Cannot read configuration file: {e}") from e

        return config_path

    @staticmethod
    def validate_combination_method(
        method: str,
        models: Sequence[str],
        valid_methods: frozenset[str] | None = None,
    ) -> str:
        """Validate combination method compatibility.

        Args:
            method: Combination method.
            models: List of models.
            valid_methods: Set of valid methods (default: VALID_COMBINATION_METHODS).

        Returns:
            Validated method.

        Raises:
            click.BadParameter: If incompatible configuration.
        """
        valid = valid_methods or CLIValidators.VALID_COMBINATION_METHODS

        if method not in valid:
            raise click.BadParameter(
                f"Invalid combination method: {method}. "
                f"Valid methods: {', '.join(sorted(valid))}"
            )

        min_models = 2
        if method in {"weighted_avg", "stacking", "markov"}:
            if len(models) < min_models:
                raise click.BadParameter(
                    f"Combination method '{method}' requires multiple models. "
                    f"Received: {len(models)} model(s)"
                )

        return method

    @staticmethod
    def validate_reconciliation_method(
        method: str,
        valid_methods: frozenset[str] | None = None,
    ) -> str:
        """Validate reconciliation method.

        Args:
            method: Reconciliation method name.
            valid_methods: Set of valid methods.

        Returns:
            Validated method.

        Raises:
            click.BadParameter: If invalid method.
        """
        valid = valid_methods or frozenset({"mint", "ols", "wls", "shrinkage"})

        if method not in valid:
            raise click.BadParameter(
                f"Invalid reconciliation method: {method}. "
                f"Valid methods: {', '.join(sorted(valid))}"
            )

        return method

    @staticmethod
    def validate_output_format(
        format_str: str,
        valid_formats: frozenset[str] | None = None,
    ) -> str:
        """Validate output format.

        Args:
            format_str: Output format string.
            valid_formats: Set of valid formats.

        Returns:
            Validated format string.

        Raises:
            click.BadParameter: If invalid format.
        """
        valid = valid_formats or CLIValidators.VALID_OUTPUT_FORMATS

        if format_str not in valid:
            raise click.BadParameter(
                f"Invalid format: {format_str}. " f"Valid formats: {', '.join(sorted(valid))}"
            )

        return format_str


# Custom Click parameter types for common validations


class DateParamType(click.ParamType):
    """Custom parameter type for date validation.

    Accepts dates in YYYY-MM-DD format.
    """

    name = "date"

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> datetime:
        """Convert string to datetime.

        Args:
            value: Input value.
            param: Click parameter.
            ctx: Click context.

        Returns:
            Parsed datetime.

        Raises:
            click.BadParameter: If parsing fails.
        """
        if isinstance(value, datetime):
            return value

        try:
            return datetime.strptime(str(value), "%Y-%m-%d")
        except ValueError:
            self.fail(
                f"Invalid date format: {value}. Expected: YYYY-MM-DD",
                param,
                ctx,
            )


class AreaParamType(click.ParamType):
    """Custom parameter type for area validation.

    Validates area codes against the valid areas list.
    """

    name = "area"

    def __init__(self, valid_areas: frozenset[str] | None = None) -> None:
        """Initialize with valid areas.

        Args:
            valid_areas: Set of valid area codes.
        """
        self.valid_areas = valid_areas or CLIValidators.VALID_AREAS

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> str:
        """Validate and return area code.

        Args:
            value: Input value.
            param: Click parameter.
            ctx: Click context.

        Returns:
            Validated area code.

        Raises:
            click.BadParameter: If invalid area.
        """
        area = str(value)
        if area not in self.valid_areas:
            self.fail(
                f"Invalid area: {area}. " f"Valid areas: {', '.join(sorted(self.valid_areas))}",
                param,
                ctx,
            )
        return area


class HorizonParamType(click.ParamType):
    """Custom parameter type for horizon validation.

    Validates horizon values (0-8 for D+0 to D+8).
    """

    name = "horizon"

    def __init__(self, valid_horizons: tuple[int, ...] | None = None) -> None:
        """Initialize with valid horizons.

        Args:
            valid_horizons: Tuple of valid horizon values.
        """
        self.valid_horizons = valid_horizons or CLIValidators.VALID_HORIZONS

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> int:
        """Validate and return horizon value.

        Args:
            value: Input value.
            param: Click parameter.
            ctx: Click context.

        Returns:
            Validated horizon value.

        Raises:
            click.BadParameter: If invalid horizon.
        """
        try:
            horizon = int(value)
        except (ValueError, TypeError):
            self.fail(f"Invalid horizon value: {value}. Must be an integer.", param, ctx)

        if horizon not in self.valid_horizons:
            self.fail(
                f"Invalid horizon: {horizon}. " f"Valid horizons: 0-8 (D+0 to D+8)",
                param,
                ctx,
            )
        return horizon


class ModelParamType(click.ParamType):
    """Custom parameter type for model type validation.

    Validates model type codes against the valid models list.
    """

    name = "model"

    def __init__(self, valid_models: frozenset[str] | None = None) -> None:
        """Initialize with valid models.

        Args:
            valid_models: Set of valid model type codes.
        """
        self.valid_models = valid_models or CLIValidators.VALID_MODEL_TYPES

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> str:
        """Validate and return model type.

        Args:
            value: Input value.
            param: Click parameter.
            ctx: Click context.

        Returns:
            Validated model type.

        Raises:
            click.BadParameter: If invalid model.
        """
        model = str(value).lower()
        if model not in self.valid_models:
            self.fail(
                f"Invalid model: {model}. " f"Valid models: {', '.join(sorted(self.valid_models))}",
                param,
                ctx,
            )
        return model


class PathWithExtensionParamType(click.ParamType):
    """Custom parameter type for path validation with extension check.

    Validates file paths and optionally enforces specific extensions.
    """

    name = "path"

    def __init__(self, expected_extension: str | None = None) -> None:
        """Initialize with expected extension.

        Args:
            expected_extension: Expected file extension (e.g., ".csv").
        """
        self.expected_extension = expected_extension

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> Path:
        """Validate and return path.

        Args:
            value: Input value.
            param: Click parameter.
            ctx: Click context.

        Returns:
            Validated Path object.

        Raises:
            click.BadParameter: If validation fails.
        """
        path = Path(str(value))

        if self.expected_extension:
            if not path.suffix:
                self.fail(
                    f"Path must have extension {self.expected_extension}",
                    param,
                    ctx,
                )
            if path.suffix.lower() != self.expected_extension.lower():
                self.fail(
                    f"Expected extension {self.expected_extension}, got {path.suffix}",
                    param,
                    ctx,
                )

        return path


# Convenience instances
DATE = DateParamType()
AREA = AreaParamType()
HORIZON = HorizonParamType()
MODEL = ModelParamType()
