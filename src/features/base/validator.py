"""Feature validation classes for plugin outputs.

This module provides validation utilities for ensuring plugin outputs
meet quality standards, including column validation, NaN detection,
and feature name validation.

Example:
    ```python
    from src.features.base.validator import FeatureValidator

    validator = FeatureValidator()

    # Validate a single plugin output
    result = validator.validate_output(df, expected_columns=["feature_a", "feature_b"])
    if not result.is_valid:
        print(result.errors)

    # Validate feature names
    report = validator.validate_feature_names(["hour", "day_of_week", "hour"])
    if report.has_duplicates:
        print(report.duplicates)
    ```
"""

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Pattern for valid feature names: letters, digits, underscores; must start with letter
FEATURE_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")


@dataclass
class ValidationResult:
    """Result of a single validation operation.

    Attributes:
        is_valid: Whether the validation passed.
        errors: List of error messages if validation failed.
        warnings: List of warning messages (non-fatal issues).
        details: Additional validation details as key-value pairs.
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error message and mark result as invalid.

        Args:
            message: Error message to add.
        """
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message (does not affect validity).

        Args:
            message: Warning message to add.
        """
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another ValidationResult into this one.

        Args:
            other: ValidationResult to merge.

        Returns:
            Self with merged results.
        """
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.details.update(other.details)
        return self


@dataclass
class ValidationReport:
    """Comprehensive validation report for feature outputs.

    Attributes:
        plugin_name: Name of the plugin that generated the features.
        is_valid: Overall validation status.
        column_validation: Results of column presence validation.
        nan_validation: Results of NaN detection validation.
        name_validation: Results of feature name validation.
        total_rows: Total number of rows in the DataFrame.
        total_columns: Total number of columns validated.
    """

    plugin_name: str = ""
    is_valid: bool = True
    column_validation: ValidationResult = field(default_factory=ValidationResult)
    nan_validation: ValidationResult = field(default_factory=ValidationResult)
    name_validation: ValidationResult = field(default_factory=ValidationResult)
    total_rows: int = 0
    total_columns: int = 0

    def update_validity(self) -> None:
        """Update overall validity based on sub-validations."""
        self.is_valid = (
            self.column_validation.is_valid
            and self.nan_validation.is_valid
            and self.name_validation.is_valid
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary representation.

        Returns:
            Dictionary containing all validation results.
        """
        return {
            "plugin_name": self.plugin_name,
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "column_validation": {
                "is_valid": self.column_validation.is_valid,
                "errors": self.column_validation.errors,
                "warnings": self.column_validation.warnings,
            },
            "nan_validation": {
                "is_valid": self.nan_validation.is_valid,
                "errors": self.nan_validation.errors,
                "warnings": self.nan_validation.warnings,
                "details": self.nan_validation.details,
            },
            "name_validation": {
                "is_valid": self.name_validation.is_valid,
                "errors": self.name_validation.errors,
                "warnings": self.name_validation.warnings,
            },
        }

    def log_summary(self) -> None:
        """Log a summary of the validation report."""
        status = "PASSED" if self.is_valid else "FAILED"
        logger.info(
            "Validation %s for plugin '%s': %d rows, %d columns",
            status,
            self.plugin_name,
            self.total_rows,
            self.total_columns,
        )

        if not self.column_validation.is_valid:
            for error in self.column_validation.errors:
                logger.error("Column validation: %s", error)

        if not self.nan_validation.is_valid:
            for error in self.nan_validation.errors:
                logger.error("NaN validation: %s", error)

        if not self.name_validation.is_valid:
            for error in self.name_validation.errors:
                logger.error("Name validation: %s", error)

        for warning in (
            self.column_validation.warnings
            + self.nan_validation.warnings
            + self.name_validation.warnings
        ):
            logger.warning("Validation warning: %s", warning)


class FeatureValidator:
    """Validator for feature plugin outputs.

    This class provides validation methods for ensuring plugin outputs
    meet quality standards including correct columns, valid feature names,
    and NaN handling.

    Example:
        ```python
        validator = FeatureValidator()

        # Full validation
        report = validator.validate(
            df=output_df,
            plugin_name="temporal",
            expected_columns=["hour", "day_of_week"],
            allow_nan_columns=["optional_feature"]
        )

        if not report.is_valid:
            for error in report.column_validation.errors:
                print(f"Error: {error}")
        ```
    """

    def validate(
        self,
        df: pd.DataFrame,
        plugin_name: str,
        expected_columns: list[str] | None = None,
        allow_nan_columns: list[str] | None = None,
    ) -> ValidationReport:
        """Perform full validation on plugin output DataFrame.

        Args:
            df: DataFrame output from a feature plugin.
            plugin_name: Name of the plugin that generated the output.
            expected_columns: List of columns that must be present.
            allow_nan_columns: List of columns where NaN values are allowed.

        Returns:
            ValidationReport with all validation results.
        """
        report = ValidationReport(
            plugin_name=plugin_name,
            total_rows=len(df),
            total_columns=len(expected_columns) if expected_columns else 0,
        )

        # Validate columns
        if expected_columns:
            report.column_validation = self.validate_columns(df, expected_columns)

        # Validate NaN values
        columns_to_check = expected_columns or list(df.columns)
        report.nan_validation = self.validate_nan(df, columns_to_check, allow_nan_columns or [])

        # Validate feature names
        if expected_columns:
            report.name_validation = self.validate_feature_names(expected_columns)

        report.update_validity()
        return report

    def validate_columns(
        self,
        df: pd.DataFrame,
        expected_columns: list[str],
    ) -> ValidationResult:
        """Validate that expected columns are present in the DataFrame.

        Args:
            df: DataFrame to validate.
            expected_columns: List of column names that must be present.

        Returns:
            ValidationResult with column validation status.
        """
        result = ValidationResult()

        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            result.add_error(f"Missing columns: {sorted(missing_columns)}")

        extra_columns = set(df.columns) - set(expected_columns)
        if extra_columns:
            result.add_warning(f"Extra columns present (not validated): {sorted(extra_columns)}")

        result.details["expected"] = expected_columns
        result.details["present"] = list(df.columns)
        result.details["missing"] = list(missing_columns)

        return result

    def validate_nan(
        self,
        df: pd.DataFrame,
        columns: list[str],
        allow_nan_columns: list[str],
    ) -> ValidationResult:
        """Validate NaN values in specified columns.

        Args:
            df: DataFrame to validate.
            columns: List of columns to check for NaN values.
            allow_nan_columns: List of columns where NaN is allowed.

        Returns:
            ValidationResult with NaN validation status.
        """
        result = ValidationResult()
        nan_counts: dict[str, int] = {}

        for col in columns:
            if col not in df.columns:
                continue

            nan_count = df[col].isna().sum()
            if nan_count > 0:
                nan_counts[col] = int(nan_count)

                if col not in allow_nan_columns:
                    result.add_error(
                        f"Column '{col}' has {nan_count} NaN values "
                        f"({nan_count/len(df)*100:.1f}%)"
                    )
                else:
                    result.add_warning(f"Column '{col}' has {nan_count} NaN values (allowed)")

        result.details["nan_counts"] = nan_counts
        result.details["total_nan"] = sum(nan_counts.values())

        return result

    def validate_feature_names(
        self,
        feature_names: list[str],
    ) -> ValidationResult:
        """Validate feature names for duplicates and valid characters.

        Args:
            feature_names: List of feature names to validate.

        Returns:
            ValidationResult with name validation status.
        """
        result = ValidationResult()

        # Check for duplicates
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in feature_names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)

        if duplicates:
            result.add_error(f"Duplicate feature names: {duplicates}")

        # Check for valid characters
        invalid_names: list[str] = []
        for name in feature_names:
            if not FEATURE_NAME_PATTERN.match(name):
                invalid_names.append(name)

        if invalid_names:
            result.add_error(
                f"Invalid feature names (must match {FEATURE_NAME_PATTERN.pattern}): "
                f"{invalid_names}"
            )

        result.details["total_names"] = len(feature_names)
        result.details["unique_names"] = len(seen)
        result.details["duplicates"] = duplicates
        result.details["invalid_names"] = invalid_names

        return result

    def check_feature_collision(
        self,
        existing_columns: list[str],
        new_columns: list[str],
    ) -> ValidationResult:
        """Check for collisions between existing and new feature names.

        Args:
            existing_columns: List of existing column names.
            new_columns: List of new column names to add.

        Returns:
            ValidationResult with collision information.
        """
        result = ValidationResult()

        collisions = set(existing_columns) & set(new_columns)
        if collisions:
            result.add_error(f"Feature name collisions detected: {sorted(collisions)}")

        result.details["collisions"] = list(collisions)
        result.details["existing_count"] = len(existing_columns)
        result.details["new_count"] = len(new_columns)

        return result
