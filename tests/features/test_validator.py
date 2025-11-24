"""Tests for FeatureValidator class.

This module contains comprehensive tests for the FeatureValidator class,
including column validation, NaN detection, and feature name validation.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.base.validator import (
    FEATURE_NAME_PATTERN,
    FeatureValidator,
    ValidationReport,
    ValidationResult,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_values(self):
        """Test ValidationResult default values."""
        result = ValidationResult()

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.details == {}

    def test_add_error(self):
        """Test adding an error invalidates result."""
        result = ValidationResult()
        result.add_error("Test error")

        assert result.is_valid is False
        assert "Test error" in result.errors

    def test_add_multiple_errors(self):
        """Test adding multiple errors."""
        result = ValidationResult()
        result.add_error("Error 1")
        result.add_error("Error 2")

        assert result.is_valid is False
        assert len(result.errors) == 2

    def test_add_warning(self):
        """Test adding warning does not invalidate result."""
        result = ValidationResult()
        result.add_warning("Test warning")

        assert result.is_valid is True
        assert "Test warning" in result.warnings

    def test_merge_results(self):
        """Test merging two ValidationResults."""
        result1 = ValidationResult()
        result1.add_error("Error from result1")
        result1.details["key1"] = "value1"

        result2 = ValidationResult()
        result2.add_warning("Warning from result2")
        result2.details["key2"] = "value2"

        result1.merge(result2)

        assert result1.is_valid is False
        assert "Error from result1" in result1.errors
        assert "Warning from result2" in result1.warnings
        assert result1.details["key1"] == "value1"
        assert result1.details["key2"] == "value2"

    def test_merge_invalid_into_valid(self):
        """Test merging invalid result into valid makes invalid."""
        result1 = ValidationResult()  # Valid
        result2 = ValidationResult()
        result2.add_error("Error")  # Invalid

        result1.merge(result2)

        assert result1.is_valid is False


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_default_values(self):
        """Test ValidationReport default values."""
        report = ValidationReport()

        assert report.plugin_name == ""
        assert report.is_valid is True
        assert report.total_rows == 0
        assert report.total_columns == 0

    def test_update_validity(self):
        """Test update_validity aggregates sub-validations."""
        report = ValidationReport()
        report.column_validation.add_error("Column error")

        report.update_validity()

        assert report.is_valid is False

    def test_update_validity_all_valid(self):
        """Test update_validity when all sub-validations pass."""
        report = ValidationReport()
        # All sub-validations are valid by default

        report.update_validity()

        assert report.is_valid is True

    def test_to_dict(self):
        """Test to_dict returns proper structure."""
        report = ValidationReport(plugin_name="test_plugin", total_rows=100)
        report.column_validation.add_error("Missing column")

        result = report.to_dict()

        assert result["plugin_name"] == "test_plugin"
        assert result["total_rows"] == 100
        assert "column_validation" in result
        assert result["column_validation"]["is_valid"] is False

    def test_log_summary_valid_executes(self):
        """Test log_summary executes without error for valid report."""
        report = ValidationReport(plugin_name="log_test", total_rows=50)

        # Should not raise any exception
        report.log_summary()

        # Verify report state unchanged
        assert report.is_valid is True
        assert report.plugin_name == "log_test"

    def test_log_summary_invalid_executes(self):
        """Test log_summary executes without error for invalid report."""
        report = ValidationReport(plugin_name="log_test_invalid")
        report.column_validation.add_error("Test error")
        report.update_validity()

        # Should not raise any exception
        report.log_summary()

        # Verify report state unchanged
        assert report.is_valid is False
        assert report.plugin_name == "log_test_invalid"


class TestFeatureValidator:
    """Tests for FeatureValidator class."""

    @pytest.fixture
    def validator(self) -> FeatureValidator:
        """Create a FeatureValidator instance."""
        return FeatureValidator()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0],
                "feature_b": [4.0, 5.0, 6.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )


class TestFeatureValidatorValidate:
    """Tests for validate() method."""

    @pytest.fixture
    def validator(self) -> FeatureValidator:
        """Create a FeatureValidator instance."""
        return FeatureValidator()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0],
                "feature_b": [4.0, 5.0, 6.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )

    def test_validate_all_valid(
        self, validator: FeatureValidator, sample_df: pd.DataFrame
    ):
        """Test validate with all valid data."""
        report = validator.validate(
            sample_df,
            plugin_name="test_plugin",
            expected_columns=["feature_a", "feature_b"],
        )

        assert report.is_valid is True
        assert report.plugin_name == "test_plugin"
        assert report.total_rows == 3

    def test_validate_missing_columns(
        self, validator: FeatureValidator, sample_df: pd.DataFrame
    ):
        """Test validate with missing columns."""
        report = validator.validate(
            sample_df,
            plugin_name="test_plugin",
            expected_columns=["feature_a", "feature_c"],
        )

        assert report.is_valid is False
        assert report.column_validation.is_valid is False
        assert any("feature_c" in err for err in report.column_validation.errors)

    def test_validate_with_nan(self, validator: FeatureValidator):
        """Test validate detects NaN values."""
        df = pd.DataFrame(
            {
                "feature_a": [1.0, np.nan, 3.0],
                "feature_b": [4.0, 5.0, 6.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )

        report = validator.validate(
            df,
            plugin_name="test_plugin",
            expected_columns=["feature_a", "feature_b"],
        )

        assert report.nan_validation.is_valid is False

    def test_validate_with_allowed_nan(self, validator: FeatureValidator):
        """Test validate allows NaN in specified columns."""
        df = pd.DataFrame(
            {
                "feature_a": [1.0, np.nan, 3.0],
                "feature_b": [4.0, 5.0, 6.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )

        report = validator.validate(
            df,
            plugin_name="test_plugin",
            expected_columns=["feature_a", "feature_b"],
            allow_nan_columns=["feature_a"],
        )

        # NaN is allowed in feature_a, so validation should pass
        assert report.nan_validation.is_valid is True


class TestFeatureValidatorValidateColumns:
    """Tests for validate_columns() method."""

    @pytest.fixture
    def validator(self) -> FeatureValidator:
        """Create a FeatureValidator instance."""
        return FeatureValidator()

    def test_validate_columns_all_present(self, validator: FeatureValidator):
        """Test validate_columns when all expected columns present."""
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})

        result = validator.validate_columns(df, ["col_a", "col_b"])

        assert result.is_valid is True
        assert result.errors == []

    def test_validate_columns_missing(self, validator: FeatureValidator):
        """Test validate_columns when columns are missing."""
        df = pd.DataFrame({"col_a": [1]})

        result = validator.validate_columns(df, ["col_a", "col_b"])

        assert result.is_valid is False
        assert any("col_b" in err for err in result.errors)

    def test_validate_columns_extra_warning(self, validator: FeatureValidator):
        """Test validate_columns warns about extra columns."""
        df = pd.DataFrame({"col_a": [1], "col_b": [2], "col_c": [3]})

        result = validator.validate_columns(df, ["col_a", "col_b"])

        # Extra columns are warnings, not errors
        assert result.is_valid is True
        assert any("col_c" in warn for warn in result.warnings)

    def test_validate_columns_details(self, validator: FeatureValidator):
        """Test validate_columns populates details."""
        df = pd.DataFrame({"col_a": [1]})

        result = validator.validate_columns(df, ["col_a", "col_b"])

        assert "expected" in result.details
        assert "missing" in result.details
        assert "col_b" in result.details["missing"]


class TestFeatureValidatorValidateNan:
    """Tests for validate_nan() method."""

    @pytest.fixture
    def validator(self) -> FeatureValidator:
        """Create a FeatureValidator instance."""
        return FeatureValidator()

    def test_validate_nan_no_nan(self, validator: FeatureValidator):
        """Test validate_nan with no NaN values."""
        df = pd.DataFrame({"col_a": [1.0, 2.0], "col_b": [3.0, 4.0]})

        result = validator.validate_nan(df, ["col_a", "col_b"], [])

        assert result.is_valid is True
        assert result.details["total_nan"] == 0

    def test_validate_nan_with_nan(self, validator: FeatureValidator):
        """Test validate_nan detects NaN values."""
        df = pd.DataFrame({"col_a": [1.0, np.nan], "col_b": [3.0, 4.0]})

        result = validator.validate_nan(df, ["col_a", "col_b"], [])

        assert result.is_valid is False
        assert result.details["nan_counts"]["col_a"] == 1

    def test_validate_nan_allowed(self, validator: FeatureValidator):
        """Test validate_nan allows NaN in specified columns."""
        df = pd.DataFrame({"col_a": [1.0, np.nan], "col_b": [3.0, 4.0]})

        result = validator.validate_nan(df, ["col_a", "col_b"], ["col_a"])

        assert result.is_valid is True
        assert any("allowed" in warn for warn in result.warnings)

    def test_validate_nan_missing_column(self, validator: FeatureValidator):
        """Test validate_nan handles missing columns gracefully."""
        df = pd.DataFrame({"col_a": [1.0, 2.0]})

        result = validator.validate_nan(df, ["col_a", "col_missing"], [])

        # Should not error, just skip missing column
        assert result.is_valid is True


class TestFeatureValidatorValidateFeatureNames:
    """Tests for validate_feature_names() method."""

    @pytest.fixture
    def validator(self) -> FeatureValidator:
        """Create a FeatureValidator instance."""
        return FeatureValidator()

    def test_validate_feature_names_valid(self, validator: FeatureValidator):
        """Test validate_feature_names with valid names."""
        result = validator.validate_feature_names(["feature_a", "feature_b"])

        assert result.is_valid is True

    def test_validate_feature_names_duplicates(self, validator: FeatureValidator):
        """Test validate_feature_names detects duplicates."""
        result = validator.validate_feature_names(["feature_a", "feature_a"])

        assert result.is_valid is False
        assert any("Duplicate" in err for err in result.errors)
        assert "feature_a" in result.details["duplicates"]

    def test_validate_feature_names_invalid_chars(self, validator: FeatureValidator):
        """Test validate_feature_names detects invalid characters."""
        result = validator.validate_feature_names(["feature-a", "feature.b"])

        assert result.is_valid is False
        assert any("Invalid" in err for err in result.errors)

    def test_validate_feature_names_starts_with_number(
        self, validator: FeatureValidator
    ):
        """Test validate_feature_names detects names starting with number."""
        result = validator.validate_feature_names(["1feature", "2_feature"])

        assert result.is_valid is False
        assert "1feature" in result.details["invalid_names"]

    def test_validate_feature_names_empty_list(self, validator: FeatureValidator):
        """Test validate_feature_names with empty list."""
        result = validator.validate_feature_names([])

        assert result.is_valid is True


class TestFeatureValidatorCheckCollision:
    """Tests for check_feature_collision() method."""

    @pytest.fixture
    def validator(self) -> FeatureValidator:
        """Create a FeatureValidator instance."""
        return FeatureValidator()

    def test_check_collision_no_collision(self, validator: FeatureValidator):
        """Test check_feature_collision with no collisions."""
        result = validator.check_feature_collision(
            existing_columns=["col_a", "col_b"],
            new_columns=["col_c", "col_d"],
        )

        assert result.is_valid is True
        assert result.details["collisions"] == []

    def test_check_collision_with_collision(self, validator: FeatureValidator):
        """Test check_feature_collision detects collisions."""
        result = validator.check_feature_collision(
            existing_columns=["col_a", "col_b"],
            new_columns=["col_b", "col_c"],
        )

        assert result.is_valid is False
        assert "col_b" in result.details["collisions"]

    def test_check_collision_multiple_collisions(self, validator: FeatureValidator):
        """Test check_feature_collision detects multiple collisions."""
        result = validator.check_feature_collision(
            existing_columns=["col_a", "col_b", "col_c"],
            new_columns=["col_a", "col_c", "col_d"],
        )

        assert result.is_valid is False
        assert len(result.details["collisions"]) == 2


class TestFeatureNamePattern:
    """Tests for FEATURE_NAME_PATTERN regex."""

    @pytest.mark.parametrize(
        "valid_name",
        [
            "feature",
            "feature_a",
            "Feature123",
            "f",
            "F",
            "feature_a_b_c",
            "hourOfDay",
            "day_of_week_encoded",
        ],
    )
    def test_valid_feature_names(self, valid_name: str):
        """Test that valid feature names match pattern."""
        assert FEATURE_NAME_PATTERN.match(valid_name) is not None

    @pytest.mark.parametrize(
        "invalid_name",
        [
            "1feature",  # starts with number
            "_feature",  # starts with underscore
            "feature-a",  # hyphen
            "feature.a",  # dot
            "feature a",  # space
            "",  # empty
            "123",  # only numbers
        ],
    )
    def test_invalid_feature_names(self, invalid_name: str):
        """Test that invalid feature names don't match pattern."""
        assert FEATURE_NAME_PATTERN.match(invalid_name) is None
