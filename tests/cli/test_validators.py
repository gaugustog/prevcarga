"""Tests for CLI input validation framework.

This module contains comprehensive tests for the CLIValidators class
and custom Click parameter types.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import click
import pytest

from src.cli.validators import (
    AREA,
    DATE,
    HORIZON,
    MODEL,
    AreaParamType,
    CLIValidators,
    DateParamType,
    HorizonParamType,
    ModelParamType,
    PathWithExtensionParamType,
)


class TestCLIValidatorsDateRange:
    """Tests for validate_date_range method."""

    def test_valid_date_range(self) -> None:
        """Test valid date range passes validation."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        # Should not raise
        CLIValidators.validate_date_range(start, end, confirm_large_range=False)

    def test_start_after_end_raises(self) -> None:
        """Test date range with start after end raises error."""
        start = datetime(2023, 12, 31)
        end = datetime(2023, 1, 1)

        with pytest.raises(click.BadParameter, match="Start date must be before"):
            CLIValidators.validate_date_range(start, end)

    def test_same_dates_raises(self) -> None:
        """Test date range with same dates raises error."""
        date = datetime(2023, 6, 15)

        with pytest.raises(click.BadParameter, match="Start date must be before"):
            CLIValidators.validate_date_range(date, date)

    def test_large_range_without_confirm_raises(self) -> None:
        """Test large date range without confirmation raises error."""
        start = datetime(2020, 1, 1)
        end = datetime(2023, 1, 1)  # 3 years

        with pytest.raises(click.BadParameter, match="exceeds maximum"):
            CLIValidators.validate_date_range(start, end, confirm_large_range=False)

    @patch("click.confirm", return_value=False)
    def test_large_range_with_decline_aborts(self, mock_confirm: object) -> None:
        """Test large date range with declined confirmation aborts."""
        start = datetime(2020, 1, 1)
        end = datetime(2023, 1, 1)

        with pytest.raises(click.Abort):
            CLIValidators.validate_date_range(start, end, confirm_large_range=True)

    @patch("click.confirm", return_value=True)
    @patch("click.echo")
    def test_large_range_with_confirm_proceeds(
        self, mock_echo: object, mock_confirm: object
    ) -> None:
        """Test large date range with confirmation proceeds."""
        start = datetime(2020, 1, 1)
        end = datetime(2023, 1, 1)

        # Should not raise
        CLIValidators.validate_date_range(start, end, confirm_large_range=True)

    def test_custom_max_days(self) -> None:
        """Test custom max_days parameter."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 2, 1)  # 31 days

        with pytest.raises(click.BadParameter, match="exceeds maximum"):
            CLIValidators.validate_date_range(
                start, end, max_days=30, confirm_large_range=False
            )

    @patch("click.confirm", return_value=False)
    def test_future_date_with_decline_aborts(self, mock_confirm: object) -> None:
        """Test future start date with declined confirmation aborts."""
        future = datetime.now() + timedelta(days=30)
        end = future + timedelta(days=10)

        with pytest.raises(click.Abort):
            CLIValidators.validate_date_range(future, end, allow_future=False)

    def test_future_date_allowed(self) -> None:
        """Test future start date allowed when flag is set."""
        future = datetime.now() + timedelta(days=30)
        end = future + timedelta(days=10)

        # Should not raise
        CLIValidators.validate_date_range(future, end, allow_future=True)


class TestCLIValidatorsAreaSelection:
    """Tests for validate_area_selection method."""

    def test_valid_areas(self) -> None:
        """Test valid areas pass validation."""
        areas = ["SE", "S", "NE"]
        result = CLIValidators.validate_area_selection(areas)
        assert result == ["SE", "S", "NE"]

    def test_invalid_areas_raises(self) -> None:
        """Test invalid areas raise error."""
        areas = ["SE", "INVALID"]

        with pytest.raises(click.BadParameter, match="Invalid area"):
            CLIValidators.validate_area_selection(areas)

    def test_empty_areas_raises(self) -> None:
        """Test empty areas list raises error."""
        with pytest.raises(click.BadParameter, match="At least one area"):
            CLIValidators.validate_area_selection([])

    def test_empty_areas_allowed(self) -> None:
        """Test empty areas allowed when flag is set."""
        result = CLIValidators.validate_area_selection([], require_at_least_one=False)
        assert result == []

    def test_custom_valid_areas(self) -> None:
        """Test custom valid areas set."""
        custom_areas = frozenset({"A", "B", "C"})
        result = CLIValidators.validate_area_selection(["A", "B"], valid_areas=custom_areas)
        assert result == ["A", "B"]

    def test_all_valid_areas(self) -> None:
        """Test all predefined valid areas pass."""
        for area in CLIValidators.VALID_AREAS:
            result = CLIValidators.validate_area_selection([area])
            assert result == [area]


class TestCLIValidatorsModelSelection:
    """Tests for validate_model_selection method."""

    def test_valid_models(self) -> None:
        """Test valid models pass validation."""
        models = ["lgbm", "rf"]
        result = CLIValidators.validate_model_selection(models)
        assert result == ["lgbm", "rf"]

    def test_invalid_models_raises(self) -> None:
        """Test invalid models raise error."""
        models = ["lgbm", "invalid_model"]

        with pytest.raises(click.BadParameter, match="Invalid model"):
            CLIValidators.validate_model_selection(models)

    def test_empty_models_raises(self) -> None:
        """Test empty models list raises error."""
        with pytest.raises(click.BadParameter, match="At least one model"):
            CLIValidators.validate_model_selection([])

    def test_empty_models_allowed(self) -> None:
        """Test empty models allowed when flag is set."""
        result = CLIValidators.validate_model_selection([], require_at_least_one=False)
        assert result == []


class TestCLIValidatorsModelCompatibility:
    """Tests for validate_model_compatibility method."""

    def test_intraday_requires_lgbm(self) -> None:
        """Test intraday operation requires LGBM model."""
        with pytest.raises(click.BadParameter, match="require LGBM"):
            CLIValidators.validate_model_compatibility(["rf"], "intraday")

    def test_intraday_with_lgbm_passes(self) -> None:
        """Test intraday operation with LGBM passes."""
        # Should not raise
        CLIValidators.validate_model_compatibility(["lgbm", "rf"], "intraday")

    def test_ensemble_requires_multiple_models(self) -> None:
        """Test ensemble operation requires multiple models."""
        with pytest.raises(click.BadParameter, match="at least 2 models"):
            CLIValidators.validate_model_compatibility(["lgbm"], "ensemble")

    def test_ensemble_with_multiple_models_passes(self) -> None:
        """Test ensemble operation with multiple models passes."""
        # Should not raise
        CLIValidators.validate_model_compatibility(["lgbm", "rf"], "ensemble")


class TestCLIValidatorsOutputPath:
    """Tests for validate_output_path method."""

    def test_valid_path(self, tmp_path: Path) -> None:
        """Test valid path passes validation."""
        path = tmp_path / "output.csv"
        result = CLIValidators.validate_output_path(path, expected_format="csv")
        assert result == path

    def test_no_extension_raises(self, tmp_path: Path) -> None:
        """Test path without extension raises error."""
        path = tmp_path / "output"

        with pytest.raises(click.BadParameter, match="must include file extension"):
            CLIValidators.validate_output_path(path, expected_format="csv")

    def test_wrong_extension_raises(self, tmp_path: Path) -> None:
        """Test path with wrong extension raises error."""
        path = tmp_path / "output.json"

        with pytest.raises(click.BadParameter, match="does not match format"):
            CLIValidators.validate_output_path(path, expected_format="csv")

    @patch("click.confirm", return_value=False)
    def test_existing_file_no_overwrite_aborts(
        self, mock_confirm: object, tmp_path: Path
    ) -> None:
        """Test existing file without overwrite permission aborts."""
        path = tmp_path / "output.csv"
        path.touch()

        with pytest.raises(click.Abort):
            CLIValidators.validate_output_path(path, overwrite=False)

    def test_existing_file_with_overwrite_flag(self, tmp_path: Path) -> None:
        """Test existing file with overwrite flag passes."""
        path = tmp_path / "output.csv"
        path.touch()

        result = CLIValidators.validate_output_path(path, overwrite=True)
        assert result == path

    @patch("click.confirm", return_value=True)
    def test_missing_directory_creates(
        self, mock_confirm: object, tmp_path: Path
    ) -> None:
        """Test missing directory is created when confirmed."""
        path = tmp_path / "new_dir" / "output.csv"

        result = CLIValidators.validate_output_path(path, create_parents=True)
        assert result == path
        assert path.parent.exists()

    @patch("click.confirm", return_value=False)
    def test_missing_directory_no_create_raises(
        self, mock_confirm: object, tmp_path: Path
    ) -> None:
        """Test missing directory without creation permission raises."""
        path = tmp_path / "new_dir" / "output.csv"

        with pytest.raises(click.BadParameter, match="does not exist"):
            CLIValidators.validate_output_path(path, create_parents=True)


class TestCLIValidatorsHorizonSelection:
    """Tests for validate_horizon_selection method."""

    def test_valid_horizons(self) -> None:
        """Test valid horizons pass validation."""
        horizons = [0, 1, 2, 8]
        result = CLIValidators.validate_horizon_selection(horizons)
        assert result == [0, 1, 2, 8]

    def test_horizons_sorted(self) -> None:
        """Test horizons are sorted."""
        horizons = [5, 2, 8, 1]
        result = CLIValidators.validate_horizon_selection(horizons)
        assert result == [1, 2, 5, 8]

    def test_invalid_horizons_raises(self) -> None:
        """Test invalid horizons raise error."""
        horizons = [0, 1, 10]  # 10 is invalid

        with pytest.raises(click.BadParameter, match="Invalid horizon"):
            CLIValidators.validate_horizon_selection(horizons)

    def test_negative_horizon_raises(self) -> None:
        """Test negative horizon raises error."""
        with pytest.raises(click.BadParameter, match="Invalid horizon"):
            CLIValidators.validate_horizon_selection([-1, 0, 1])

    def test_duplicate_horizons_raises(self) -> None:
        """Test duplicate horizons raise error."""
        with pytest.raises(click.BadParameter, match="Duplicate horizons"):
            CLIValidators.validate_horizon_selection([0, 1, 1, 2])

    def test_empty_horizons_raises(self) -> None:
        """Test empty horizons list raises error."""
        with pytest.raises(click.BadParameter, match="At least one horizon"):
            CLIValidators.validate_horizon_selection([])

    def test_empty_horizons_allowed(self) -> None:
        """Test empty horizons allowed when flag is set."""
        result = CLIValidators.validate_horizon_selection([], require_at_least_one=False)
        assert result == []


class TestCLIValidatorsParallelWorkers:
    """Tests for validate_parallel_workers method."""

    def test_valid_workers(self) -> None:
        """Test valid worker count passes."""
        result = CLIValidators.validate_parallel_workers(4)
        assert result == 4

    def test_zero_workers_raises(self) -> None:
        """Test zero workers raises error."""
        with pytest.raises(click.BadParameter, match="must be positive"):
            CLIValidators.validate_parallel_workers(0)

    def test_negative_workers_raises(self) -> None:
        """Test negative workers raises error."""
        with pytest.raises(click.BadParameter, match="must be positive"):
            CLIValidators.validate_parallel_workers(-1)

    def test_too_many_workers_raises(self) -> None:
        """Test too many workers raises error."""
        with pytest.raises(click.BadParameter, match="exceeds maximum"):
            CLIValidators.validate_parallel_workers(100)

    def test_custom_max_workers(self) -> None:
        """Test custom max_workers parameter."""
        with pytest.raises(click.BadParameter, match="exceeds maximum"):
            CLIValidators.validate_parallel_workers(5, max_workers=4)

    @patch("click.echo")
    def test_warns_excessive_workers(self, mock_echo: object) -> None:
        """Test warning when workers exceed CPU count * 2."""
        # Use a high number but within max limit
        CLIValidators.validate_parallel_workers(30, warn_if_excessive=True)
        # mock_echo should have been called with warning
        # (depends on actual CPU count)

    def test_no_warn_when_disabled(self) -> None:
        """Test no warning when warn_if_excessive is False."""
        # Should not raise or print warning
        result = CLIValidators.validate_parallel_workers(30, warn_if_excessive=False)
        assert result == 30


class TestCLIValidatorsConfigFile:
    """Tests for validate_config_file method."""

    def test_valid_config_file(self, tmp_path: Path) -> None:
        """Test valid config file passes validation."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("key: value")

        result = CLIValidators.validate_config_file(config_path)
        assert result == config_path

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Test missing file raises error."""
        config_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(click.BadParameter, match="not found"):
            CLIValidators.validate_config_file(config_path)

    def test_directory_raises(self, tmp_path: Path) -> None:
        """Test directory instead of file raises error."""
        with pytest.raises(click.BadParameter, match="not a file"):
            CLIValidators.validate_config_file(tmp_path)


class TestCLIValidatorsCombinationMethod:
    """Tests for validate_combination_method method."""

    def test_valid_simple_avg(self) -> None:
        """Test simple_avg method passes."""
        result = CLIValidators.validate_combination_method("simple_avg", ["lgbm"])
        assert result == "simple_avg"

    def test_invalid_method_raises(self) -> None:
        """Test invalid method raises error."""
        with pytest.raises(click.BadParameter, match="Invalid combination method"):
            CLIValidators.validate_combination_method("invalid", ["lgbm"])

    def test_weighted_avg_requires_multiple_models(self) -> None:
        """Test weighted_avg requires multiple models."""
        with pytest.raises(click.BadParameter, match="requires multiple models"):
            CLIValidators.validate_combination_method("weighted_avg", ["lgbm"])

    def test_stacking_requires_multiple_models(self) -> None:
        """Test stacking requires multiple models."""
        with pytest.raises(click.BadParameter, match="requires multiple models"):
            CLIValidators.validate_combination_method("stacking", ["lgbm"])

    def test_markov_requires_multiple_models(self) -> None:
        """Test markov requires multiple models."""
        with pytest.raises(click.BadParameter, match="requires multiple models"):
            CLIValidators.validate_combination_method("markov", ["lgbm"])

    def test_weighted_avg_with_multiple_models(self) -> None:
        """Test weighted_avg with multiple models passes."""
        result = CLIValidators.validate_combination_method(
            "weighted_avg", ["lgbm", "rf"]
        )
        assert result == "weighted_avg"


class TestCLIValidatorsReconciliationMethod:
    """Tests for validate_reconciliation_method method."""

    def test_valid_methods(self) -> None:
        """Test valid reconciliation methods pass."""
        for method in ["mint", "ols", "wls", "shrinkage"]:
            result = CLIValidators.validate_reconciliation_method(method)
            assert result == method

    def test_invalid_method_raises(self) -> None:
        """Test invalid method raises error."""
        with pytest.raises(click.BadParameter, match="Invalid reconciliation method"):
            CLIValidators.validate_reconciliation_method("invalid")


class TestCLIValidatorsOutputFormat:
    """Tests for validate_output_format method."""

    def test_valid_formats(self) -> None:
        """Test valid output formats pass."""
        for fmt in ["parquet", "csv", "feather", "json", "html"]:
            result = CLIValidators.validate_output_format(fmt)
            assert result == fmt

    def test_invalid_format_raises(self) -> None:
        """Test invalid format raises error."""
        with pytest.raises(click.BadParameter, match="Invalid format"):
            CLIValidators.validate_output_format("invalid")


class TestDateParamType:
    """Tests for DateParamType."""

    def test_valid_date_string(self) -> None:
        """Test valid date string conversion."""
        param_type = DateParamType()
        result = param_type.convert("2023-06-15", None, None)
        assert result == datetime(2023, 6, 15)

    def test_datetime_passthrough(self) -> None:
        """Test datetime passes through unchanged."""
        param_type = DateParamType()
        dt = datetime(2023, 6, 15)
        result = param_type.convert(dt, None, None)
        assert result == dt

    def test_invalid_format_fails(self) -> None:
        """Test invalid date format fails."""
        param_type = DateParamType()
        with pytest.raises(click.exceptions.BadParameter, match="Invalid date format"):
            param_type.convert("15-06-2023", None, None)

    def test_convenience_instance(self) -> None:
        """Test DATE convenience instance."""
        result = DATE.convert("2023-01-01", None, None)
        assert isinstance(result, datetime)


class TestAreaParamType:
    """Tests for AreaParamType."""

    def test_valid_area(self) -> None:
        """Test valid area conversion."""
        param_type = AreaParamType()
        result = param_type.convert("SE", None, None)
        assert result == "SE"

    def test_invalid_area_fails(self) -> None:
        """Test invalid area fails."""
        param_type = AreaParamType()
        with pytest.raises(click.exceptions.BadParameter, match="Invalid area"):
            param_type.convert("INVALID", None, None)

    def test_custom_valid_areas(self) -> None:
        """Test custom valid areas."""
        param_type = AreaParamType(valid_areas=frozenset({"A", "B"}))
        result = param_type.convert("A", None, None)
        assert result == "A"

    def test_convenience_instance(self) -> None:
        """Test AREA convenience instance."""
        result = AREA.convert("SE", None, None)
        assert result == "SE"


class TestHorizonParamType:
    """Tests for HorizonParamType."""

    def test_valid_horizon(self) -> None:
        """Test valid horizon conversion."""
        param_type = HorizonParamType()
        result = param_type.convert("3", None, None)
        assert result == 3

    def test_invalid_horizon_fails(self) -> None:
        """Test invalid horizon fails."""
        param_type = HorizonParamType()
        with pytest.raises(click.exceptions.BadParameter, match="Invalid horizon"):
            param_type.convert("10", None, None)

    def test_non_integer_fails(self) -> None:
        """Test non-integer fails."""
        param_type = HorizonParamType()
        with pytest.raises(click.exceptions.BadParameter, match="Must be an integer"):
            param_type.convert("abc", None, None)

    def test_convenience_instance(self) -> None:
        """Test HORIZON convenience instance."""
        result = HORIZON.convert("5", None, None)
        assert result == 5


class TestModelParamType:
    """Tests for ModelParamType."""

    def test_valid_model(self) -> None:
        """Test valid model conversion."""
        param_type = ModelParamType()
        result = param_type.convert("lgbm", None, None)
        assert result == "lgbm"

    def test_case_insensitive(self) -> None:
        """Test case insensitive conversion."""
        param_type = ModelParamType()
        result = param_type.convert("LGBM", None, None)
        assert result == "lgbm"

    def test_invalid_model_fails(self) -> None:
        """Test invalid model fails."""
        param_type = ModelParamType()
        with pytest.raises(click.exceptions.BadParameter, match="Invalid model"):
            param_type.convert("invalid", None, None)

    def test_convenience_instance(self) -> None:
        """Test MODEL convenience instance."""
        result = MODEL.convert("rf", None, None)
        assert result == "rf"


class TestPathWithExtensionParamType:
    """Tests for PathWithExtensionParamType."""

    def test_valid_path(self) -> None:
        """Test valid path conversion."""
        param_type = PathWithExtensionParamType(expected_extension=".csv")
        result = param_type.convert("output.csv", None, None)
        assert result == Path("output.csv")

    def test_no_extension_fails(self) -> None:
        """Test path without extension fails."""
        param_type = PathWithExtensionParamType(expected_extension=".csv")
        with pytest.raises(click.exceptions.BadParameter, match="must have extension"):
            param_type.convert("output", None, None)

    def test_wrong_extension_fails(self) -> None:
        """Test wrong extension fails."""
        param_type = PathWithExtensionParamType(expected_extension=".csv")
        with pytest.raises(click.exceptions.BadParameter, match="Expected extension"):
            param_type.convert("output.json", None, None)

    def test_no_expected_extension(self) -> None:
        """Test no expected extension allows any path."""
        param_type = PathWithExtensionParamType()
        result = param_type.convert("output.anything", None, None)
        assert result == Path("output.anything")


class TestClassConstants:
    """Tests for class constants."""

    def test_max_date_range_days(self) -> None:
        """Test MAX_DATE_RANGE_DAYS is reasonable."""
        assert CLIValidators.MAX_DATE_RANGE_DAYS == 730  # 2 years

    def test_max_parallel_workers(self) -> None:
        """Test MAX_PARALLEL_WORKERS is reasonable."""
        assert CLIValidators.MAX_PARALLEL_WORKERS == 32

    def test_valid_horizons(self) -> None:
        """Test VALID_HORIZONS contains 0-8."""
        assert CLIValidators.VALID_HORIZONS == tuple(range(9))

    def test_valid_areas_not_empty(self) -> None:
        """Test VALID_AREAS is not empty."""
        assert len(CLIValidators.VALID_AREAS) > 0
        assert "SE" in CLIValidators.VALID_AREAS
        assert "nacional" in CLIValidators.VALID_AREAS

    def test_valid_model_types_not_empty(self) -> None:
        """Test VALID_MODEL_TYPES is not empty."""
        assert len(CLIValidators.VALID_MODEL_TYPES) > 0
        assert "lgbm" in CLIValidators.VALID_MODEL_TYPES

    def test_valid_output_formats_not_empty(self) -> None:
        """Test VALID_OUTPUT_FORMATS is not empty."""
        assert len(CLIValidators.VALID_OUTPUT_FORMATS) > 0
        assert "parquet" in CLIValidators.VALID_OUTPUT_FORMATS

    def test_valid_combination_methods_not_empty(self) -> None:
        """Test VALID_COMBINATION_METHODS is not empty."""
        assert len(CLIValidators.VALID_COMBINATION_METHODS) > 0
        assert "simple_avg" in CLIValidators.VALID_COMBINATION_METHODS
