"""Tests for the baseline validator module.

Tests cover:
- ValidationConfig creation and validation
- BaselineReproductionResult dataclass
- ComparisonResult dataclass
- BaselineValidator class methods
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.validation.baseline_validator import (
    BaselineReproductionResult,
    BaselineValidator,
    ComparisonResult,
    ValidationConfig,
)
from src.validation.baseline_data_loader import BaselineDataset, PredictionDataset
from src.validation.statistical_tests import ValidationResult


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""

    def test_create_config_defaults(self) -> None:
        """Test creating config with default values."""
        config = ValidationConfig(baseline_path="/data/baseline")

        assert config.baseline_path == Path("/data/baseline")
        assert len(config.areas) > 0
        assert len(config.models) > 0
        assert config.mape_tolerance == 0.05
        assert config.mae_tolerance == 100.0
        assert config.rmse_tolerance == 150.0
        assert config.confidence_level == 0.95

    def test_create_config_custom(self) -> None:
        """Test creating config with custom values."""
        config = ValidationConfig(
            baseline_path="/custom/path",
            areas=["SECO", "S"],
            models=["lgbm", "rf"],
            mape_tolerance=0.10,
            mae_tolerance=200.0,
            rmse_tolerance=300.0,
            confidence_level=0.99,
        )

        assert config.baseline_path == Path("/custom/path")
        assert config.areas == ["SECO", "S"]
        assert config.models == ["lgbm", "rf"]
        assert config.mape_tolerance == 0.10
        assert config.mae_tolerance == 200.0
        assert config.rmse_tolerance == 300.0
        assert config.confidence_level == 0.99

    def test_config_negative_mape_tolerance_raises(self) -> None:
        """Test that negative mape_tolerance raises error."""
        with pytest.raises(ValueError, match="mape_tolerance"):
            ValidationConfig(baseline_path="/data", mape_tolerance=-0.05)

    def test_config_negative_mae_tolerance_raises(self) -> None:
        """Test that negative mae_tolerance raises error."""
        with pytest.raises(ValueError, match="mae_tolerance"):
            ValidationConfig(baseline_path="/data", mae_tolerance=-100)

    def test_config_negative_rmse_tolerance_raises(self) -> None:
        """Test that negative rmse_tolerance raises error."""
        with pytest.raises(ValueError, match="rmse_tolerance"):
            ValidationConfig(baseline_path="/data", rmse_tolerance=-150)

    def test_config_invalid_confidence_level_raises(self) -> None:
        """Test that invalid confidence_level raises error."""
        with pytest.raises(ValueError, match="confidence_level"):
            ValidationConfig(baseline_path="/data", confidence_level=1.5)

        with pytest.raises(ValueError, match="confidence_level"):
            ValidationConfig(baseline_path="/data", confidence_level=0)

    def test_tolerance_thresholds_property(self) -> None:
        """Test tolerance_thresholds property."""
        config = ValidationConfig(
            baseline_path="/data",
            mape_tolerance=0.08,
            mae_tolerance=120.0,
            rmse_tolerance=180.0,
        )

        thresholds = config.tolerance_thresholds
        assert thresholds["mape"] == 0.08
        assert thresholds["mae"] == 120.0
        assert thresholds["rmse"] == 180.0

    def test_config_to_dict(self) -> None:
        """Test config to_dict method."""
        config = ValidationConfig(
            baseline_path="/data/baseline",
            areas=["SECO"],
            models=["lgbm"],
        )

        d = config.to_dict()
        assert d["baseline_path"] == "/data/baseline"
        assert d["areas"] == ["SECO"]
        assert d["models"] == ["lgbm"]
        assert "mape_tolerance" in d
        assert "mae_tolerance" in d
        assert "rmse_tolerance" in d
        assert "confidence_level" in d


class TestBaselineReproductionResult:
    """Tests for BaselineReproductionResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating reproduction result."""
        validation_result = ValidationResult(
            is_valid=True,
            validations={"SECO_lgbm": {"all_valid": True}},
            summary={"pass_rate": 1.0},
        )

        result = BaselineReproductionResult(
            baseline_metrics={"SECO_lgbm": {"mape": 3.5, "mae": 80, "rmse": 120}},
            validation_result=validation_result,
            reference_dataset={"metrics": {}},
            reproduction_timestamp=datetime.now(),
        )

        assert result.baseline_metrics["SECO_lgbm"]["mape"] == 3.5
        assert result.validation_result.is_valid is True

    def test_result_to_dict(self) -> None:
        """Test result to_dict method."""
        validation_result = ValidationResult(
            is_valid=True,
            validations={},
            summary={},
        )

        config = ValidationConfig(baseline_path="/data")
        result = BaselineReproductionResult(
            baseline_metrics={"key": {"mape": 5.0}},
            validation_result=validation_result,
            reference_dataset={},
            reproduction_timestamp=datetime(2024, 1, 15, 10, 30),
            config=config,
        )

        d = result.to_dict()
        assert "baseline_metrics" in d
        assert "validation_result" in d
        assert "reference_dataset" in d
        assert "timestamp" in d
        assert "config" in d


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_passes_validation_no_violations(self) -> None:
        """Test passes_validation with no violations."""
        result = ComparisonResult(
            metrics={"key": {"difference": 0.01}},
            summary={"pass_rate": 1.0},
            violations=[],
        )

        assert result.passes_validation() is True

    def test_passes_validation_with_violations(self) -> None:
        """Test passes_validation with violations."""
        result = ComparisonResult(
            metrics={"key": {"difference": 0.10}},
            summary={"pass_rate": 0.5},
            violations=["key: exceeds tolerance"],
        )

        assert result.passes_validation() is False

    def test_comparison_result_to_dict(self) -> None:
        """Test to_dict method."""
        result = ComparisonResult(
            metrics={"SECO_lgbm": {"system_mape": 3.5, "baseline_mape": 3.0}},
            summary={"mean_difference": 0.5},
            violations=["SECO_lgbm: exceeds tolerance"],
            statistical_tests={"SECO_lgbm": {"p_value": 0.05}},
        )

        d = result.to_dict()
        assert "metrics" in d
        assert "summary" in d
        assert "violations" in d
        assert "passes_validation" in d
        assert "statistical_tests" in d
        assert d["passes_validation"] is False


class TestBaselineValidator:
    """Tests for BaselineValidator class."""

    @pytest.fixture
    def temp_baseline_dir(self) -> Path:
        """Create temporary directory with mock baseline data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir)

            # Create mock baseline CSV
            dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
            data = []
            for date in dates:
                for area in ["SECO", "S", "NE", "N"]:
                    for model in ["lgbm", "rf"]:
                        actual = 10000 + np.random.normal(0, 500)
                        prediction = actual + np.random.normal(0, 300)
                        data.append({
                            "date": date,
                            "area": area,
                            "model": model,
                            "prediction": prediction,
                            "actual": actual,
                            "horizon": 1,
                        })

            df = pd.DataFrame(data)
            df.to_csv(baseline_path / "baseline_2024-01.csv", index=False)

            yield baseline_path

    @pytest.fixture
    def config(self, temp_baseline_dir: Path) -> ValidationConfig:
        """Create validation config."""
        return ValidationConfig(
            baseline_path=temp_baseline_dir,
            areas=["SECO", "S", "NE", "N"],
            models=["lgbm", "rf"],
        )

    @pytest.fixture
    def validator(self, config: ValidationConfig) -> BaselineValidator:
        """Create validator instance."""
        return BaselineValidator(config)

    def test_init(self, validator: BaselineValidator, config: ValidationConfig) -> None:
        """Test validator initialization."""
        assert validator.config == config
        assert validator.baseline_data is not None
        assert validator.statistical_tests is not None

    def test_reproduce_baseline(self, validator: BaselineValidator) -> None:
        """Test baseline reproduction."""
        result = validator.reproduce_baseline(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )

        assert isinstance(result, BaselineReproductionResult)
        assert len(result.baseline_metrics) > 0
        assert result.validation_result is not None
        assert result.reproduction_timestamp is not None

    def test_reproduce_baseline_with_specific_models(
        self, validator: BaselineValidator
    ) -> None:
        """Test baseline reproduction with specific models."""
        result = validator.reproduce_baseline(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            models=["lgbm"],
        )

        # Should only have lgbm metrics
        for key in result.baseline_metrics:
            assert "lgbm" in key

    def test_compare_with_baseline(self, validator: BaselineValidator) -> None:
        """Test comparison with baseline."""
        # Create mock datasets
        dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")
        data = []
        for date in dates:
            for area in ["SECO", "S"]:
                for model in ["lgbm", "rf"]:
                    actual = 10000 + np.random.normal(0, 200)
                    prediction = actual + np.random.normal(0, 100)
                    data.append({
                        "date": date,
                        "area": area,
                        "model": model,
                        "prediction": prediction,
                        "actual": actual,
                    })

        df = pd.DataFrame(data)

        system_predictions = PredictionDataset(
            data=df.copy(),
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
            areas=["SECO", "S"],
            models=["lgbm", "rf"],
        )

        baseline_predictions = BaselineDataset(
            data=df.copy(),
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
            areas=["SECO", "S"],
            models=["lgbm", "rf"],
        )

        result = validator.compare_with_baseline(
            system_predictions, baseline_predictions
        )

        assert isinstance(result, ComparisonResult)
        assert len(result.metrics) > 0
        assert "mean_difference" in result.summary

    def test_compare_with_baseline_no_common_areas(
        self, validator: BaselineValidator
    ) -> None:
        """Test comparison when no common areas."""
        df1 = pd.DataFrame({
            "date": [datetime(2024, 1, 1)],
            "area": ["AREA_X"],
            "model": ["lgbm"],
            "prediction": [100],
            "actual": [100],
        })
        df2 = pd.DataFrame({
            "date": [datetime(2024, 1, 1)],
            "area": ["AREA_Y"],
            "model": ["lgbm"],
            "prediction": [100],
            "actual": [100],
        })

        system_predictions = PredictionDataset(
            data=df1,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1),
            areas=["AREA_X"],
            models=["lgbm"],
        )

        baseline_predictions = BaselineDataset(
            data=df2,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1),
            areas=["AREA_Y"],
            models=["lgbm"],
        )

        result = validator.compare_with_baseline(
            system_predictions, baseline_predictions
        )

        assert len(result.violations) > 0

    def test_validate_data_alignment(self, validator: BaselineValidator) -> None:
        """Test data alignment validation."""
        dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")
        data = []
        for date in dates:
            for area in ["SECO", "S"]:
                data.append({
                    "date": date,
                    "area": area,
                    "model": "lgbm",
                    "prediction": 100,
                    "actual": 100,
                })

        df = pd.DataFrame(data)

        system_data = PredictionDataset(
            data=df.copy(),
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
            areas=["SECO", "S"],
            models=["lgbm"],
        )

        baseline_data = BaselineDataset(
            data=df.copy(),
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
            areas=["SECO", "S"],
            models=["lgbm"],
        )

        alignment = validator.validate_data_alignment(system_data, baseline_data)

        assert alignment["date_range_match"] is True
        assert alignment["area_match"] is True
        assert alignment["model_match"] is True
        assert alignment["overall_aligned"] is True

    def test_validate_data_alignment_mismatch(
        self, validator: BaselineValidator
    ) -> None:
        """Test data alignment validation with mismatches."""
        df1 = pd.DataFrame({
            "date": pd.date_range("2024-01-01", "2024-01-10", freq="D"),
            "area": ["SECO"] * 10,
            "model": ["lgbm"] * 10,
            "prediction": [100] * 10,
            "actual": [100] * 10,
        })
        df2 = pd.DataFrame({
            "date": pd.date_range("2024-02-01", "2024-02-10", freq="D"),
            "area": ["S"] * 10,
            "model": ["rf"] * 10,
            "prediction": [100] * 10,
            "actual": [100] * 10,
        })

        system_data = PredictionDataset(
            data=df1,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
            areas=["SECO"],
            models=["lgbm"],
        )

        baseline_data = BaselineDataset(
            data=df2,
            start_date=datetime(2024, 2, 1),
            end_date=datetime(2024, 2, 10),
            areas=["S"],
            models=["rf"],
        )

        alignment = validator.validate_data_alignment(system_data, baseline_data)

        assert alignment["date_range_match"] is False
        assert alignment["overall_aligned"] is False

    def test_generate_reference_archive(self, validator: BaselineValidator) -> None:
        """Test reference archive generation."""
        validation_result = ValidationResult(
            is_valid=True,
            validations={
                "SECO_lgbm": {"mape_valid": True, "mae_valid": True, "rmse_valid": True, "all_valid": True}
            },
            summary={"pass_rate": 1.0},
        )

        reproduction_result = BaselineReproductionResult(
            baseline_metrics={
                "SECO_lgbm": {"mape": 3.5, "mae": 80, "rmse": 120},
                "S_lgbm": {"mape": 4.0, "mae": 90, "rmse": 130},
            },
            validation_result=validation_result,
            reference_dataset={},
            reproduction_timestamp=datetime.now(),
        )

        archive = validator.generate_reference_archive(reproduction_result)

        assert "generated_at" in archive
        assert "config" in archive
        assert "time_series" in archive
        assert "SECO" in archive["time_series"]
        assert "lgbm" in archive["time_series"]["SECO"]
        assert archive["time_series"]["SECO"]["lgbm"]["mape"] == 3.5

    def test_calculate_mape(self, validator: BaselineValidator) -> None:
        """Test MAPE calculation."""
        predictions = np.array([100, 200, 300, 400, 500])
        actuals = np.array([105, 195, 310, 390, 510])

        mape = validator._calculate_mape(predictions, actuals)

        assert mape > 0
        assert mape < 10  # Should be small percentage

    def test_calculate_mape_empty(self, validator: BaselineValidator) -> None:
        """Test MAPE calculation with empty arrays."""
        mape = validator._calculate_mape(np.array([]), np.array([]))
        assert np.isnan(mape)

    def test_calculate_mape_zeros(self, validator: BaselineValidator) -> None:
        """Test MAPE calculation with zero actuals."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([0, 0, 0])

        mape = validator._calculate_mape(predictions, actuals)
        assert np.isnan(mape)

    def test_calculate_summary(self, validator: BaselineValidator) -> None:
        """Test summary calculation."""
        comparison_metrics = {
            "SECO_lgbm": {"difference": 0.5, "within_tolerance": 1.0},
            "S_lgbm": {"difference": -0.3, "within_tolerance": 1.0},
            "NE_lgbm": {"difference": 0.8, "within_tolerance": 0.0},
        }

        summary = validator._calculate_summary(comparison_metrics)

        assert "mean_difference" in summary
        assert "std_difference" in summary
        assert "max_difference" in summary
        assert "min_difference" in summary
        assert "pass_rate" in summary
        assert summary["pass_rate"] == pytest.approx(2 / 3)

    def test_calculate_summary_empty(self, validator: BaselineValidator) -> None:
        """Test summary calculation with empty metrics."""
        summary = validator._calculate_summary({})

        assert summary["pass_rate"] == 0.0
        assert summary["total_comparisons"] == 0


class TestIntegration:
    """Integration tests for baseline validation workflow."""

    def test_full_validation_workflow(self) -> None:
        """Test complete validation workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir)

            # Create mock baseline data
            dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
            data = []
            for date in dates:
                for area in ["SECO", "S"]:
                    for model in ["lgbm"]:
                        actual = 10000 + np.random.normal(0, 500)
                        prediction = actual * (1 + np.random.normal(0, 0.02))
                        data.append({
                            "date": date,
                            "area": area,
                            "model": model,
                            "prediction": prediction,
                            "actual": actual,
                            "horizon": 1,
                        })

            df = pd.DataFrame(data)
            df.to_csv(baseline_path / "baseline.csv", index=False)

            # Create config and validator
            config = ValidationConfig(
                baseline_path=baseline_path,
                areas=["SECO", "S"],
                models=["lgbm"],
                mape_tolerance=0.05,
            )
            validator = BaselineValidator(config)

            # Step 1: Reproduce baseline
            reproduction_result = validator.reproduce_baseline(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
            )

            assert isinstance(reproduction_result, BaselineReproductionResult)
            assert len(reproduction_result.baseline_metrics) > 0

            # Step 2: Generate reference archive
            archive = validator.generate_reference_archive(reproduction_result)

            assert "time_series" in archive
            assert len(archive["time_series"]) > 0

            # Step 3: Compare with "system" (using same data as baseline)
            baseline_dataset = validator.baseline_data.create_baseline_dataset(
                datetime(2024, 1, 1), datetime(2024, 1, 31)
            )

            # Create slightly modified system predictions
            system_df = df.copy()
            system_df["prediction"] = system_df["prediction"] * 1.01  # 1% difference

            system_predictions = PredictionDataset(
                data=system_df,
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
                areas=["SECO", "S"],
                models=["lgbm"],
            )

            comparison = validator.compare_with_baseline(
                system_predictions, baseline_dataset
            )

            assert isinstance(comparison, ComparisonResult)
            assert "mean_difference" in comparison.summary
