"""Tests for the baseline data loader module.

Tests cover:
- BaselineDataset and PredictionDataset dataclasses
- BaselineDataLoader class methods
- Metric calculations (MAPE, MAE, RMSE)
- File discovery and loading
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.validation.baseline_data_loader import (
    BaselineDataLoader,
    BaselineDataset,
    PredictionDataset,
    _calculate_mae,
    _calculate_mape,
    _calculate_rmse,
)


class TestMetricCalculations:
    """Tests for metric calculation functions."""

    def test_calculate_mape_normal(self) -> None:
        """Test MAPE calculation with normal values."""
        predictions = np.array([100, 200, 300, 400, 500])
        actuals = np.array([105, 195, 310, 390, 510])

        mape = _calculate_mape(predictions, actuals)

        # Should be positive and reasonable
        assert 0 < mape < 10

    def test_calculate_mape_perfect(self) -> None:
        """Test MAPE with perfect predictions."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([100, 200, 300])

        mape = _calculate_mape(predictions, actuals)
        assert mape == 0.0

    def test_calculate_mape_empty(self) -> None:
        """Test MAPE with empty arrays."""
        mape = _calculate_mape(np.array([]), np.array([]))
        assert np.isnan(mape)

    def test_calculate_mape_zeros(self) -> None:
        """Test MAPE with zero actuals."""
        predictions = np.array([100, 200])
        actuals = np.array([0, 0])

        mape = _calculate_mape(predictions, actuals)
        assert np.isnan(mape)

    def test_calculate_mape_mismatched_lengths(self) -> None:
        """Test MAPE with mismatched array lengths."""
        predictions = np.array([100, 200, 300, 400, 500])
        actuals = np.array([100, 200, 300])

        mape = _calculate_mape(predictions, actuals)
        # Should use minimum length
        assert not np.isnan(mape)

    def test_calculate_mae_normal(self) -> None:
        """Test MAE calculation."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([110, 190, 310])

        mae = _calculate_mae(predictions, actuals)
        assert mae == 10.0

    def test_calculate_mae_empty(self) -> None:
        """Test MAE with empty arrays."""
        mae = _calculate_mae(np.array([]), np.array([]))
        assert np.isnan(mae)

    def test_calculate_rmse_normal(self) -> None:
        """Test RMSE calculation."""
        predictions = np.array([100, 200, 300])
        actuals = np.array([110, 190, 310])

        rmse = _calculate_rmse(predictions, actuals)
        assert rmse == 10.0  # All errors are 10

    def test_calculate_rmse_empty(self) -> None:
        """Test RMSE with empty arrays."""
        rmse = _calculate_rmse(np.array([]), np.array([]))
        assert np.isnan(rmse)


class TestBaselineDataset:
    """Tests for BaselineDataset dataclass."""

    @pytest.fixture
    def sample_dataset(self) -> BaselineDataset:
        """Create sample dataset."""
        data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "area": ["SECO"] * 5 + ["S"] * 5,
            "model": ["lgbm"] * 10,
            "prediction": list(range(100, 200, 10)),
            "actual": list(range(105, 205, 10)),
        })
        return BaselineDataset(
            data=data,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
            areas=["SECO", "S"],
            models=["lgbm"],
        )

    def test_get_predictions(self, sample_dataset: BaselineDataset) -> None:
        """Test getting predictions for area and model."""
        preds = sample_dataset.get_predictions("SECO", "lgbm")
        assert len(preds) == 5
        assert preds[0] == 100

    def test_get_actuals(self, sample_dataset: BaselineDataset) -> None:
        """Test getting actuals for area."""
        actuals = sample_dataset.get_actuals("SECO")
        assert len(actuals) == 5
        assert actuals[0] == 105

    def test_get_mape(self, sample_dataset: BaselineDataset) -> None:
        """Test getting MAPE for area and model."""
        mape = sample_dataset.get_mape("SECO", "lgbm")
        assert mape > 0


class TestPredictionDataset:
    """Tests for PredictionDataset dataclass."""

    @pytest.fixture
    def sample_dataset(self) -> PredictionDataset:
        """Create sample dataset."""
        data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "area": ["SECO"] * 5 + ["S"] * 5,
            "model": ["lgbm"] * 10,
            "prediction": list(range(100, 200, 10)),
            "actual": list(range(105, 205, 10)),
        })
        return PredictionDataset(
            data=data,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
            areas=["SECO", "S"],
            models=["lgbm"],
        )

    def test_get_predictions(self, sample_dataset: PredictionDataset) -> None:
        """Test getting predictions."""
        preds = sample_dataset.get_predictions("SECO", "lgbm")
        assert len(preds) == 5

    def test_get_actuals_unique(self, sample_dataset: PredictionDataset) -> None:
        """Test getting unique actuals."""
        actuals = sample_dataset.get_actuals("SECO")
        assert len(actuals) > 0


class TestBaselineDataLoader:
    """Tests for BaselineDataLoader class."""

    @pytest.fixture
    def temp_dir_with_data(self) -> Path:
        """Create temp directory with baseline data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create baseline CSV
            dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
            data = []
            for date in dates:
                for area in ["SECO", "S"]:
                    for model in ["lgbm", "rf"]:
                        data.append({
                            "date": date,
                            "area": area,
                            "model": model,
                            "prediction": 10000 + np.random.normal(0, 500),
                            "actual": 10000 + np.random.normal(0, 500),
                        })

            df = pd.DataFrame(data)
            df.to_csv(path / "baseline_2024-01-01.csv", index=False)

            yield path

    @pytest.fixture
    def loader(self, temp_dir_with_data: Path) -> BaselineDataLoader:
        """Create loader instance."""
        return BaselineDataLoader(temp_dir_with_data)

    def test_init(self, loader: BaselineDataLoader) -> None:
        """Test loader initialization."""
        assert loader.baseline_path.exists()
        assert loader.cache == {}

    def test_load_historical_data(self, loader: BaselineDataLoader) -> None:
        """Test loading historical data."""
        data = loader.load_historical_data(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert "date" in data.columns
        assert "area" in data.columns
        assert "model" in data.columns

    def test_load_historical_data_caching(self, loader: BaselineDataLoader) -> None:
        """Test that data is cached."""
        data1 = loader.load_historical_data(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )
        data2 = loader.load_historical_data(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )

        # Should be same object from cache
        assert data1 is data2

    def test_calculate_metrics(self, loader: BaselineDataLoader) -> None:
        """Test metric calculation."""
        metrics = loader.calculate_metrics(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            models=["lgbm", "rf"],
        )

        assert len(metrics) > 0
        for key, m in metrics.items():
            assert "mape" in m
            assert "mae" in m
            assert "rmse" in m

    def test_to_reference_format(self, loader: BaselineDataLoader) -> None:
        """Test reference format conversion."""
        metrics = {"SECO_lgbm": {"mape": 3.5, "mae": 80, "rmse": 120}}
        ref = loader.to_reference_format(metrics)

        assert "metrics" in ref
        assert "generated_at" in ref
        assert "format_version" in ref

    def test_create_baseline_dataset(self, loader: BaselineDataLoader) -> None:
        """Test baseline dataset creation."""
        dataset = loader.create_baseline_dataset(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )

        assert isinstance(dataset, BaselineDataset)
        assert len(dataset.areas) > 0
        assert len(dataset.models) > 0

    def test_discover_baseline_files(self, loader: BaselineDataLoader) -> None:
        """Test file discovery."""
        files = loader._discover_baseline_files(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )

        assert len(files) >= 1

    def test_discover_files_nonexistent_path(self) -> None:
        """Test file discovery with nonexistent path."""
        loader = BaselineDataLoader("/nonexistent/path")
        files = loader._discover_baseline_files(
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
        )

        assert len(files) == 0

    def test_extract_date_from_filename_yyyy_mm_dd(self) -> None:
        """Test date extraction from filename with YYYY-MM-DD format."""
        date = BaselineDataLoader._extract_date_from_filename("baseline_2024-01-15.csv")
        assert date == datetime(2024, 1, 15)

    def test_extract_date_from_filename_yyyy_mm_dd_underscore(self) -> None:
        """Test date extraction from filename with YYYY_MM_DD format."""
        date = BaselineDataLoader._extract_date_from_filename("baseline_2024_01_15.csv")
        assert date == datetime(2024, 1, 15)

    def test_extract_date_from_filename_yyyymmdd(self) -> None:
        """Test date extraction from filename with YYYYMMDD format."""
        date = BaselineDataLoader._extract_date_from_filename("baseline_20240115.csv")
        assert date == datetime(2024, 1, 15)

    def test_extract_date_from_filename_no_date(self) -> None:
        """Test date extraction from filename without date."""
        date = BaselineDataLoader._extract_date_from_filename("baseline.csv")
        assert date is None

    def test_load_historical_data_empty_path(self) -> None:
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = BaselineDataLoader(tmpdir)
            data = loader.load_historical_data(
                datetime(2024, 1, 1),
                datetime(2024, 1, 31),
            )

            assert len(data) == 0

    def test_load_baseline_file_invalid(self) -> None:
        """Test loading invalid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_file = Path(tmpdir) / "invalid.csv"
            invalid_file.write_text("not,valid,csv\ndata")

            loader = BaselineDataLoader(tmpdir)
            # Should handle gracefully
            df = loader._load_baseline_file(invalid_file)
            # May return df or None depending on pandas version
            assert df is None or isinstance(df, pd.DataFrame)


class TestEdgeCases:
    """Edge case tests."""

    def test_loader_with_various_date_formats(self) -> None:
        """Test loader handles various date column names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create file with 'datetime' column instead of 'date'
            df = pd.DataFrame({
                "datetime": pd.date_range("2024-01-01", periods=10),
                "area": ["SECO"] * 10,
                "model": ["lgbm"] * 10,
                "prediction": [100] * 10,
                "actual": [100] * 10,
            })
            df.to_csv(path / "baseline.csv", index=False)

            loader = BaselineDataLoader(path)
            data = loader.load_historical_data(
                datetime(2024, 1, 1),
                datetime(2024, 1, 31),
            )

            assert len(data) > 0

    def test_metrics_calculation_with_missing_area_model(self) -> None:
        """Test metrics when area/model combination is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Only SECO data
            df = pd.DataFrame({
                "date": pd.date_range("2024-01-01", periods=10),
                "area": ["SECO"] * 10,
                "model": ["lgbm"] * 10,
                "prediction": [100] * 10,
                "actual": [100] * 10,
            })
            df.to_csv(path / "baseline.csv", index=False)

            loader = BaselineDataLoader(path)
            metrics = loader.calculate_metrics(
                datetime(2024, 1, 1),
                datetime(2024, 1, 31),
                models=["lgbm", "rf"],  # rf doesn't exist
            )

            # Should only have SECO_lgbm
            assert "SECO_lgbm" in metrics
            assert "S_lgbm" not in metrics
