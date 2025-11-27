"""Tests for DemandProcessor class."""

import numpy as np
import pandas as pd
import pytest

from src.models.demand_mean.demand_processor import DemandProcessor


class TestDemandProcessor:
    """Test suite for DemandProcessor."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = DemandProcessor()
        assert processor is not None

    def test_fill_missing_values(self):
        """Test missing value interpolation."""
        processor = DemandProcessor()

        # Create series with missing values
        series = pd.Series(
            [10.0, np.nan, 20.0, np.nan, np.nan, 30.0],
            index=pd.date_range("2024-01-01", periods=6, freq="D"),
        )

        filled = processor._fill_missing_values(series)

        # Check all values are filled
        assert not filled.isna().any()
        assert len(filled) == len(series)

        # Check interpolation is reasonable
        assert filled.iloc[1] == 15.0  # Linear interpolation between 10 and 20

    def test_fill_missing_values_edge_cases(self):
        """Test missing value handling at edges."""
        processor = DemandProcessor()

        # Missing at start
        series = pd.Series(
            [np.nan, np.nan, 10.0, 20.0],
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )

        filled = processor._fill_missing_values(series)
        assert not filled.isna().any()
        assert filled.iloc[0] == 10.0  # Backward filled

        # Missing at end
        series = pd.Series(
            [10.0, 20.0, np.nan, np.nan],
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )

        filled = processor._fill_missing_values(series)
        assert not filled.isna().any()
        assert filled.iloc[3] == 20.0  # Forward filled

    def test_detect_outliers(self):
        """Test outlier detection using z-score."""
        processor = DemandProcessor()

        # Create series with obvious outlier
        series = pd.Series(
            [10.0, 12.0, 11.0, 500.0, 13.0, 12.0, 11.0],
            index=pd.date_range("2024-01-01", periods=7, freq="D"),
        )

        outlier_mask = processor._detect_outliers(series, z_threshold=3.0)

        # Check that the outlier (500.0) is detected
        assert outlier_mask.iloc[3]  # Index 3 has value 500.0
        assert not outlier_mask.iloc[0]  # Normal values not flagged
        assert not outlier_mask.iloc[1]

    def test_detect_outliers_insufficient_data(self):
        """Test outlier detection with insufficient data."""
        processor = DemandProcessor()

        # Very short series
        series = pd.Series(
            [10.0, 20.0],
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )

        outlier_mask = processor._detect_outliers(series, z_threshold=3.0)

        # Should not flag outliers with insufficient data
        assert not outlier_mask.any()

    def test_handle_outliers_winsorization(self):
        """Test outlier handling using winsorization."""
        processor = DemandProcessor()

        # Create series with outlier
        series = pd.Series(
            [10.0, 12.0, 11.0, 500.0, 13.0, 12.0, 11.0, 10.0, 12.0, 11.0],
            index=pd.date_range("2024-01-01", periods=10, freq="D"),
        )

        outlier_mask = processor._detect_outliers(series, z_threshold=3.0)
        treated = processor._handle_outliers(series, outlier_mask)

        # Outlier should be clipped
        assert treated.iloc[3] < 500.0
        assert treated.iloc[3] <= series.quantile(0.95)

        # Normal values unchanged
        assert treated.iloc[0] == series.iloc[0]
        assert treated.iloc[1] == series.iloc[1]

    def test_process_demand_series_full_pipeline(self):
        """Test full processing pipeline."""
        processor = DemandProcessor()

        # Create series with missing values and outlier
        series = pd.Series(
            [10.0, 12.0, np.nan, 500.0, 13.0, np.nan, 11.0],
            index=pd.date_range("2024-01-01", periods=7, freq="D"),
        )

        processed = processor.process_demand_series(
            series,
            fill_missing=True,
            handle_outliers=True,
            outlier_z_threshold=3.0,
        )

        # Check no missing values
        assert not processed.isna().any()

        # Check outlier was handled
        assert processed.iloc[3] < 500.0

        # Check length preserved
        assert len(processed) == len(series)

    def test_process_demand_series_no_processing(self):
        """Test processing with all options disabled."""
        processor = DemandProcessor()

        series = pd.Series(
            [10.0, 12.0, 11.0, 13.0],
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )

        processed = processor.process_demand_series(
            series,
            fill_missing=False,
            handle_outliers=False,
        )

        # Should return identical series
        pd.testing.assert_series_equal(processed, series)

    def test_process_demand_series_empty_error(self):
        """Test error on empty series."""
        processor = DemandProcessor()

        series = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="Input series is empty"):
            processor.process_demand_series(series)

    def test_process_demand_series_all_nan_error(self):
        """Test error when all values are NaN."""
        processor = DemandProcessor()

        series = pd.Series(
            [np.nan, np.nan, np.nan],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="no valid values"):
            processor.process_demand_series(series)

    def test_validate_for_arima_success(self):
        """Test successful ARIMA validation."""
        processor = DemandProcessor()

        # Good series
        series = pd.Series(
            np.random.uniform(100, 200, 30),
            index=pd.date_range("2024-01-01", periods=30, freq="D"),
        )

        validation = processor.validate_for_arima(series, min_observations=14)

        assert validation["is_valid"]
        assert validation["n_observations"] == 30
        assert validation["n_missing"] == 0
        assert validation["has_variance"]
        assert validation["all_positive"]
        assert len(validation["issues"]) == 0

    def test_validate_for_arima_insufficient_data(self):
        """Test validation failure with insufficient data."""
        processor = DemandProcessor()

        # Too short
        series = pd.Series(
            [100.0, 110.0, 105.0],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        validation = processor.validate_for_arima(series, min_observations=14)

        assert not validation["is_valid"]
        assert "Insufficient data" in validation["issues"][0]
        assert validation["n_observations"] == 3

    def test_validate_for_arima_missing_values(self):
        """Test validation failure with missing values."""
        processor = DemandProcessor()

        series = pd.Series(
            [100.0, np.nan, 110.0, 105.0] * 5,
            index=pd.date_range("2024-01-01", periods=20, freq="D"),
        )

        validation = processor.validate_for_arima(series, min_observations=14)

        assert not validation["is_valid"]
        assert any("missing values" in issue for issue in validation["issues"])
        assert validation["n_missing"] == 5

    def test_validate_for_arima_no_variance(self):
        """Test validation failure with constant values."""
        processor = DemandProcessor()

        # All same value
        series = pd.Series(
            [100.0] * 20,
            index=pd.date_range("2024-01-01", periods=20, freq="D"),
        )

        validation = processor.validate_for_arima(series, min_observations=14)

        assert not validation["is_valid"]
        assert any("no variance" in issue for issue in validation["issues"])
        assert not validation["has_variance"]

    def test_validate_for_arima_negative_values(self):
        """Test validation failure with negative values."""
        processor = DemandProcessor()

        series = pd.Series(
            [100.0, 110.0, -50.0, 105.0, 120.0] * 4,
            index=pd.date_range("2024-01-01", periods=20, freq="D"),
        )

        validation = processor.validate_for_arima(series, min_observations=14)

        assert not validation["is_valid"]
        assert any("negative values" in issue for issue in validation["issues"])
        assert not validation["all_positive"]
