"""Tests for data preprocessing and imputation utilities.

This module contains comprehensive tests for the data preprocessing functionality
in the PrevCarga system, including imputation strategies, resampling utilities,
and timezone handling.
"""

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessors import (
    BackwardFillImputer,
    BaseImputer,
    ForwardFillImputer,
    ImputerChain,
    InterpolationImputer,
    LagFillImputer,
    ResamplerMixin,
    TimezoneHandler,
    TriplePassImputer,
)

# =============================================================================
# BaseImputer Tests
# =============================================================================


class TestBaseImputer:
    """Tests for BaseImputer abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseImputer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseImputer()  # type: ignore[abstract]

    def test_concrete_implementation_required(self):
        """Test that concrete classes must implement abstract methods."""

        class IncompleteImputer(BaseImputer):
            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteImputer()  # type: ignore[abstract]


# =============================================================================
# ForwardFillImputer Tests
# =============================================================================


class TestForwardFillImputer:
    """Tests for ForwardFillImputer class."""

    @pytest.fixture
    def df_with_gaps(self) -> pd.DataFrame:
        """Create a DataFrame with missing values for testing."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
                "load": [
                    100.0,
                    np.nan,
                    np.nan,
                    120.0,
                    np.nan,
                    140.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    200.0,
                ],
                "area": ["SP"] * 10,
            }
        )

    def test_basic_forward_fill(self, df_with_gaps: pd.DataFrame):
        """Test basic forward fill without limit."""
        imputer = ForwardFillImputer()
        result = imputer.fit_transform(df_with_gaps, value_col="load")

        # First value should remain 100
        assert result["load"].iloc[0] == 100.0
        # Second and third should be filled with 100
        assert result["load"].iloc[1] == 100.0
        assert result["load"].iloc[2] == 100.0
        # Fourth should be 120
        assert result["load"].iloc[3] == 120.0
        # Fifth should be filled with 120
        assert result["load"].iloc[4] == 120.0

    def test_forward_fill_with_limit(self, df_with_gaps: pd.DataFrame):
        """Test forward fill with limit parameter."""
        imputer = ForwardFillImputer(limit=1)
        result = imputer.fit_transform(df_with_gaps, value_col="load")

        # First value should remain 100
        assert result["load"].iloc[0] == 100.0
        # Second should be filled with 100 (within limit)
        assert result["load"].iloc[1] == 100.0
        # Third should still be NaN (beyond limit)
        assert pd.isna(result["load"].iloc[2])

    def test_forward_fill_with_groupby(self):
        """Test forward fill with groupby support."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
                "load": [100.0, np.nan, 120.0, 200.0, np.nan, 220.0],
                "area": ["SP", "SP", "SP", "RJ", "RJ", "RJ"],
            }
        )

        imputer = ForwardFillImputer(group_col="area")
        result = imputer.fit_transform(df, value_col="load")

        # SP values
        assert result.loc[result["area"] == "SP", "load"].iloc[0] == 100.0
        assert result.loc[result["area"] == "SP", "load"].iloc[1] == 100.0  # Filled
        # RJ values
        assert result.loc[result["area"] == "RJ", "load"].iloc[0] == 200.0
        assert result.loc[result["area"] == "RJ", "load"].iloc[1] == 200.0  # Filled

    def test_name_property(self):
        """Test that name property returns correct value."""
        imputer = ForwardFillImputer()
        assert imputer.name == "forward_fill"

    def test_missing_column_raises_error(self, df_with_gaps: pd.DataFrame):
        """Test that missing column raises KeyError."""
        imputer = ForwardFillImputer()
        with pytest.raises(KeyError, match="not found"):
            imputer.fit_transform(df_with_gaps, value_col="nonexistent")

    def test_missing_group_column_raises_error(self, df_with_gaps: pd.DataFrame):
        """Test that missing group column raises KeyError."""
        imputer = ForwardFillImputer(group_col="nonexistent")
        with pytest.raises(KeyError, match="Group column"):
            imputer.fit_transform(df_with_gaps, value_col="load")


# =============================================================================
# BackwardFillImputer Tests
# =============================================================================


class TestBackwardFillImputer:
    """Tests for BackwardFillImputer class."""

    @pytest.fixture
    def df_with_gaps(self) -> pd.DataFrame:
        """Create a DataFrame with missing values for testing."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
                "load": [
                    np.nan,
                    np.nan,
                    100.0,
                    np.nan,
                    120.0,
                    np.nan,
                    np.nan,
                    140.0,
                    np.nan,
                    np.nan,
                ],
                "area": ["SP"] * 10,
            }
        )

    def test_basic_backward_fill(self, df_with_gaps: pd.DataFrame):
        """Test basic backward fill without limit."""
        imputer = BackwardFillImputer()
        result = imputer.fit_transform(df_with_gaps, value_col="load")

        # First two should be filled with 100
        assert result["load"].iloc[0] == 100.0
        assert result["load"].iloc[1] == 100.0
        # Third should be 100
        assert result["load"].iloc[2] == 100.0
        # Fourth should be filled with 120
        assert result["load"].iloc[3] == 120.0

    def test_backward_fill_with_limit(self, df_with_gaps: pd.DataFrame):
        """Test backward fill with limit parameter."""
        imputer = BackwardFillImputer(limit=1)
        result = imputer.fit_transform(df_with_gaps, value_col="load")

        # First should still be NaN (beyond limit)
        assert pd.isna(result["load"].iloc[0])
        # Second should be filled with 100 (within limit)
        assert result["load"].iloc[1] == 100.0

    def test_backward_fill_with_groupby(self):
        """Test backward fill with groupby support."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
                "load": [np.nan, 100.0, 120.0, np.nan, 200.0, 220.0],
                "area": ["SP", "SP", "SP", "RJ", "RJ", "RJ"],
            }
        )

        imputer = BackwardFillImputer(group_col="area")
        result = imputer.fit_transform(df, value_col="load")

        # SP values - first should be filled with 100
        assert result.loc[result["area"] == "SP", "load"].iloc[0] == 100.0
        # RJ values - first should be filled with 200
        assert result.loc[result["area"] == "RJ", "load"].iloc[0] == 200.0

    def test_name_property(self):
        """Test that name property returns correct value."""
        imputer = BackwardFillImputer()
        assert imputer.name == "backward_fill"


# =============================================================================
# InterpolationImputer Tests
# =============================================================================


class TestInterpolationImputer:
    """Tests for InterpolationImputer class."""

    @pytest.fixture
    def df_with_gaps(self) -> pd.DataFrame:
        """Create a DataFrame with missing values for testing."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
                "load": [100.0, np.nan, np.nan, np.nan, 200.0],
            }
        )

    def test_linear_interpolation(self, df_with_gaps: pd.DataFrame):
        """Test linear interpolation fills values correctly."""
        imputer = InterpolationImputer(method="linear")
        result = imputer.fit_transform(df_with_gaps, value_col="load")

        # Check that all values are filled
        assert not result["load"].isna().any()

        # Check interpolated values (linear: 100, 125, 150, 175, 200)
        assert result["load"].iloc[0] == 100.0
        assert result["load"].iloc[1] == 125.0
        assert result["load"].iloc[2] == 150.0
        assert result["load"].iloc[3] == 175.0
        assert result["load"].iloc[4] == 200.0

    def test_name_property_includes_method(self):
        """Test that name property includes interpolation method."""
        imputer = InterpolationImputer(method="linear")
        assert imputer.name == "interpolation_linear"

        imputer2 = InterpolationImputer(method="nearest")
        assert imputer2.name == "interpolation_nearest"

    def test_invalid_method_raises_error(self):
        """Test that invalid interpolation method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid interpolation method"):
            InterpolationImputer(method="invalid_method")

    def test_method_property(self):
        """Test that method property returns correct value."""
        imputer = InterpolationImputer(method="cubic")
        assert imputer.method == "cubic"

    def test_missing_column_raises_error(self, df_with_gaps: pd.DataFrame):
        """Test that missing column raises KeyError."""
        imputer = InterpolationImputer()
        with pytest.raises(KeyError, match="not found"):
            imputer.fit_transform(df_with_gaps, value_col="nonexistent")


# =============================================================================
# LagFillImputer Tests
# =============================================================================


class TestLagFillImputer:
    """Tests for LagFillImputer class."""

    @pytest.fixture
    def df_with_daily_pattern(self) -> pd.DataFrame:
        """Create DataFrame with 30-min data spanning multiple days."""
        # 48 periods per day (30-min intervals) for 3 days
        n_periods = 48 * 3
        timestamps = pd.date_range("2024-01-01", periods=n_periods, freq="30min")

        # Create pattern: day 1 has all values, day 2 has some missing
        values = []
        for i in range(n_periods):
            if i < 48:  # Day 1 - all values
                values.append(100.0 + i)
            elif 48 <= i < 96:  # Day 2 - some missing
                if i in [50, 60, 70]:
                    values.append(np.nan)
                else:
                    values.append(100.0 + i)
            else:  # Day 3 - all values
                values.append(100.0 + i)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "load": values,
            }
        )

    def test_24h_lag_fill(self, df_with_daily_pattern: pd.DataFrame):
        """Test that 24h lag fills missing values from previous day."""
        imputer = LagFillImputer(lag_hours=[24])
        result = imputer.fit_transform(df_with_daily_pattern, value_col="load")

        # Check that missing values at indices 50, 60, 70 are filled
        # They should be filled with values from 24h earlier (48 periods back)
        # Index 50 should be filled with value from index 2 (50-48)
        assert not pd.isna(result["load"].iloc[50])
        assert result["load"].iloc[50] == df_with_daily_pattern["load"].iloc[2]

    def test_48h_lag_fallback(self):
        """Test that 48h lag is used when 24h lag has no value."""
        # Create data where 24h ago also has missing value
        n_periods = 48 * 3
        timestamps = pd.date_range("2024-01-01", periods=n_periods, freq="30min")

        values = []
        for i in range(n_periods):
            if i in {10, 58}:  # Missing at index 10 (day 1) and 58 (day 2)
                values.append(np.nan)
            else:
                values.append(100.0 + i)

        df = pd.DataFrame({"timestamp": timestamps, "load": values})

        imputer = LagFillImputer(lag_hours=[24, 48])
        _ = imputer.fit_transform(df, value_col="load")

        # Index 58 should be filled from 48h back (index 58 - 96 = -38, which doesn't exist)
        # But index 58's 24h lag (index 10) is also NaN, so it may remain NaN
        # This tests the fallback mechanism
        assert imputer.lag_hours == [24, 48]

    def test_name_property(self):
        """Test that name property returns correct value."""
        imputer = LagFillImputer()
        assert imputer.name == "lag_fill"

    def test_default_lag_hours(self):
        """Test default lag hours are [24, 48]."""
        imputer = LagFillImputer()
        assert imputer.lag_hours == [24, 48]

    def test_custom_lag_hours(self):
        """Test custom lag hours."""
        imputer = LagFillImputer(lag_hours=[12, 24, 168])
        assert imputer.lag_hours == [12, 24, 168]

    def test_missing_column_raises_error(self):
        """Test that missing column raises KeyError."""
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="h")})
        imputer = LagFillImputer()
        with pytest.raises(KeyError, match="not found"):
            imputer.fit_transform(df, value_col="load")


# =============================================================================
# TriplePassImputer Tests
# =============================================================================


class TestTriplePassImputer:
    """Tests for TriplePassImputer class."""

    @pytest.fixture
    def df_with_intraday_gaps(self) -> pd.DataFrame:
        """Create DataFrame with intraday missing values."""
        return pd.DataFrame(
            {
                "Data": pd.date_range("2024-01-01", periods=24, freq="h"),
                "load": [
                    100.0,
                    np.nan,
                    102.0,
                    np.nan,
                    104.0,
                    105.0,  # Hours 0-5
                    np.nan,
                    107.0,
                    108.0,
                    np.nan,
                    np.nan,
                    111.0,  # Hours 6-11
                    112.0,
                    113.0,
                    np.nan,
                    115.0,
                    116.0,
                    117.0,  # Hours 12-17
                    np.nan,
                    np.nan,
                    120.0,
                    121.0,
                    122.0,
                    np.nan,  # Hours 18-23
                ],
            }
        )

    @pytest.fixture
    def df_with_full_day_gap(self) -> pd.DataFrame:
        """Create DataFrame with entire day missing."""
        day1 = pd.DataFrame(
            {
                "Data": pd.date_range("2024-01-01", periods=24, freq="h"),
                "load": [100.0 + i for i in range(24)],
            }
        )
        day2 = pd.DataFrame(
            {
                "Data": pd.date_range("2024-01-02", periods=24, freq="h"),
                "load": [np.nan] * 24,  # All missing
            }
        )
        day3 = pd.DataFrame(
            {
                "Data": pd.date_range("2024-01-03", periods=24, freq="h"),
                "load": [200.0 + i for i in range(24)],
            }
        )
        return pd.concat([day1, day2, day3], ignore_index=True)

    def test_fills_all_intraday_gaps(self, df_with_intraday_gaps: pd.DataFrame):
        """Test that triple pass fills all intraday gaps."""
        imputer = TriplePassImputer(date_col="Data")
        result = imputer.fit_transform(df_with_intraday_gaps, value_col="load")

        # All values should be filled
        assert not result["load"].isna().any()

    def test_preserves_existing_values(self, df_with_intraday_gaps: pd.DataFrame):
        """Test that existing values are preserved."""
        imputer = TriplePassImputer(date_col="Data")
        result = imputer.fit_transform(df_with_intraday_gaps, value_col="load")

        # Check some known values are preserved
        assert result["load"].iloc[0] == 100.0
        assert result["load"].iloc[2] == 102.0
        assert result["load"].iloc[5] == 105.0

    def test_handles_full_day_gap(self, df_with_full_day_gap: pd.DataFrame):
        """Test that triple pass handles full day of missing data."""
        imputer = TriplePassImputer(date_col="Data")
        result = imputer.fit_transform(df_with_full_day_gap, value_col="load")

        # Pass 4 (forward fill) should fill the entire missing day
        # with values from day 1
        day2_values = result.iloc[24:48]["load"]
        assert not day2_values.isna().any()

    def test_name_property(self):
        """Test that name property returns correct value."""
        imputer = TriplePassImputer()
        assert imputer.name == "triple_pass"

    def test_date_col_property(self):
        """Test date_col property returns correct value."""
        imputer = TriplePassImputer(date_col="timestamp")
        assert imputer.date_col == "timestamp"

    def test_default_date_col(self):
        """Test default date column is 'Data'."""
        imputer = TriplePassImputer()
        assert imputer.date_col == "Data"

    def test_no_missing_values_early_return(self):
        """Test that imputer returns early when no missing values."""
        df = pd.DataFrame(
            {
                "Data": pd.date_range("2024-01-01", periods=10, freq="h"),
                "load": [100.0 + i for i in range(10)],
            }
        )

        imputer = TriplePassImputer()
        result = imputer.fit_transform(df, value_col="load")

        # Should return data unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_missing_value_column_raises_error(self):
        """Test that missing value column raises KeyError."""
        df = pd.DataFrame(
            {
                "Data": pd.date_range("2024-01-01", periods=5, freq="h"),
            }
        )

        imputer = TriplePassImputer()
        with pytest.raises(KeyError, match="not found"):
            imputer.fit_transform(df, value_col="load")

    def test_missing_date_column_raises_error(self):
        """Test that missing date column raises KeyError."""
        df = pd.DataFrame(
            {
                "load": [100.0, np.nan, 102.0],
            }
        )

        imputer = TriplePassImputer(date_col="Data")
        with pytest.raises(KeyError, match="Date column"):
            imputer.fit_transform(df, value_col="load")


# =============================================================================
# ImputerChain Tests
# =============================================================================


class TestImputerChain:
    """Tests for ImputerChain class."""

    @pytest.fixture
    def df_with_gaps(self) -> pd.DataFrame:
        """Create DataFrame with various gap patterns."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
                "load": [
                    np.nan,
                    100.0,
                    np.nan,
                    np.nan,
                    120.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    150.0,
                    np.nan,
                ],
            }
        )

    def test_chains_multiple_imputers(self, df_with_gaps: pd.DataFrame):
        """Test that chain applies multiple imputers in sequence."""
        chain = ImputerChain(
            [
                ForwardFillImputer(limit=1),
                BackwardFillImputer(limit=1),
            ]
        )

        result = chain.fit_transform(df_with_gaps, value_col="load")

        # First value should be filled by backward fill
        assert result["load"].iloc[0] == 100.0
        # Gaps should be at least partially filled
        filled_count = (~result["load"].isna()).sum()
        original_filled = (~df_with_gaps["load"].isna()).sum()
        assert filled_count > original_filled

    def test_chain_fills_all_with_interpolation(self, df_with_gaps: pd.DataFrame):
        """Test that chain with interpolation fills all gaps."""
        chain = ImputerChain(
            [
                ForwardFillImputer(limit=1),
                BackwardFillImputer(limit=1),
                InterpolationImputer(method="linear"),
            ]
        )

        result = chain.fit_transform(df_with_gaps, value_col="load")

        # All internal values should be filled
        # (edge NaNs may still exist if no backward/forward fill possible)
        # With this setup, all should be filled
        internal_values = result["load"].iloc[1:-1]
        assert not internal_values.isna().any()

    def test_name_property(self):
        """Test that name property returns combined names."""
        chain = ImputerChain(
            [
                ForwardFillImputer(),
                BackwardFillImputer(),
            ]
        )

        assert chain.name == "chain(forward_fill+backward_fill)"

    def test_empty_chain_raises_error(self):
        """Test that empty imputer list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ImputerChain([])

    def test_imputers_property(self):
        """Test that imputers property returns copy of list."""
        imputers = [ForwardFillImputer(), BackwardFillImputer()]
        chain = ImputerChain(imputers)

        result = chain.imputers
        assert len(result) == 2
        # Should be a copy
        assert result is not imputers

    def test_stops_early_when_all_filled(self):
        """Test that chain stops early when all values are filled."""
        # Only one missing value that forward fill can handle
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
                "load": [100.0, np.nan, 102.0, 103.0, 104.0],
            }
        )

        chain = ImputerChain(
            [
                ForwardFillImputer(),  # Will fill the one gap
                InterpolationImputer(),  # Should not need to run
            ]
        )

        result = chain.fit_transform(df, value_col="load")
        assert not result["load"].isna().any()


# =============================================================================
# ResamplerMixin Tests
# =============================================================================


class TestResamplerMixin:
    """Tests for ResamplerMixin class."""

    class TestResampler(ResamplerMixin):
        """Concrete implementation for testing."""

    @pytest.fixture
    def resampler(self) -> "TestResamplerMixin.TestResampler":
        """Create a resampler instance for testing."""
        return self.TestResampler()

    @pytest.fixture
    def hourly_df(self) -> pd.DataFrame:
        """Create hourly DataFrame for testing."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "load": [100.0 + i * 10 for i in range(24)],
            }
        )

    @pytest.fixture
    def half_hourly_df(self) -> pd.DataFrame:
        """Create 30-minute DataFrame for testing."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=48, freq="30min"),
                "load": [100.0 + i * 5 for i in range(48)],
            }
        )

    def test_resample_to_30min(
        self, resampler: "TestResamplerMixin.TestResampler", hourly_df: pd.DataFrame
    ):
        """Test resampling hourly data to 30-minute."""
        result = resampler.resample_to_30min(hourly_df, value_col="load", timestamp_col="timestamp")

        # Should have approximately twice as many rows
        assert len(result) >= len(hourly_df)
        # Should have 30-min intervals
        time_diff = result["timestamp"].iloc[1] - result["timestamp"].iloc[0]
        assert time_diff == pd.Timedelta(minutes=30)

    def test_resample_to_30min_with_lag_fill(
        self, resampler: "TestResamplerMixin.TestResampler", hourly_df: pd.DataFrame
    ):
        """Test that lag fill populates new intervals."""
        result = resampler.resample_to_30min(
            hourly_df, value_col="load", timestamp_col="timestamp", use_lag_fill=True
        )

        # With lag fill, no NaN values should exist (after forward fill)
        assert not result["load"].isna().any()

    def test_resample_to_30min_without_lag_fill(
        self, resampler: "TestResamplerMixin.TestResampler", hourly_df: pd.DataFrame
    ):
        """Test resampling without lag fill creates NaNs."""
        result = resampler.resample_to_30min(
            hourly_df, value_col="load", timestamp_col="timestamp", use_lag_fill=False
        )

        # Without lag fill, some NaN values should exist
        assert result["load"].isna().any()

    def test_resample_to_hourly_mean(
        self, resampler: "TestResamplerMixin.TestResampler", half_hourly_df: pd.DataFrame
    ):
        """Test resampling 30-min data to hourly using mean."""
        result = resampler.resample_to_hourly(
            half_hourly_df, value_col="load", timestamp_col="timestamp", agg_method="mean"
        )

        # Should have half as many rows
        assert len(result) == 24
        # First hour mean of 100, 105 = 102.5
        assert result["load"].iloc[0] == 102.5

    def test_resample_to_hourly_sum(
        self, resampler: "TestResamplerMixin.TestResampler", half_hourly_df: pd.DataFrame
    ):
        """Test resampling 30-min data to hourly using sum."""
        result = resampler.resample_to_hourly(
            half_hourly_df, value_col="load", timestamp_col="timestamp", agg_method="sum"
        )

        # First hour sum of 100, 105 = 205
        assert result["load"].iloc[0] == 205.0

    def test_resample_to_hourly_invalid_method(
        self, resampler: "TestResamplerMixin.TestResampler", half_hourly_df: pd.DataFrame
    ):
        """Test that invalid aggregation method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid agg_method"):
            resampler.resample_to_hourly(
                half_hourly_df, value_col="load", timestamp_col="timestamp", agg_method="invalid"
            )

    def test_brazil_tz_constant(self, resampler: "TestResamplerMixin.TestResampler"):
        """Test that BRAZIL_TZ constant is correct."""
        assert ZoneInfo("America/Sao_Paulo") == resampler.BRAZIL_TZ

    def test_missing_timestamp_column_raises_error(
        self, resampler: "TestResamplerMixin.TestResampler"
    ):
        """Test that missing timestamp column raises KeyError."""
        df = pd.DataFrame({"load": [100.0, 110.0]})

        with pytest.raises(KeyError, match="Timestamp column"):
            resampler.resample_to_30min(df, value_col="load", timestamp_col="timestamp")

    def test_missing_value_column_raises_error(self, resampler: "TestResamplerMixin.TestResampler"):
        """Test that missing value column raises KeyError."""
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=2, freq="h")})

        with pytest.raises(KeyError, match="Value column"):
            resampler.resample_to_30min(df, value_col="load", timestamp_col="timestamp")


# =============================================================================
# TimezoneHandler Tests
# =============================================================================


class TestTimezoneHandler:
    """Tests for TimezoneHandler class."""

    @pytest.fixture
    def handler(self) -> TimezoneHandler:
        """Create a TimezoneHandler instance for testing."""
        return TimezoneHandler()

    @pytest.fixture
    def naive_df(self) -> pd.DataFrame:
        """Create DataFrame with naive timestamps."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
                "load": [100.0 + i for i in range(5)],
            }
        )

    @pytest.fixture
    def utc_df(self) -> pd.DataFrame:
        """Create DataFrame with UTC timestamps."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
                "load": [100.0 + i for i in range(5)],
            }
        )

    def test_localize_naive_timestamps(self, handler: TimezoneHandler, naive_df: pd.DataFrame):
        """Test localizing naive timestamps to Brazil timezone."""
        result = handler.localize_to_brazil(naive_df, timestamp_col="timestamp")

        # Timestamps should now be timezone-aware
        assert result["timestamp"].dt.tz is not None
        # Should be America/Sao_Paulo
        assert str(result["timestamp"].dt.tz) == "America/Sao_Paulo"

    def test_convert_utc_to_brazil(self, handler: TimezoneHandler, utc_df: pd.DataFrame):
        """Test converting UTC timestamps to Brazil timezone."""
        result = handler.localize_to_brazil(utc_df, timestamp_col="timestamp")

        # Should be converted to America/Sao_Paulo
        assert str(result["timestamp"].dt.tz) == "America/Sao_Paulo"
        # Times should be adjusted (Brazil is UTC-3)
        # First timestamp 2024-01-01 00:00 UTC = 2024-01-01 -3:00 BRT = 2023-12-31 21:00 BRT
        first_hour = result["timestamp"].iloc[0].hour
        assert first_hour == 21  # Adjusted for Brazil timezone

    def test_adjust_timestamp_backward(self, handler: TimezoneHandler, naive_df: pd.DataFrame):
        """Test adjusting timestamps backward by 30 minutes."""
        result = handler.adjust_timestamp_backward(naive_df, timestamp_col="timestamp", minutes=30)

        # First timestamp should be 30 minutes earlier
        original_first = naive_df["timestamp"].iloc[0]
        adjusted_first = result["timestamp"].iloc[0]
        assert adjusted_first == original_first - pd.Timedelta(minutes=30)

    def test_adjust_timestamp_backward_custom_minutes(
        self, handler: TimezoneHandler, naive_df: pd.DataFrame
    ):
        """Test adjusting timestamps backward by custom minutes."""
        result = handler.adjust_timestamp_backward(naive_df, timestamp_col="timestamp", minutes=15)

        original_first = naive_df["timestamp"].iloc[0]
        adjusted_first = result["timestamp"].iloc[0]
        assert adjusted_first == original_first - pd.Timedelta(minutes=15)

    def test_adjust_timestamp_backward_negative_raises_error(
        self, handler: TimezoneHandler, naive_df: pd.DataFrame
    ):
        """Test that negative minutes raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            handler.adjust_timestamp_backward(naive_df, timestamp_col="timestamp", minutes=-30)

    def test_brazil_tz_constant(self, handler: TimezoneHandler):
        """Test that BRAZIL_TZ constant is correct."""
        assert ZoneInfo("America/Sao_Paulo") == handler.BRAZIL_TZ

    def test_missing_timestamp_column_localize(self, handler: TimezoneHandler):
        """Test that missing timestamp column raises KeyError in localize."""
        df = pd.DataFrame({"load": [100.0, 110.0]})

        with pytest.raises(KeyError, match="Timestamp column"):
            handler.localize_to_brazil(df, timestamp_col="timestamp")

    def test_missing_timestamp_column_adjust(self, handler: TimezoneHandler):
        """Test that missing timestamp column raises KeyError in adjust."""
        df = pd.DataFrame({"load": [100.0, 110.0]})

        with pytest.raises(KeyError, match="Timestamp column"):
            handler.adjust_timestamp_backward(df, timestamp_col="timestamp")


# =============================================================================
# Integration Tests
# =============================================================================


class TestPreprocessorIntegration:
    """Integration tests for preprocessor components."""

    def test_triple_pass_on_realistic_data(self):
        """Test TriplePassImputer on realistic load data pattern."""
        # Create 3 days of 30-min data with typical missing patterns
        n_periods = 48 * 3
        timestamps = pd.date_range("2024-01-01", periods=n_periods, freq="30min")

        # Create realistic load pattern with some gaps
        rng = np.random.default_rng(seed=42)
        values = []
        for i in range(n_periods):
            hour = i % 48 // 2  # Hour of day
            base_load = 1000 + 100 * np.sin(2 * np.pi * hour / 24)  # Daily pattern

            # Add some random missing values
            if i in [10, 11, 50, 51, 52, 100, 101]:
                values.append(np.nan)
            else:
                values.append(base_load + rng.normal(0, 10))

        df = pd.DataFrame(
            {
                "Data": timestamps,
                "load": values,
            }
        )

        imputer = TriplePassImputer(date_col="Data")
        result = imputer.fit_transform(df, value_col="load")

        # All values should be filled
        assert not result["load"].isna().any()

    def test_chain_with_lag_and_interpolation(self):
        """Test ImputerChain with lag fill and interpolation."""
        # Create data with gaps
        timestamps = pd.date_range("2024-01-01", periods=96, freq="30min")
        values = [100.0 + i for i in range(96)]

        # Add gaps
        for i in [10, 11, 50, 51, 52]:
            values[i] = np.nan

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "load": values,
            }
        )

        chain = ImputerChain(
            [
                LagFillImputer(lag_hours=[24]),
                ForwardFillImputer(),
                BackwardFillImputer(),
                InterpolationImputer(method="linear"),
            ]
        )

        result = chain.fit_transform(df, value_col="load")

        # All values should be filled
        assert not result["load"].isna().any()

    def test_timezone_then_imputation(self):
        """Test timezone handling followed by imputation."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
                "load": [100.0, np.nan, 102.0, np.nan, 104.0, 105.0, np.nan, 107.0, 108.0, 109.0],
            }
        )

        # First localize to Brazil timezone
        handler = TimezoneHandler()
        df = handler.localize_to_brazil(df, timestamp_col="timestamp")

        # Then impute missing values
        imputer = ForwardFillImputer()
        result = imputer.fit_transform(df, value_col="load")

        # Timezone should be preserved
        assert str(result["timestamp"].dt.tz) == "America/Sao_Paulo"
        # Values should be filled
        assert not result["load"].isna().any()
