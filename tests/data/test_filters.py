"""Tests for data filtering and validation utilities.

This module contains comprehensive tests for the filtering functionality
in the PrevCarga system, including complete day filtering, duplicate handling,
and chronological validation.
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.data.filters import (
    ChronologicalValidator,
    CompleteDayFilter,
    DuplicateHandler,
    FilterConfig,
    FilteringReport,
)

# ruff: noqa: A001


# =============================================================================
# FilterConfig Tests
# =============================================================================


class TestFilterConfig:
    """Tests for FilterConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FilterConfig()

        assert config.semi_hourly_records == 48
        assert config.hourly_records == 24
        assert config.max_missing_pct == 0.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FilterConfig(
            semi_hourly_records=96,
            hourly_records=48,
            max_missing_pct=5.0,
        )

        assert config.semi_hourly_records == 96
        assert config.hourly_records == 48
        assert config.max_missing_pct == 5.0

    def test_max_missing_pct_bounds(self):
        """Test max_missing_pct boundary validation."""
        # Valid at lower bound
        config = FilterConfig(max_missing_pct=0.0)
        assert config.max_missing_pct == 0.0

        # Valid at upper bound
        config = FilterConfig(max_missing_pct=100.0)
        assert config.max_missing_pct == 100.0

    def test_max_missing_pct_invalid(self):
        """Test that invalid max_missing_pct raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FilterConfig(max_missing_pct=-1.0)

        with pytest.raises(ValidationError):
            FilterConfig(max_missing_pct=101.0)

    def test_semi_hourly_records_bounds(self):
        """Test semi_hourly_records boundary validation."""
        from pydantic import ValidationError

        # Valid bounds
        config = FilterConfig(semi_hourly_records=1)
        assert config.semi_hourly_records == 1

        config = FilterConfig(semi_hourly_records=96)
        assert config.semi_hourly_records == 96

        # Invalid bounds
        with pytest.raises(ValidationError):
            FilterConfig(semi_hourly_records=0)

        with pytest.raises(ValidationError):
            FilterConfig(semi_hourly_records=97)

    def test_hourly_records_bounds(self):
        """Test hourly_records boundary validation."""
        from pydantic import ValidationError

        # Valid bounds
        config = FilterConfig(hourly_records=1)
        assert config.hourly_records == 1

        config = FilterConfig(hourly_records=48)
        assert config.hourly_records == 48

        # Invalid bounds
        with pytest.raises(ValidationError):
            FilterConfig(hourly_records=0)

        with pytest.raises(ValidationError):
            FilterConfig(hourly_records=49)


# =============================================================================
# FilteringReport Tests
# =============================================================================


class TestFilteringReport:
    """Tests for FilteringReport class."""

    def test_default_values(self):
        """Test default initialization values."""
        report = FilteringReport()

        assert report.total_days == 0
        assert report.complete_days == 0
        assert report.incomplete_days == 0
        assert report.reasons == {}
        assert report.affected_areas == set()

    def test_retention_rate_calculation(self):
        """Test retention_rate property calculation."""
        report = FilteringReport()
        report.total_days = 100
        report.complete_days = 95

        assert report.retention_rate == 0.95

    def test_retention_rate_zero_total(self):
        """Test retention_rate when total_days is zero."""
        report = FilteringReport()
        report.total_days = 0

        # Should return 1.0 to avoid division by zero
        assert report.retention_rate == 1.0

    def test_retention_rate_all_complete(self):
        """Test retention_rate when all days are complete."""
        report = FilteringReport()
        report.total_days = 50
        report.complete_days = 50

        assert report.retention_rate == 1.0

    def test_retention_rate_none_complete(self):
        """Test retention_rate when no days are complete."""
        report = FilteringReport()
        report.total_days = 50
        report.complete_days = 0

        assert report.retention_rate == 0.0

    def test_to_dict(self):
        """Test to_dict returns correct dictionary representation."""
        report = FilteringReport()
        report.total_days = 100
        report.complete_days = 95
        report.incomplete_days = 5
        report.reasons = {"insufficient_records": 3, "too_many_missing": 2}
        report.affected_areas = {"SP", "RJ"}

        result = report.to_dict()

        assert result["total_days"] == 100
        assert result["complete_days"] == 95
        assert result["incomplete_days"] == 5
        assert result["retention_rate"] == 0.95
        assert result["reasons"] == {"insufficient_records": 3, "too_many_missing": 2}
        assert result["affected_areas"] == ["RJ", "SP"]  # Sorted

    def test_to_dict_empty_report(self):
        """Test to_dict with empty report."""
        report = FilteringReport()
        result = report.to_dict()

        assert result["total_days"] == 0
        assert result["complete_days"] == 0
        assert result["incomplete_days"] == 0
        assert result["retention_rate"] == 1.0
        assert result["reasons"] == {}
        assert result["affected_areas"] == []

    def test_log_summary(self):
        """Test that log_summary executes without error."""
        report = FilteringReport()
        report.total_days = 100
        report.complete_days = 90
        report.incomplete_days = 10
        report.reasons = {"insufficient_records": 5}
        report.affected_areas = {"SP"}

        # log_summary should execute without raising an exception
        report.log_summary()

        # Verify the report state is unchanged
        assert report.total_days == 100
        assert report.complete_days == 90
        assert report.incomplete_days == 10

    def test_log_summary_no_issues(self):
        """Test log_summary with no filtering issues."""
        report = FilteringReport()
        report.total_days = 100
        report.complete_days = 100
        report.incomplete_days = 0

        # log_summary should execute without raising an exception
        report.log_summary()


# =============================================================================
# CompleteDayFilter Tests
# =============================================================================


class TestCompleteDayFilter:
    """Tests for CompleteDayFilter class."""

    @pytest.fixture
    def complete_semi_hourly_day(self) -> pd.DataFrame:
        """Create a DataFrame with one complete semi-hourly day (48 records)."""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(minutes=30 * i) for i in range(48)]
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "cod_area": ["SP"] * 48,
                "load": [100.0 + i for i in range(48)],
            }
        )

    @pytest.fixture
    def complete_hourly_day(self) -> pd.DataFrame:
        """Create a DataFrame with one complete hourly day (24 records)."""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(24)]
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "cod_area": ["SP"] * 24,
                "load": [100.0 + i for i in range(24)],
            }
        )

    @pytest.fixture
    def incomplete_day(self) -> pd.DataFrame:
        """Create a DataFrame with one incomplete day (only 40 records)."""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(minutes=30 * i) for i in range(40)]
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "cod_area": ["SP"] * 40,
                "load": [100.0 + i for i in range(40)],
            }
        )

    def test_filter_complete_semi_hourly_day(self, complete_semi_hourly_day: pd.DataFrame):
        """Test filtering with a complete semi-hourly day."""
        filter = CompleteDayFilter()
        filtered_df, report = filter.filter(complete_semi_hourly_day)

        assert len(filtered_df) == 48
        assert report.total_days == 1
        assert report.complete_days == 1
        assert report.incomplete_days == 0
        assert report.retention_rate == 1.0

    def test_filter_complete_hourly_day(self, complete_hourly_day: pd.DataFrame):
        """Test filtering with a complete hourly day."""
        filter = CompleteDayFilter()
        filtered_df, report = filter.filter(complete_hourly_day)

        assert len(filtered_df) == 24
        assert report.total_days == 1
        assert report.complete_days == 1
        assert report.incomplete_days == 0

    def test_filter_incomplete_day(self, incomplete_day: pd.DataFrame):
        """Test filtering removes incomplete day."""
        filter = CompleteDayFilter()
        filtered_df, report = filter.filter(incomplete_day)

        assert len(filtered_df) == 0
        assert report.total_days == 1
        assert report.complete_days == 0
        assert report.incomplete_days == 1
        assert "insufficient_records" in report.reasons

    def test_filter_mixed_days(self, complete_semi_hourly_day: pd.DataFrame):
        """Test filtering with mix of complete and incomplete days."""
        # Add an incomplete second day
        base_date = datetime(2024, 1, 2)
        incomplete_timestamps = [base_date + timedelta(minutes=30 * i) for i in range(40)]
        incomplete_df = pd.DataFrame(
            {
                "timestamp": incomplete_timestamps,
                "cod_area": ["SP"] * 40,
                "load": [200.0 + i for i in range(40)],
            }
        )

        combined_df = pd.concat([complete_semi_hourly_day, incomplete_df], ignore_index=True)

        filter = CompleteDayFilter()
        filtered_df, report = filter.filter(combined_df)

        assert len(filtered_df) == 48  # Only complete day retained
        assert report.total_days == 2
        assert report.complete_days == 1
        assert report.incomplete_days == 1
        assert report.retention_rate == 0.5

    def test_filter_multiple_areas(self):
        """Test filtering with multiple areas."""
        # Create complete day for SP, incomplete for RJ
        base_date = datetime(2024, 1, 1)

        sp_timestamps = [base_date + timedelta(minutes=30 * i) for i in range(48)]
        sp_df = pd.DataFrame(
            {
                "timestamp": sp_timestamps,
                "cod_area": ["SP"] * 48,
                "load": [100.0] * 48,
            }
        )

        rj_timestamps = [base_date + timedelta(minutes=30 * i) for i in range(40)]
        rj_df = pd.DataFrame(
            {
                "timestamp": rj_timestamps,
                "cod_area": ["RJ"] * 40,
                "load": [200.0] * 40,
            }
        )

        combined_df = pd.concat([sp_df, rj_df], ignore_index=True)

        filter = CompleteDayFilter()
        filtered_df, report = filter.filter(combined_df)

        assert len(filtered_df) == 48  # Only SP retained
        assert report.total_days == 2  # SP-day1, RJ-day1
        assert report.complete_days == 1
        assert report.incomplete_days == 1
        assert "RJ" in report.affected_areas
        assert "SP" not in report.affected_areas

    def test_filter_with_missing_values(self, complete_semi_hourly_day: pd.DataFrame):
        """Test filtering with missing value threshold."""
        df = complete_semi_hourly_day.copy()
        # Add 10% missing values (about 5 records)
        df.loc[0:4, "load"] = None

        # With 0% max missing, should filter out
        filter_strict = CompleteDayFilter(FilterConfig(max_missing_pct=0.0))
        filtered_df, report = filter_strict.filter(df, value_cols=["load"])

        assert len(filtered_df) == 0
        assert report.incomplete_days == 1
        assert "too_many_missing" in report.reasons

        # With 15% max missing, should retain
        filter_lenient = CompleteDayFilter(FilterConfig(max_missing_pct=15.0))
        filtered_df, report = filter_lenient.filter(df, value_cols=["load"])

        assert len(filtered_df) == 48
        assert report.complete_days == 1

    def test_filter_empty_dataframe(self):
        """Test filtering with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["timestamp", "cod_area", "load"])

        filter = CompleteDayFilter()
        filtered_df, report = filter.filter(empty_df)

        assert len(filtered_df) == 0
        assert report.total_days == 0
        assert report.complete_days == 0

    def test_filter_missing_timestamp_column(self):
        """Test that missing timestamp column raises KeyError."""
        df = pd.DataFrame({"cod_area": ["SP"], "load": [100.0]})

        filter = CompleteDayFilter()
        with pytest.raises(KeyError) as exc_info:
            filter.filter(df)

        assert "timestamp" in str(exc_info.value)

    def test_filter_missing_area_column(self):
        """Test that missing area column raises KeyError."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "load": [100.0],
            }
        )

        filter = CompleteDayFilter()
        with pytest.raises(KeyError) as exc_info:
            filter.filter(df, area_col="cod_area")

        assert "cod_area" in str(exc_info.value)

    def test_filter_no_area_column(self):
        """Test filtering without area grouping."""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(minutes=30 * i) for i in range(48)]
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "load": [100.0 + i for i in range(48)],
            }
        )

        filter = CompleteDayFilter()
        filtered_df, report = filter.filter(df, area_col=None)

        assert len(filtered_df) == 48
        assert report.complete_days == 1

    def test_filter_auto_detect_semi_hourly(self):
        """Test auto-detection of semi-hourly frequency."""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(minutes=30 * i) for i in range(48)]
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "cod_area": ["SP"] * 48,
                "load": [100.0] * 48,
            }
        )

        filter = CompleteDayFilter()
        filtered_df, _report = filter.filter(df)

        # Should detect 30-min intervals and expect 48 records
        assert len(filtered_df) == 48

    def test_filter_auto_detect_hourly(self):
        """Test auto-detection of hourly frequency."""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(24)]
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "cod_area": ["SP"] * 24,
                "load": [100.0] * 24,
            }
        )

        filter = CompleteDayFilter()
        filtered_df, _report = filter.filter(df)

        # Should detect 60-min intervals and expect 24 records
        assert len(filtered_df) == 24

    def test_filter_single_record(self):
        """Test filtering with single record."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "cod_area": ["SP"],
                "load": [100.0],
            }
        )

        filter = CompleteDayFilter()
        filtered_df, report = filter.filter(df)

        # Single record day should be incomplete
        assert len(filtered_df) == 0
        assert report.incomplete_days == 1

    def test_filter_excess_records(self):
        """Test filtering day with excess records."""
        # Create a day with 50 records all within the same day (using smaller intervals)
        base_date = datetime(2024, 1, 1)
        # Use 28-minute intervals so all 50 records fit within 24 hours
        # This creates 50 records for a single day, which exceeds the expected 48
        timestamps = [base_date + timedelta(minutes=28 * i) for i in range(50)]
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "cod_area": ["SP"] * 50,
                "load": [100.0] * 50,
            }
        )

        filter = CompleteDayFilter()
        filtered_df, report = filter.filter(df)

        # The day has 50 records but expected is 48, so it should be filtered
        assert len(filtered_df) == 0
        assert "excess_records" in report.reasons

    def test_filter_custom_config(self):
        """Test filtering with custom configuration."""
        config = FilterConfig(
            semi_hourly_records=48,
            max_missing_pct=10.0,
        )
        filter = CompleteDayFilter(config=config)

        assert filter.config.semi_hourly_records == 48
        assert filter.config.max_missing_pct == 10.0

    def test_filter_preserves_columns(self, complete_semi_hourly_day: pd.DataFrame):
        """Test that filtering preserves all original columns."""
        df = complete_semi_hourly_day.copy()
        df["extra_col"] = "test"

        filter = CompleteDayFilter()
        filtered_df, _report = filter.filter(df)

        assert "extra_col" in filtered_df.columns
        assert "timestamp" in filtered_df.columns
        assert "cod_area" in filtered_df.columns
        assert "load" in filtered_df.columns
        # Internal column should be removed
        assert "_filter_date" not in filtered_df.columns

    def test_filter_performance_one_year(self):
        """Test performance with approximately one year of data."""
        # ~17,520 records for semi-hourly data (365 days * 48)
        base_date = datetime(2024, 1, 1)
        num_records = 365 * 48
        timestamps = [base_date + timedelta(minutes=30 * i) for i in range(num_records)]
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "cod_area": ["SP"] * num_records,
                "load": [100.0] * num_records,
            }
        )

        filter = CompleteDayFilter()

        start_time = time.time()
        filtered_df, report = filter.filter(df)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 1.0, f"Filter took {elapsed_time:.2f}s, expected < 1s"
        assert report.total_days == 365
        assert report.complete_days == 365
        assert len(filtered_df) == num_records

    def test_filter_multiple_value_columns(self):
        """Test missing value check across multiple columns."""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(minutes=30 * i) for i in range(48)]
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "cod_area": ["SP"] * 48,
                "load1": [100.0] * 48,
                "load2": [200.0] * 48,
            }
        )

        # Add missing values to both columns
        df.loc[0, "load1"] = None
        df.loc[1, "load2"] = None

        filter = CompleteDayFilter(FilterConfig(max_missing_pct=0.0))
        filtered_df, report = filter.filter(df, value_cols=["load1", "load2"])

        assert len(filtered_df) == 0
        assert "too_many_missing" in report.reasons

    def test_filter_nonexistent_value_column(self):
        """Test with value column that doesn't exist."""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(minutes=30 * i) for i in range(48)]
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "cod_area": ["SP"] * 48,
                "load": [100.0] * 48,
            }
        )

        day_filter = CompleteDayFilter()
        # Should not raise, just ignore non-existent column
        filtered_df, _report = day_filter.filter(df, value_cols=["nonexistent"])

        assert len(filtered_df) == 48


# =============================================================================
# DuplicateHandler Tests
# =============================================================================


class TestDuplicateHandler:
    """Tests for DuplicateHandler class."""

    def test_no_duplicates(self):
        """Test detection with no duplicates."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00",
                        "2024-01-01 00:30:00",
                        "2024-01-01 01:00:00",
                    ]
                ),
                "cod_area": ["SP", "SP", "SP"],
                "value": [100, 110, 120],
            }
        )

        deduplicated_df, num_removed = DuplicateHandler.detect_and_remove(
            df, key_columns=["timestamp", "cod_area"]
        )

        assert num_removed == 0
        assert len(deduplicated_df) == 3

    def test_removes_duplicates(self):
        """Test that duplicates are removed, keeping first."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00",
                        "2024-01-01 00:00:00",  # Duplicate
                        "2024-01-01 01:00:00",
                    ]
                ),
                "cod_area": ["SP", "SP", "SP"],
                "value": [100, 999, 120],  # Second value is different
            }
        )

        deduplicated_df, num_removed = DuplicateHandler.detect_and_remove(
            df, key_columns=["timestamp", "cod_area"]
        )

        assert num_removed == 1
        assert len(deduplicated_df) == 2
        # First occurrence should be kept
        assert deduplicated_df.iloc[0]["value"] == 100

    def test_multiple_duplicates(self):
        """Test removal of multiple duplicate sets."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00",
                        "2024-01-01 00:00:00",
                        "2024-01-01 00:00:00",  # Three duplicates
                        "2024-01-01 01:00:00",
                        "2024-01-01 01:00:00",  # Two duplicates
                    ]
                ),
                "cod_area": ["SP", "SP", "SP", "RJ", "RJ"],
                "value": [100, 100, 100, 200, 200],
            }
        )

        deduplicated_df, num_removed = DuplicateHandler.detect_and_remove(
            df, key_columns=["timestamp", "cod_area"]
        )

        assert num_removed == 3
        assert len(deduplicated_df) == 2

    def test_empty_dataframe(self):
        """Test removal from empty DataFrame."""
        df = pd.DataFrame(columns=["timestamp", "cod_area", "value"])

        deduplicated_df, num_removed = DuplicateHandler.detect_and_remove(
            df, key_columns=["timestamp", "cod_area"]
        )

        assert num_removed == 0
        assert len(deduplicated_df) == 0

    def test_single_key_column(self):
        """Test removal with single key column."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 2, 3],
                "value": [100, 200, 200, 300],
            }
        )

        deduplicated_df, num_removed = DuplicateHandler.detect_and_remove(df, key_columns=["id"])

        assert num_removed == 1
        assert len(deduplicated_df) == 3

    def test_empty_key_columns(self):
        """Test that empty key_columns raises ValueError."""
        df = pd.DataFrame({"timestamp": [datetime(2024, 1, 1)], "value": [100]})

        with pytest.raises(ValueError) as exc_info:
            DuplicateHandler.detect_and_remove(df, key_columns=[])

        assert "empty" in str(exc_info.value).lower()

    def test_missing_key_column(self):
        """Test removal when key column doesn't exist."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01"]),
                "value": [100],
            }
        )

        deduplicated_df, num_removed = DuplicateHandler.detect_and_remove(
            df, key_columns=["nonexistent_col"]
        )

        # Should return original df unchanged
        assert num_removed == 0
        assert len(deduplicated_df) == 1

    def test_partial_key_columns_exist(self):
        """Test removal when only some key columns exist."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00",
                        "2024-01-01 00:00:00",
                    ]
                ),
                "value": [100, 100],
            }
        )

        deduplicated_df, num_removed = DuplicateHandler.detect_and_remove(
            df, key_columns=["timestamp", "nonexistent"]
        )

        # Should use only existing column
        assert num_removed == 1
        assert len(deduplicated_df) == 1

    def test_returns_copy(self):
        """Test that a copy is returned, not a view."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01"]),
                "value": [100],
            }
        )

        deduplicated_df, _num_removed = DuplicateHandler.detect_and_remove(
            df, key_columns=["timestamp"]
        )

        # Modify the returned df
        deduplicated_df.loc[0, "value"] = 999

        # Original should be unchanged
        assert df.loc[0, "value"] == 100


# =============================================================================
# ChronologicalValidator Tests
# =============================================================================


class TestChronologicalValidator:
    """Tests for ChronologicalValidator class."""

    def test_already_sorted(self):
        """Test validation with already sorted data."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00",
                        "2024-01-01 01:00:00",
                        "2024-01-01 02:00:00",
                    ]
                ),
                "cod_area": ["SP", "SP", "SP"],
            }
        )

        result = ChronologicalValidator.validate_and_sort(df)

        assert len(result) == 3
        assert result["timestamp"].is_monotonic_increasing

    def test_auto_sort_unsorted(self):
        """Test auto-sorting of unsorted data."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 02:00:00",
                        "2024-01-01 00:00:00",
                        "2024-01-01 01:00:00",
                    ]
                ),
                "cod_area": ["SP", "SP", "SP"],
            }
        )

        result = ChronologicalValidator.validate_and_sort(df, auto_sort=True)

        assert result["timestamp"].is_monotonic_increasing
        assert result.iloc[0]["timestamp"] == pd.Timestamp("2024-01-01 00:00:00")
        assert result.iloc[2]["timestamp"] == pd.Timestamp("2024-01-01 02:00:00")

    def test_raise_if_unsorted(self):
        """Test that ValueError is raised when auto_sort=False and unsorted."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 02:00:00",
                        "2024-01-01 00:00:00",
                    ]
                ),
                "cod_area": ["SP", "SP"],
            }
        )

        with pytest.raises(ValueError) as exc_info:
            ChronologicalValidator.validate_and_sort(df, auto_sort=False)

        assert "not chronologically sorted" in str(exc_info.value)

    def test_sorted_within_groups(self):
        """Test validation with sorted data within groups."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00",
                        "2024-01-01 01:00:00",
                        "2024-01-01 00:00:00",  # RJ starts here
                        "2024-01-01 01:00:00",
                    ]
                ),
                "cod_area": ["SP", "SP", "RJ", "RJ"],
            }
        )

        result = ChronologicalValidator.validate_and_sort(df, group_col="cod_area", auto_sort=False)

        # Should pass validation as each group is sorted
        assert len(result) == 4

    def test_unsorted_within_groups(self):
        """Test detection of unsorted data within groups."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 01:00:00",  # SP out of order
                        "2024-01-01 00:00:00",
                        "2024-01-01 00:00:00",
                        "2024-01-01 01:00:00",
                    ]
                ),
                "cod_area": ["SP", "SP", "RJ", "RJ"],
            }
        )

        with pytest.raises(ValueError):
            ChronologicalValidator.validate_and_sort(df, group_col="cod_area", auto_sort=False)

    def test_sort_by_group_and_timestamp(self):
        """Test sorting by both group and timestamp."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 01:00:00",
                        "2024-01-01 00:00:00",
                        "2024-01-01 01:00:00",
                        "2024-01-01 00:00:00",
                    ]
                ),
                "cod_area": ["RJ", "SP", "SP", "RJ"],
                "value": [1, 2, 3, 4],
            }
        )

        result = ChronologicalValidator.validate_and_sort(df, group_col="cod_area", auto_sort=True)

        # Should be sorted by area then timestamp
        assert result.iloc[0]["cod_area"] == "RJ"
        assert result.iloc[0]["timestamp"] == pd.Timestamp("2024-01-01 00:00:00")
        assert result.iloc[1]["cod_area"] == "RJ"
        assert result.iloc[1]["timestamp"] == pd.Timestamp("2024-01-01 01:00:00")

    def test_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame(columns=["timestamp", "cod_area"])

        result = ChronologicalValidator.validate_and_sort(df)

        assert len(result) == 0

    def test_single_record(self):
        """Test validation with single record."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 00:00:00"]),
                "cod_area": ["SP"],
            }
        )

        result = ChronologicalValidator.validate_and_sort(df, auto_sort=False)

        assert len(result) == 1

    def test_missing_timestamp_column(self):
        """Test that missing timestamp column raises KeyError."""
        df = pd.DataFrame({"cod_area": ["SP"], "value": [100]})

        with pytest.raises(KeyError) as exc_info:
            ChronologicalValidator.validate_and_sort(df)

        assert "timestamp" in str(exc_info.value)

    def test_missing_group_column(self):
        """Test that missing group column raises KeyError."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01"]),
                "value": [100],
            }
        )

        with pytest.raises(KeyError) as exc_info:
            ChronologicalValidator.validate_and_sort(df, group_col="cod_area")

        assert "cod_area" in str(exc_info.value)

    def test_no_group_column(self):
        """Test validation without grouping."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 02:00:00",
                        "2024-01-01 00:00:00",
                        "2024-01-01 01:00:00",
                    ]
                ),
                "value": [3, 1, 2],
            }
        )

        result = ChronologicalValidator.validate_and_sort(df, group_col=None, auto_sort=True)

        assert result["timestamp"].is_monotonic_increasing

    def test_custom_timestamp_column(self):
        """Test validation with custom timestamp column name."""
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2024-01-01 02:00:00",
                        "2024-01-01 00:00:00",
                    ]
                ),
                "area": ["SP", "SP"],
            }
        )

        result = ChronologicalValidator.validate_and_sort(
            df, timestamp_col="datetime", group_col="area", auto_sort=True
        )

        assert result["datetime"].is_monotonic_increasing

    def test_returns_copy(self):
        """Test that a copy is returned when already sorted."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01"]),
                "cod_area": ["SP"],
                "value": [100],
            }
        )

        result = ChronologicalValidator.validate_and_sort(df)

        # Modify the returned df
        result.loc[0, "value"] = 999

        # Original should be unchanged
        assert df.loc[0, "value"] == 100

    def test_resets_index(self):
        """Test that index is reset after sorting."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 02:00:00",
                        "2024-01-01 00:00:00",
                    ]
                ),
                "cod_area": ["SP", "SP"],
            }
        )

        result = ChronologicalValidator.validate_and_sort(df, auto_sort=True)

        # Index should be 0, 1 not original indices
        assert list(result.index) == [0, 1]


# =============================================================================
# Integration Tests
# =============================================================================


class TestFilterIntegration:
    """Integration tests for filter components working together."""

    def test_full_pipeline(self):
        """Test complete pipeline: deduplicate -> validate -> filter."""
        base_date = datetime(2024, 1, 1)

        # Create data with duplicates and out-of-order timestamps
        timestamps = (
            [base_date + timedelta(minutes=30 * i) for i in range(48)]
            + [base_date + timedelta(minutes=30 * 5)]  # Duplicate
            + [base_date + timedelta(days=1, minutes=30 * i) for i in range(40)]  # Incomplete day
        )
        areas = ["SP"] * 49 + ["SP"] * 40

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "cod_area": areas,
                "load": [100.0] * len(timestamps),
            }
        )

        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Step 1: Remove duplicates
        df, num_removed = DuplicateHandler.detect_and_remove(
            df, key_columns=["timestamp", "cod_area"]
        )
        assert num_removed == 1

        # Step 2: Sort chronologically
        df = ChronologicalValidator.validate_and_sort(
            df, timestamp_col="timestamp", group_col="cod_area", auto_sort=True
        )
        assert df["timestamp"].is_monotonic_increasing

        # Step 3: Filter to complete days
        filter = CompleteDayFilter()
        filtered_df, report = filter.filter(df)

        assert report.complete_days == 1
        assert report.incomplete_days == 1
        assert len(filtered_df) == 48

    def test_multi_area_pipeline(self):
        """Test pipeline with multiple areas."""
        base_date = datetime(2024, 1, 1)

        # SP: 2 complete days
        sp_timestamps = [base_date + timedelta(minutes=30 * i) for i in range(48 * 2)]

        # RJ: 1 complete day, 1 incomplete
        rj_timestamps = [base_date + timedelta(minutes=30 * i) for i in range(48)] + [
            base_date + timedelta(days=1, minutes=30 * i) for i in range(30)
        ]

        df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "timestamp": sp_timestamps,
                        "cod_area": ["SP"] * len(sp_timestamps),
                        "load": [100.0] * len(sp_timestamps),
                    }
                ),
                pd.DataFrame(
                    {
                        "timestamp": rj_timestamps,
                        "cod_area": ["RJ"] * len(rj_timestamps),
                        "load": [200.0] * len(rj_timestamps),
                    }
                ),
            ],
            ignore_index=True,
        )

        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Process
        df = ChronologicalValidator.validate_and_sort(
            df, timestamp_col="timestamp", group_col="cod_area", auto_sort=True
        )

        day_filter = CompleteDayFilter()
        _filtered_df, report = day_filter.filter(df)

        assert report.total_days == 4  # SP: 2, RJ: 2
        assert report.complete_days == 3  # SP: 2, RJ: 1
        assert report.incomplete_days == 1  # RJ day 2
        assert "RJ" in report.affected_areas


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_dst_transition_day(self):
        """Test handling of DST transition day (Brazil uses DST in some years)."""
        # Brazil DST typically starts in October and ends in February
        # This test uses naive timestamps which don't have DST issues
        base_date = datetime(2023, 10, 15)  # Example DST transition period
        timestamps = [base_date + timedelta(minutes=30 * i) for i in range(48)]

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "cod_area": ["SP"] * 48,
                "load": [100.0] * 48,
            }
        )

        filter = CompleteDayFilter()
        filtered_df, report = filter.filter(df)

        # Should still work with naive timestamps
        assert len(filtered_df) == 48
        assert report.complete_days == 1

    def test_year_boundary(self):
        """Test filtering across year boundary."""
        # Dec 31, 2023 and Jan 1, 2024
        dec31_timestamps = [datetime(2023, 12, 31) + timedelta(minutes=30 * i) for i in range(48)]
        jan1_timestamps = [datetime(2024, 1, 1) + timedelta(minutes=30 * i) for i in range(48)]

        df = pd.DataFrame(
            {
                "timestamp": dec31_timestamps + jan1_timestamps,
                "cod_area": ["SP"] * 96,
                "load": [100.0] * 96,
            }
        )

        filter = CompleteDayFilter()
        filtered_df, report = filter.filter(df)

        assert report.total_days == 2
        assert report.complete_days == 2
        assert len(filtered_df) == 96

    def test_leap_year_day(self):
        """Test filtering on leap year day (Feb 29)."""
        base_date = datetime(2024, 2, 29)  # 2024 is a leap year
        timestamps = [base_date + timedelta(minutes=30 * i) for i in range(48)]

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "cod_area": ["SP"] * 48,
                "load": [100.0] * 48,
            }
        )

        filter = CompleteDayFilter()
        filtered_df, report = filter.filter(df)

        assert len(filtered_df) == 48
        assert report.complete_days == 1

    def test_all_missing_values(self):
        """Test filtering day with all missing values."""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(minutes=30 * i) for i in range(48)]

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "cod_area": ["SP"] * 48,
                "load": [None] * 48,  # All missing
            }
        )

        filter = CompleteDayFilter(FilterConfig(max_missing_pct=0.0))
        filtered_df, report = filter.filter(df, value_cols=["load"])

        assert len(filtered_df) == 0
        assert "too_many_missing" in report.reasons

    def test_exactly_threshold_missing(self):
        """Test filtering at exactly the missing threshold."""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(minutes=30 * i) for i in range(48)]

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "cod_area": ["SP"] * 48,
                "load": [100.0] * 48,
            }
        )

        # Set exactly 10% missing (4.8 -> 5 records)
        df.loc[0:4, "load"] = None  # 5 out of 48 = 10.4%

        # With 10% threshold, should be filtered out (10.4% > 10%)
        filter = CompleteDayFilter(FilterConfig(max_missing_pct=10.0))
        filtered_df, _report = filter.filter(df, value_cols=["load"])

        assert len(filtered_df) == 0

        # With 11% threshold, should pass
        day_filter = CompleteDayFilter(FilterConfig(max_missing_pct=11.0))
        filtered_df, _report = day_filter.filter(df, value_cols=["load"])

        assert len(filtered_df) == 48
