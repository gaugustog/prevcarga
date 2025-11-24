"""Data filtering and validation utilities for PrevCarga.

This module provides classes for filtering datasets to complete days,
handling duplicates, and validating chronological order of time series data.

Example:
    ```python
    import pandas as pd
    from src.data.filters import (
        CompleteDayFilter,
        DuplicateHandler,
        ChronologicalValidator,
    )

    # Filter to complete days only
    day_filter = CompleteDayFilter()
    filtered_df, report = day_filter.filter(df)
    report.log_summary()

    # Remove duplicates
    df, num_removed = DuplicateHandler.detect_and_remove(
        df, key_columns=["timestamp", "cod_area"]
    )

    # Validate and sort chronologically
    df = ChronologicalValidator.validate_and_sort(
        df, timestamp_col="timestamp", group_col="cod_area"
    )
    ```
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.utils import get_logger

logger = get_logger(__name__)

# Constants for frequency detection
_HIGH_MISSING_THRESHOLD = 20.0
_MIN_RECORDS_FOR_DETECTION = 2
_SEMI_HOURLY_MIN_MINUTES = 25
_SEMI_HOURLY_MAX_MINUTES = 35
_HOURLY_MIN_MINUTES = 55
_HOURLY_MAX_MINUTES = 65


class FilterConfig(BaseModel):
    """Configuration for complete day filtering.

    Attributes:
        semi_hourly_records: Expected number of records per day for semi-hourly data.
        hourly_records: Expected number of records per day for hourly data.
        max_missing_pct: Maximum percentage of missing values allowed per day (0-100).

    Example:
        ```python
        from src.data.filters import FilterConfig

        config = FilterConfig(
            semi_hourly_records=48,
            hourly_records=24,
            max_missing_pct=5.0,
        )
        ```
    """

    semi_hourly_records: int = Field(default=48, ge=1, le=96)
    hourly_records: int = Field(default=24, ge=1, le=48)
    max_missing_pct: float = Field(default=0.0, ge=0.0, le=100.0)

    @field_validator("max_missing_pct")
    @classmethod
    def validate_max_missing_pct(cls, v: float) -> float:
        """Validate max_missing_pct and warn if high.

        Args:
            v: The max_missing_pct value to validate.

        Returns:
            The validated max_missing_pct value.
        """
        if v > _HIGH_MISSING_THRESHOLD:
            logger.warning(
                "High max_missing_pct threshold: %.1f%%. "
                "This may allow days with significant missing data.",
                v,
            )
        return v


class FilteringReport:
    """Report containing filtering results and statistics.

    Collects and summarizes filtering outcomes including counts of
    complete/incomplete days, reasons for filtering, and affected areas.

    Attributes:
        total_days: Total number of unique days in the input data.
        complete_days: Number of days that passed the completeness check.
        incomplete_days: Number of days filtered out due to incompleteness.
        reasons: Dictionary mapping reason descriptions to counts.
        affected_areas: Set of area codes affected by filtering.

    Example:
        ```python
        from src.data.filters import FilteringReport

        report = FilteringReport()
        report.total_days = 365
        report.complete_days = 350
        report.incomplete_days = 15
        report.reasons = {"insufficient_records": 10, "too_many_missing": 5}
        report.affected_areas = {"SP", "RJ"}
        report.log_summary()
        print(f"Retention rate: {report.retention_rate:.1%}")
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty filtering report."""
        self.total_days: int = 0
        self.complete_days: int = 0
        self.incomplete_days: int = 0
        self.reasons: dict[str, int] = {}
        self.affected_areas: set[str] = set()

    @property
    def retention_rate(self) -> float:
        """Calculate the retention rate of days after filtering.

        Returns:
            Float between 0 and 1 representing the proportion of days retained.
            Returns 1.0 if total_days is 0 to avoid division by zero.

        Example:
            ```python
            report = FilteringReport()
            report.total_days = 100
            report.complete_days = 95
            print(f"Retention rate: {report.retention_rate:.1%}")  # 95.0%
            ```
        """
        if self.total_days == 0:
            return 1.0
        return self.complete_days / self.total_days

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a dictionary representation.

        Returns:
            Dictionary containing all report attributes.

        Example:
            ```python
            report = FilteringReport()
            report.total_days = 100
            report_dict = report.to_dict()
            print(report_dict["total_days"])  # 100
            ```
        """
        return {
            "total_days": self.total_days,
            "complete_days": self.complete_days,
            "incomplete_days": self.incomplete_days,
            "retention_rate": self.retention_rate,
            "reasons": self.reasons,
            "affected_areas": sorted(self.affected_areas),
        }

    def log_summary(self) -> None:
        """Log a summary of the filtering report.

        Outputs filtering statistics to the logger at INFO level,
        including total, complete, and incomplete day counts, plus
        reason distribution and affected areas.

        Example:
            ```python
            report = FilteringReport()
            report.total_days = 100
            report.complete_days = 90
            report.incomplete_days = 10
            report.log_summary()
            # Logs: "Filtering complete: 100 total days, 90 complete, 10 incomplete"
            ```
        """
        logger.info(
            "Filtering complete: %d total days, %d complete, %d incomplete (%.1f%% retention)",
            self.total_days,
            self.complete_days,
            self.incomplete_days,
            self.retention_rate * 100,
        )
        if self.reasons:
            logger.info("Filtering reasons: %s", self.reasons)
        if self.affected_areas:
            logger.info("Affected areas: %s", sorted(self.affected_areas))


class CompleteDayFilter:
    """Filter datasets to retain only complete days.

    A complete day is defined as having the expected number of records
    (48 for semi-hourly data, 24 for hourly data) with missing values
    below a configurable threshold.

    The filter auto-detects the data frequency from timestamp intervals
    and applies appropriate completeness checks per area and date.

    Attributes:
        config: FilterConfig instance with filtering parameters.

    Example:
        ```python
        import pandas as pd
        from src.data.filters import CompleteDayFilter, FilterConfig

        # Create filter with custom configuration
        config = FilterConfig(max_missing_pct=5.0)
        day_filter = CompleteDayFilter(config=config)

        # Filter DataFrame to complete days
        filtered_df, report = day_filter.filter(
            df,
            timestamp_col="timestamp",
            area_col="cod_area",
            value_cols=["load"],
        )

        print(f"Retained {report.retention_rate:.1%} of days")
        ```
    """

    def __init__(self, config: FilterConfig | None = None) -> None:
        """Initialize the complete day filter.

        Args:
            config: FilterConfig instance with filtering parameters.
                If None, uses default configuration.
        """
        self._config = config if config is not None else FilterConfig()

    @property
    def config(self) -> FilterConfig:
        """Return the filter configuration."""
        return self._config

    def filter(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        area_col: str | None = "cod_area",
        value_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, FilteringReport]:
        """Filter DataFrame to retain only complete days.

        Analyzes each (area, date) combination and retains only those
        with the expected number of records and acceptable missing value
        percentage.

        Args:
            df: Input DataFrame containing time series data.
            timestamp_col: Name of the timestamp column.
            area_col: Name of the area column for grouping. If None,
                treats all data as a single group.
            value_cols: List of column names to check for missing values.
                If None, no missing value check is performed.

        Returns:
            Tuple of (filtered_df, report) where filtered_df contains
            only complete days and report contains filtering statistics.

        Raises:
            KeyError: If timestamp_col is not found in DataFrame.
            ValueError: If DataFrame is missing required columns.

        Example:
            ```python
            day_filter = CompleteDayFilter()
            filtered_df, report = day_filter.filter(
                df,
                timestamp_col="timestamp",
                area_col="cod_area",
                value_cols=["load"],
            )
            ```
        """
        report = FilteringReport()

        if df.empty:
            logger.debug("CompleteDayFilter: empty DataFrame, returning empty result")
            return df.copy(), report

        if timestamp_col not in df.columns:
            msg = f"Timestamp column '{timestamp_col}' not found in DataFrame"
            raise KeyError(msg)

        if area_col is not None and area_col not in df.columns:
            msg = f"Area column '{area_col}' not found in DataFrame"
            raise KeyError(msg)

        # Ensure timestamp is datetime
        result = df.copy()
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])

        # Extract date from timestamp
        result["_filter_date"] = result[timestamp_col].dt.date

        # Detect frequency
        expected_records = self._detect_expected_records(result, timestamp_col)
        logger.debug(
            "CompleteDayFilter: detected frequency, expecting %d records/day",
            expected_records,
        )

        # Determine groupby columns
        group_cols = ["_filter_date"]
        if area_col is not None:
            group_cols = [area_col, "_filter_date"]

        # Calculate statistics per group
        group_stats = self._calculate_group_stats(result, group_cols, value_cols)

        # Identify complete groups
        complete_mask = (group_stats["record_count"] == expected_records) & (
            group_stats["missing_pct"] <= self._config.max_missing_pct
        )

        # Build report
        report.total_days = len(group_stats)
        report.complete_days = complete_mask.sum()
        report.incomplete_days = report.total_days - report.complete_days

        # Track reasons
        insufficient_records = (group_stats["record_count"] < expected_records).sum()
        excess_records = (group_stats["record_count"] > expected_records).sum()
        too_many_missing = (
            (group_stats["missing_pct"] > self._config.max_missing_pct)
            & (group_stats["record_count"] == expected_records)
        ).sum()

        if insufficient_records > 0:
            report.reasons["insufficient_records"] = int(insufficient_records)
        if excess_records > 0:
            report.reasons["excess_records"] = int(excess_records)
        if too_many_missing > 0:
            report.reasons["too_many_missing"] = int(too_many_missing)

        # Track affected areas
        if area_col is not None:
            incomplete_groups = group_stats[~complete_mask]
            if len(incomplete_groups) > 0:
                report.affected_areas = set(incomplete_groups.index.get_level_values(area_col))

        # Filter to complete groups
        complete_groups = group_stats[complete_mask].index
        if area_col is not None:
            # Create tuple keys for filtering
            result_keys = list(zip(result[area_col], result["_filter_date"], strict=True))
            mask = pd.Series(result_keys).isin(complete_groups)
            filtered_df = result[mask.to_numpy()].copy()
        else:
            filtered_df = result[
                result["_filter_date"].isin(complete_groups.get_level_values(0))
            ].copy()

        # Remove helper column
        filtered_df = filtered_df.drop(columns=["_filter_date"])

        logger.info(
            "CompleteDayFilter: retained %d/%d days (%.1f%%)",
            report.complete_days,
            report.total_days,
            report.retention_rate * 100,
        )

        return filtered_df, report

    def _detect_expected_records(self, df: pd.DataFrame, timestamp_col: str) -> int:
        """Auto-detect expected records per day from timestamp intervals.

        Uses the mode of timestamp differences to determine frequency.

        Args:
            df: DataFrame with timestamp column.
            timestamp_col: Name of the timestamp column.

        Returns:
            Expected number of records per day (48 for semi-hourly, 24 for hourly).
        """
        if len(df) < _MIN_RECORDS_FOR_DETECTION:
            # Default to semi-hourly if not enough data
            return self._config.semi_hourly_records

        # Sort by timestamp and compute differences
        sorted_ts = df[timestamp_col].sort_values()
        diffs = sorted_ts.diff().dropna()

        if len(diffs) == 0:
            return self._config.semi_hourly_records

        # Get mode of differences (most common interval)
        # Convert to minutes for easier comparison
        diff_minutes = diffs.dt.total_seconds() / 60
        mode_minutes = diff_minutes.mode()

        if len(mode_minutes) == 0:
            return self._config.semi_hourly_records

        mode_val = mode_minutes.iloc[0]

        # Determine frequency based on mode
        if _SEMI_HOURLY_MIN_MINUTES <= mode_val <= _SEMI_HOURLY_MAX_MINUTES:
            # ~30 minutes -> semi-hourly
            return self._config.semi_hourly_records
        if _HOURLY_MIN_MINUTES <= mode_val <= _HOURLY_MAX_MINUTES:
            # ~60 minutes -> hourly
            return self._config.hourly_records
        # Default to semi-hourly for other intervals
        logger.warning(
            "CompleteDayFilter: unusual interval of %.0f minutes detected, "
            "defaulting to semi-hourly (%d records/day)",
            mode_val,
            self._config.semi_hourly_records,
        )
        return self._config.semi_hourly_records

    def _calculate_group_stats(
        self,
        df: pd.DataFrame,
        group_cols: list[str],
        value_cols: list[str] | None,
    ) -> pd.DataFrame:
        """Calculate statistics per group for filtering decisions.

        Args:
            df: DataFrame to analyze.
            group_cols: Columns to group by.
            value_cols: Columns to check for missing values.

        Returns:
            DataFrame indexed by group with record_count and missing_pct columns.
        """
        # Count records per group
        record_counts = df.groupby(group_cols).size().rename("record_count")

        # Calculate missing percentage per group
        if value_cols is not None and len(value_cols) > 0:
            # Check only existing value columns
            existing_value_cols = [c for c in value_cols if c in df.columns]

            if existing_value_cols:
                # Calculate missing percentage across value columns
                def calc_missing_pct(group: pd.DataFrame) -> float:
                    total_values = len(group) * len(existing_value_cols)
                    if total_values == 0:
                        return 0.0
                    missing = group[existing_value_cols].isna().sum().sum()
                    return (missing / total_values) * 100

                missing_pcts = (
                    df.groupby(group_cols)
                    .apply(calc_missing_pct, include_groups=False)
                    .rename("missing_pct")
                )
            else:
                missing_pcts = pd.Series(0.0, index=record_counts.index, name="missing_pct")
        else:
            missing_pcts = pd.Series(0.0, index=record_counts.index, name="missing_pct")

        # Combine into single DataFrame
        return pd.concat([record_counts, missing_pcts], axis=1)


class DuplicateHandler:
    """Utility class for detecting and removing duplicate records.

    Provides static methods for identifying and removing duplicate rows
    based on specified key columns, keeping the first occurrence.

    Example:
        ```python
        import pandas as pd
        from src.data.filters import DuplicateHandler

        df = pd.DataFrame({
            "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "cod_area": ["SP", "SP", "SP"],
            "value": [100, 100, 200],
        })

        # Remove duplicates, keeping first occurrence
        deduplicated_df, num_removed = DuplicateHandler.detect_and_remove(
            df, key_columns=["timestamp", "cod_area"]
        )
        print(f"Removed {num_removed} duplicate rows")
        ```
    """

    @staticmethod
    def detect_and_remove(
        df: pd.DataFrame,
        key_columns: list[str],
    ) -> tuple[pd.DataFrame, int]:
        """Detect and remove duplicate rows based on key columns.

        Identifies duplicates based on the specified key columns and removes
        them, keeping the first occurrence of each unique combination.

        Args:
            df: Input DataFrame to deduplicate.
            key_columns: List of column names that form the unique key.

        Returns:
            Tuple of (deduplicated_df, num_removed) where deduplicated_df
            contains only unique rows and num_removed is the count of
            removed duplicates.

        Raises:
            ValueError: If key_columns is empty.

        Example:
            ```python
            df, removed = DuplicateHandler.detect_and_remove(
                df, key_columns=["timestamp", "cod_area"]
            )
            if removed > 0:
                logger.warning("Removed %d duplicate rows", removed)
            ```
        """
        if not key_columns:
            msg = "key_columns cannot be empty"
            raise ValueError(msg)

        if df.empty:
            return df.copy(), 0

        # Check which columns exist
        existing_cols = [col for col in key_columns if col in df.columns]

        if not existing_cols:
            logger.warning(
                "DuplicateHandler: none of the key columns %s exist in DataFrame",
                key_columns,
            )
            return df.copy(), 0

        if len(existing_cols) < len(key_columns):
            missing_cols = set(key_columns) - set(existing_cols)
            logger.warning(
                "DuplicateHandler: columns %s not found, using only %s",
                missing_cols,
                existing_cols,
            )

        # Count initial rows
        initial_count = len(df)

        # Remove duplicates, keeping first occurrence
        deduplicated_df = df.drop_duplicates(subset=existing_cols, keep="first")

        # Calculate removed count
        num_removed = initial_count - len(deduplicated_df)

        if num_removed > 0:
            logger.info(
                "DuplicateHandler: removed %d duplicate rows based on columns %s",
                num_removed,
                existing_cols,
            )

        return deduplicated_df.copy(), num_removed


class ChronologicalValidator:
    """Utility class for validating and sorting chronological order.

    Provides static methods for checking if time series data is properly
    sorted by timestamp within each group and optionally sorting it.

    Example:
        ```python
        import pandas as pd
        from src.data.filters import ChronologicalValidator

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-01"]),
            "cod_area": ["SP", "SP"],
        })

        # Auto-sort if not in order
        sorted_df = ChronologicalValidator.validate_and_sort(
            df,
            timestamp_col="timestamp",
            group_col="cod_area",
            auto_sort=True,
        )

        # Raise error if not in order
        try:
            df = ChronologicalValidator.validate_and_sort(
                df,
                timestamp_col="timestamp",
                auto_sort=False,
            )
        except ValueError as e:
            print(f"Data not sorted: {e}")
        ```
    """

    @staticmethod
    def validate_and_sort(
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        group_col: str | None = "cod_area",
        auto_sort: bool = True,
    ) -> pd.DataFrame:
        """Validate chronological order and optionally sort.

        Checks if timestamps are in ascending order within each group.
        If auto_sort is True, sorts the data. If False and data is not
        sorted, raises ValueError.

        Args:
            df: Input DataFrame with timestamp data.
            timestamp_col: Name of the timestamp column.
            group_col: Column to group by before checking order. If None,
                checks the entire DataFrame as one series.
            auto_sort: If True, automatically sort data if not in order.
                If False, raise ValueError when data is not sorted.

        Returns:
            DataFrame with chronologically sorted data (if auto_sort=True)
            or the original DataFrame (if already sorted and auto_sort=False).

        Raises:
            KeyError: If timestamp_col or group_col is not found in DataFrame.
            ValueError: If auto_sort=False and data is not chronologically sorted.

        Example:
            ```python
            sorted_df = ChronologicalValidator.validate_and_sort(
                df,
                timestamp_col="timestamp",
                group_col="cod_area",
                auto_sort=True,
            )
            ```
        """
        if df.empty:
            return df.copy()

        if timestamp_col not in df.columns:
            msg = f"Timestamp column '{timestamp_col}' not found in DataFrame"
            raise KeyError(msg)

        if group_col is not None and group_col not in df.columns:
            msg = f"Group column '{group_col}' not found in DataFrame"
            raise KeyError(msg)

        result = df.copy()
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])

        # Check if sorted within each group
        is_sorted = ChronologicalValidator._check_sorted(result, timestamp_col, group_col)

        if is_sorted:
            logger.debug("ChronologicalValidator: data is already sorted")
            return result

        if not auto_sort:
            msg = f"Data is not chronologically sorted by '{timestamp_col}'" + (
                f" within '{group_col}' groups" if group_col else ""
            )
            raise ValueError(msg)

        # Sort the data
        sort_cols = [group_col, timestamp_col] if group_col else [timestamp_col]
        sort_cols = [c for c in sort_cols if c is not None]
        result = result.sort_values(sort_cols).reset_index(drop=True)

        logger.info(
            "ChronologicalValidator: sorted data by %s",
            sort_cols,
        )

        return result

    @staticmethod
    def _check_sorted(
        df: pd.DataFrame,
        timestamp_col: str,
        group_col: str | None,
    ) -> bool:
        """Check if DataFrame is sorted by timestamp within groups.

        Args:
            df: DataFrame to check.
            timestamp_col: Name of the timestamp column.
            group_col: Column to group by, or None for no grouping.

        Returns:
            True if data is sorted, False otherwise.
        """
        if len(df) < _MIN_RECORDS_FOR_DETECTION:
            return True

        if group_col is not None:
            # Check each group separately
            for _group_name, group_df in df.groupby(group_col):
                timestamps = group_df[timestamp_col]
                if not timestamps.is_monotonic_increasing:
                    return False
            return True
        # Check entire DataFrame
        return df[timestamp_col].is_monotonic_increasing
