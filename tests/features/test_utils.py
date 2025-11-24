"""Tests for feature utility functions.

This module contains tests for the utility functions in src/features/utils.py,
including DataFrame index validation, feature name checking, and DataFrame merging.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.utils import (
    FeatureNameCollisionError,
    InvalidIndexError,
    check_feature_names,
    get_feature_statistics,
    merge_feature_dataframes,
    validate_dataframe_index,
)


class TestValidateDataFrameIndex:
    """Tests for validate_dataframe_index function."""

    def test_valid_datetime_index(self):
        """Test validation passes for valid datetime index."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )

        result = validate_dataframe_index(df)

        assert result is True

    def test_invalid_integer_index(self):
        """Test validation fails for non-datetime index."""
        df = pd.DataFrame({"value": [1, 2, 3]}, index=[0, 1, 2])

        with pytest.raises(InvalidIndexError) as exc_info:
            validate_dataframe_index(df)

        assert "DatetimeIndex" in str(exc_info.value)

    def test_invalid_string_index(self):
        """Test validation fails for string index."""
        df = pd.DataFrame({"value": [1, 2, 3]}, index=["a", "b", "c"])

        with pytest.raises(InvalidIndexError):
            validate_dataframe_index(df)

    def test_expected_name_valid(self):
        """Test validation passes when index name matches expected."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )
        df.index.name = "timestamp"

        result = validate_dataframe_index(df, expected_name="timestamp")

        assert result is True

    def test_expected_name_invalid(self):
        """Test validation fails when index name doesn't match."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )
        df.index.name = "datetime"

        with pytest.raises(InvalidIndexError) as exc_info:
            validate_dataframe_index(df, expected_name="timestamp")

        assert "timestamp" in str(exc_info.value)

    def test_unsorted_index_with_require_sorted(self):
        """Test validation fails for unsorted index when sorting required."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
        )

        with pytest.raises(InvalidIndexError) as exc_info:
            validate_dataframe_index(df, require_sorted=True)

        assert "sorted" in str(exc_info.value).lower()

    def test_unsorted_index_without_require_sorted(self):
        """Test validation passes for unsorted index when not required."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
        )

        result = validate_dataframe_index(df, require_sorted=False)

        assert result is True

    def test_duplicate_index_with_require_unique(self):
        """Test validation fails for duplicate index when uniqueness required."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        )

        with pytest.raises(InvalidIndexError) as exc_info:
            validate_dataframe_index(df, require_unique=True)

        assert "unique" in str(exc_info.value).lower()

    def test_duplicate_index_without_require_unique(self):
        """Test validation passes for duplicate index when not required."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        )

        result = validate_dataframe_index(df, require_unique=False)

        assert result is True

    def test_empty_dataframe(self):
        """Test validation passes for empty DataFrame with datetime index."""
        df = pd.DataFrame(
            columns=["value"],
            index=pd.DatetimeIndex([], name="timestamp"),
        )

        result = validate_dataframe_index(df)

        assert result is True


class TestCheckFeatureNames:
    """Tests for check_feature_names function."""

    def test_no_collision(self):
        """Test returns empty list when no collisions."""
        collisions = check_feature_names(
            existing_names=["col_a", "col_b"],
            new_names=["col_c", "col_d"],
        )

        assert collisions == []

    def test_with_collision(self):
        """Test returns colliding names."""
        collisions = check_feature_names(
            existing_names=["col_a", "col_b"],
            new_names=["col_b", "col_c"],
        )

        assert "col_b" in collisions

    def test_multiple_collisions(self):
        """Test returns all colliding names."""
        collisions = check_feature_names(
            existing_names=["col_a", "col_b", "col_c"],
            new_names=["col_a", "col_c", "col_d"],
        )

        assert len(collisions) == 2
        assert "col_a" in collisions
        assert "col_c" in collisions

    def test_raise_on_collision_true(self):
        """Test raises exception when raise_on_collision is True."""
        with pytest.raises(FeatureNameCollisionError) as exc_info:
            check_feature_names(
                existing_names=["col_a"],
                new_names=["col_a"],
                raise_on_collision=True,
            )

        assert "col_a" in exc_info.value.collisions

    def test_raise_on_collision_false(self):
        """Test doesn't raise when raise_on_collision is False."""
        # Should not raise
        collisions = check_feature_names(
            existing_names=["col_a"],
            new_names=["col_a"],
            raise_on_collision=False,
        )

        assert "col_a" in collisions

    def test_empty_lists(self):
        """Test with empty lists."""
        collisions = check_feature_names(
            existing_names=[],
            new_names=[],
        )

        assert collisions == []


class TestMergeFeatureDataFrames:
    """Tests for merge_feature_dataframes function."""

    @pytest.fixture
    def df1(self) -> pd.DataFrame:
        """Create first sample DataFrame."""
        return pd.DataFrame(
            {"feature_a": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )

    @pytest.fixture
    def df2(self) -> pd.DataFrame:
        """Create second sample DataFrame."""
        return pd.DataFrame(
            {"feature_b": [4.0, 5.0, 6.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )

    def test_merge_two_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """Test merging two DataFrames."""
        result = merge_feature_dataframes([df1, df2])

        assert "feature_a" in result.columns
        assert "feature_b" in result.columns
        assert len(result) == 3

    def test_merge_single_dataframe(self, df1: pd.DataFrame):
        """Test merging single DataFrame returns copy."""
        result = merge_feature_dataframes([df1])

        assert "feature_a" in result.columns
        assert result is not df1  # Should be a copy

    def test_merge_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            merge_feature_dataframes([])

        assert "At least one DataFrame" in str(exc_info.value)

    def test_merge_outer_join(self):
        """Test outer join behavior."""
        df1 = pd.DataFrame(
            {"feature_a": [1.0, 2.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="h"),
        )
        df2 = pd.DataFrame(
            {"feature_b": [3.0, 4.0]},
            index=pd.date_range("2024-01-01 01:00:00", periods=2, freq="h"),
        )

        result = merge_feature_dataframes([df1, df2], how="outer")

        assert len(result) == 3  # Union of indices

    def test_merge_inner_join(self):
        """Test inner join behavior."""
        df1 = pd.DataFrame(
            {"feature_a": [1.0, 2.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="h"),
        )
        df2 = pd.DataFrame(
            {"feature_b": [3.0, 4.0]},
            index=pd.date_range("2024-01-01 01:00:00", periods=2, freq="h"),
        )

        result = merge_feature_dataframes([df1, df2], how="inner")

        assert len(result) == 1  # Intersection of indices

    def test_merge_duplicate_columns_raise(self):
        """Test that duplicate columns raise by default."""
        df1 = pd.DataFrame(
            {"feature_a": [1.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="h"),
        )
        df2 = pd.DataFrame(
            {"feature_a": [2.0]},  # Duplicate column name
            index=pd.date_range("2024-01-01", periods=1, freq="h"),
        )

        with pytest.raises(FeatureNameCollisionError):
            merge_feature_dataframes([df1, df2], handle_duplicates="raise")

    def test_merge_duplicate_columns_keep_first(self):
        """Test keeping first occurrence of duplicate columns."""
        df1 = pd.DataFrame(
            {"feature_a": [1.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="h"),
        )
        df2 = pd.DataFrame(
            {"feature_a": [2.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="h"),
        )

        result = merge_feature_dataframes([df1, df2], handle_duplicates="keep_first")

        assert result["feature_a"].iloc[0] == 1.0

    def test_merge_duplicate_columns_keep_last(self):
        """Test keeping last occurrence of duplicate columns."""
        df1 = pd.DataFrame(
            {"feature_a": [1.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="h"),
        )
        df2 = pd.DataFrame(
            {"feature_a": [2.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="h"),
        )

        result = merge_feature_dataframes([df1, df2], handle_duplicates="keep_last")

        assert result["feature_a"].iloc[0] == 2.0

    def test_merge_validate_index_true(self):
        """Test that invalid index raises when validation enabled."""
        # Need at least 2 DataFrames to trigger validation (single DF returns early)
        df1 = pd.DataFrame({"feature_a": [1.0]}, index=[0])  # Integer index
        df2 = pd.DataFrame({"feature_b": [2.0]}, index=[0])  # Integer index

        with pytest.raises(InvalidIndexError):
            merge_feature_dataframes([df1, df2], validate_index=True)

    def test_merge_validate_index_false(self):
        """Test that invalid index is allowed when validation disabled."""
        df1 = pd.DataFrame({"feature_a": [1.0]}, index=[0])

        # Should not raise
        result = merge_feature_dataframes([df1], validate_index=False)

        assert "feature_a" in result.columns

    def test_merge_three_dataframes(self):
        """Test merging three DataFrames."""
        idx = pd.date_range("2024-01-01", periods=2, freq="h")
        df1 = pd.DataFrame({"a": [1, 2]}, index=idx)
        df2 = pd.DataFrame({"b": [3, 4]}, index=idx)
        df3 = pd.DataFrame({"c": [5, 6]}, index=idx)

        result = merge_feature_dataframes([df1, df2, df3])

        assert list(result.columns) == ["a", "b", "c"]


class TestGetFeatureStatistics:
    """Tests for get_feature_statistics function."""

    def test_basic_statistics(self):
        """Test basic statistics calculation."""
        df = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        stats = get_feature_statistics(df)

        assert "feature_a" in stats
        assert "feature_b" in stats
        assert stats["feature_a"]["mean"] == 3.0
        assert stats["feature_a"]["min"] == 1.0
        assert stats["feature_a"]["max"] == 5.0

    def test_statistics_with_nan(self):
        """Test statistics with NaN values."""
        df = pd.DataFrame(
            {
                "feature_a": [1.0, np.nan, 3.0],
            }
        )

        stats = get_feature_statistics(df)

        assert stats["feature_a"]["null_count"] == 1
        assert stats["feature_a"]["count"] == 2

    def test_statistics_specific_columns(self):
        """Test statistics for specific columns only."""
        df = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0],
                "feature_b": [4.0, 5.0, 6.0],
                "feature_c": [7.0, 8.0, 9.0],
            }
        )

        stats = get_feature_statistics(df, columns=["feature_a", "feature_b"])

        assert "feature_a" in stats
        assert "feature_b" in stats
        assert "feature_c" not in stats

    def test_statistics_missing_column(self, caplog):
        """Test statistics warns for missing column."""
        df = pd.DataFrame({"feature_a": [1.0, 2.0]})

        with caplog.at_level("WARNING"):
            stats = get_feature_statistics(df, columns=["feature_a", "nonexistent"])

        assert "feature_a" in stats
        assert "nonexistent" not in stats

    def test_statistics_empty_dataframe(self):
        """Test statistics for empty DataFrame."""
        df = pd.DataFrame({"feature_a": pd.Series([], dtype=float)})

        stats = get_feature_statistics(df)

        assert stats["feature_a"]["count"] == 0
        assert stats["feature_a"]["null_pct"] == 0.0

    def test_statistics_non_numeric_columns(self):
        """Test statistics filters non-numeric by default."""
        df = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0],
                "string": ["a", "b", "c"],
            }
        )

        stats = get_feature_statistics(df)

        # Only numeric column should be in stats
        assert "numeric" in stats
        assert "string" not in stats
