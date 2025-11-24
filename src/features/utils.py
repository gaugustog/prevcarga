"""Utility functions for feature engineering.

This module provides utility functions for working with feature DataFrames,
including index validation, feature name checking, and DataFrame merging.

Example:
    ```python
    from src.features.utils import (
        validate_dataframe_index,
        check_feature_names,
        merge_feature_dataframes
    )

    # Validate datetime index
    validate_dataframe_index(df)

    # Check for feature name collisions
    collisions = check_feature_names(["hour", "day"], ["day", "month"])

    # Merge feature DataFrames
    merged_df = merge_feature_dataframes([df1, df2, df3])
    ```
"""

from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureUtilsError(Exception):
    """Base exception for feature utility errors."""


class InvalidIndexError(FeatureUtilsError):
    """Exception raised when DataFrame index is invalid."""


class FeatureNameCollisionError(FeatureUtilsError):
    """Exception raised when feature names collide."""

    def __init__(self, collisions: list[str]) -> None:
        """Initialize FeatureNameCollisionError.

        Args:
            collisions: List of colliding feature names.
        """
        self.collisions = collisions
        msg = f"Feature name collisions detected: {collisions}"
        super().__init__(msg)


def validate_dataframe_index(
    df: pd.DataFrame,
    expected_name: str | None = None,
    require_sorted: bool = True,
    require_unique: bool = True,
) -> bool:
    """Validate that a DataFrame has a proper datetime index.

    This function checks that the DataFrame index is a DatetimeIndex
    and optionally validates sorting, uniqueness, and name.

    Args:
        df: DataFrame to validate.
        expected_name: Expected index name (e.g., "timestamp"). If None, any name is accepted.
        require_sorted: Whether to require the index to be sorted.
        require_unique: Whether to require unique index values.

    Returns:
        True if the index is valid.

    Raises:
        InvalidIndexError: If the index validation fails.

    Example:
        ```python
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="h")
        )
        df.index.name = "timestamp"

        validate_dataframe_index(df, expected_name="timestamp")
        ```
    """
    # Check if index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        msg = f"DataFrame index must be DatetimeIndex, got {type(df.index).__name__}"
        raise InvalidIndexError(msg)

    # Check index name if specified
    if expected_name is not None and df.index.name != expected_name:
        msg = f"Index name must be '{expected_name}', got '{df.index.name}'"
        raise InvalidIndexError(msg)

    # Check if sorted
    if require_sorted and not df.index.is_monotonic_increasing:
        msg = "DataFrame index must be sorted in ascending order"
        raise InvalidIndexError(msg)

    # Check if unique
    if require_unique and not df.index.is_unique:
        duplicate_count = df.index.duplicated().sum()
        msg = f"DataFrame index must have unique values, found {duplicate_count} duplicates"
        raise InvalidIndexError(msg)

    logger.debug(
        "DataFrame index validated: %d rows, %s to %s",
        len(df),
        df.index.min() if len(df) > 0 else "N/A",
        df.index.max() if len(df) > 0 else "N/A",
    )

    return True


def check_feature_names(
    existing_names: list[str],
    new_names: list[str],
    raise_on_collision: bool = False,
) -> list[str]:
    """Check for collisions between existing and new feature names.

    Args:
        existing_names: List of existing feature column names.
        new_names: List of new feature column names to add.
        raise_on_collision: If True, raise an exception on collision.

    Returns:
        List of colliding feature names (empty if no collisions).

    Raises:
        FeatureNameCollisionError: If raise_on_collision is True and
            collisions are detected.

    Example:
        ```python
        # Check for collisions
        collisions = check_feature_names(
            existing_names=["hour", "day_of_week"],
            new_names=["day_of_week", "month"]
        )
        # Returns: ["day_of_week"]
        ```
    """
    collisions = list(set(existing_names) & set(new_names))

    if collisions:
        logger.warning("Feature name collisions detected: %s", collisions)
        if raise_on_collision:
            raise FeatureNameCollisionError(collisions)

    return collisions


def _validate_merge_indexes(dataframes: list[pd.DataFrame]) -> None:
    """Validate indexes for all DataFrames in the merge list.

    Args:
        dataframes: List of DataFrames to validate.

    Raises:
        InvalidIndexError: If any DataFrame has an invalid index.
    """
    for i, df in enumerate(dataframes):
        try:
            validate_dataframe_index(df, require_sorted=False, require_unique=False)
        except InvalidIndexError as e:
            msg = f"DataFrame at index {i} has invalid index: {e}"
            raise InvalidIndexError(msg) from e


def _find_duplicate_columns(dataframes: list[pd.DataFrame]) -> list[str]:
    """Find duplicate column names across multiple DataFrames.

    Args:
        dataframes: List of DataFrames to check.

    Returns:
        List of duplicate column names.
    """
    all_columns: list[str] = []
    for df in dataframes:
        all_columns.extend(df.columns.tolist())

    seen: set[str] = set()
    duplicates: list[str] = []
    for col in all_columns:
        if col in seen:
            duplicates.append(col)
        seen.add(col)
    return duplicates


def _get_columns_to_add(
    df: pd.DataFrame,
    existing_columns: set[str],
    handle_duplicates: str,
) -> tuple[list[str], set[str]]:
    """Determine which columns to add from a DataFrame during merge.

    Args:
        df: DataFrame to get columns from.
        existing_columns: Set of columns already in the result.
        handle_duplicates: Strategy for handling duplicates.

    Returns:
        Tuple of (columns to add, columns to drop from existing).
    """
    columns_to_add: list[str] = []
    columns_to_drop: set[str] = set()

    for col in df.columns:
        if col not in existing_columns:
            columns_to_add.append(col)
        elif handle_duplicates == "keep_last":
            columns_to_drop.add(col)
            columns_to_add.append(col)
        # For "keep_first", we skip the column (don't add it)

    return columns_to_add, columns_to_drop


def merge_feature_dataframes(
    dataframes: list[pd.DataFrame],
    how: str = "outer",
    validate_index: bool = True,
    handle_duplicates: str = "raise",
) -> pd.DataFrame:
    """Merge multiple feature DataFrames into a single DataFrame.

    This function combines multiple DataFrames with feature columns,
    handling index alignment and duplicate column detection.

    Args:
        dataframes: List of DataFrames to merge.
        how: Join type ('inner', 'outer', 'left', 'right').
        validate_index: Whether to validate datetime index on each DataFrame.
        handle_duplicates: How to handle duplicate columns:
            - 'raise': Raise an exception
            - 'keep_first': Keep the first occurrence
            - 'keep_last': Keep the last occurrence

    Returns:
        Merged DataFrame with all feature columns.

    Raises:
        ValueError: If dataframes list is empty.
        InvalidIndexError: If validate_index is True and an index is invalid.
        FeatureNameCollisionError: If handle_duplicates is 'raise' and
            duplicate columns are found.

    Example:
        ```python
        df1 = pd.DataFrame(
            {"feature_a": [1, 2]},
            index=pd.date_range("2024-01-01", periods=2, freq="h")
        )
        df2 = pd.DataFrame(
            {"feature_b": [3, 4]},
            index=pd.date_range("2024-01-01", periods=2, freq="h")
        )

        merged = merge_feature_dataframes([df1, df2])
        # Returns DataFrame with columns: feature_a, feature_b
        ```
    """
    if not dataframes:
        msg = "At least one DataFrame is required for merging"
        raise ValueError(msg)

    if len(dataframes) == 1:
        return dataframes[0].copy()

    if validate_index:
        _validate_merge_indexes(dataframes)

    duplicates = _find_duplicate_columns(dataframes)
    if duplicates:
        if handle_duplicates == "raise":
            raise FeatureNameCollisionError(duplicates)
        logger.warning(
            "Duplicate columns found during merge: %s. Using '%s' strategy.",
            duplicates,
            handle_duplicates,
        )

    # Perform the merge
    result = dataframes[0].copy()

    for df in dataframes[1:]:
        columns_to_add, columns_to_drop = _get_columns_to_add(
            df, set(result.columns), handle_duplicates
        )

        if columns_to_drop:
            result = result.drop(columns=list(columns_to_drop))

        if columns_to_add:
            result = result.join(df[columns_to_add], how=how)

    logger.debug(
        "Merged %d DataFrames into result with %d columns",
        len(dataframes),
        len(result.columns),
    )

    return result


def get_feature_statistics(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    """Calculate basic statistics for feature columns.

    Args:
        df: DataFrame with feature columns.
        columns: Specific columns to analyze. If None, all numeric columns are used.

    Returns:
        Dictionary with statistics for each column including:
            - count: Number of non-null values
            - mean: Mean value
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value
            - null_count: Number of null values
            - null_pct: Percentage of null values

    Example:
        ```python
        stats = get_feature_statistics(df, columns=["hour", "day_of_week"])
        print(stats["hour"]["mean"])  # Average hour value
        ```
    """
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()

    stats: dict[str, Any] = {}

    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found in DataFrame", col)
            continue

        col_data = df[col]
        null_count = col_data.isna().sum()

        stats[col] = {
            "count": int(col_data.count()),
            "null_count": int(null_count),
            "null_pct": float(null_count / len(df) * 100) if len(df) > 0 else 0.0,
        }

        if pd.api.types.is_numeric_dtype(col_data):
            stats[col].update(
                {
                    "mean": float(col_data.mean()) if col_data.count() > 0 else None,
                    "std": float(col_data.std()) if col_data.count() > 1 else None,
                    "min": float(col_data.min()) if col_data.count() > 0 else None,
                    "max": float(col_data.max()) if col_data.count() > 0 else None,
                }
            )

    return stats
