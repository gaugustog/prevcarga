"""Data preprocessing and imputation utilities for PrevCarga.

This module provides classes for handling missing data in time series through
various imputation strategies, as well as utilities for resampling and timezone
handling specific to the Brazilian electric system.

Example:
    ```python
    import pandas as pd
    from src.data.preprocessors import TriplePassImputer, ImputerChain, TimezoneHandler

    # Load data with missing values
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=48, freq="30min"),
        "load": [100.0 if i % 5 != 0 else None for i in range(48)],
    })

    # Apply triple pass imputation (matches R implementation)
    imputer = TriplePassImputer(date_col="timestamp")
    df = imputer.fit_transform(df, value_col="load")

    # Or chain multiple imputers
    chain = ImputerChain([
        ForwardFillImputer(limit=2),
        BackwardFillImputer(limit=2),
        InterpolationImputer(method="linear"),
    ])
    df = chain.fit_transform(df, value_col="load")

    # Handle timezones
    handler = TimezoneHandler()
    df = handler.localize_to_brazil(df, timestamp_col="timestamp")
    ```
"""

from abc import ABC, abstractmethod
from zoneinfo import ZoneInfo

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


class BaseImputer(ABC):
    """Abstract base class for all imputation strategies.

    Imputers fill missing values in pandas DataFrames using various strategies.
    All concrete imputers must implement the `fit_transform` method and provide
    a unique `name` property.

    Example:
        ```python
        class MyImputer(BaseImputer):
            @property
            def name(self) -> str:
                return "my_imputer"

            def fit_transform(
                self, df: pd.DataFrame, value_col: str
            ) -> pd.DataFrame:
                # Implement imputation logic
                return df.copy()
        ```
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this imputer.

        Returns:
            A string identifier for the imputer.
        """
        ...

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Fit the imputer to the data and transform it by filling missing values.

        Args:
            df: Input DataFrame containing the column to impute.
            value_col: Name of the column containing values to impute.

        Returns:
            DataFrame with missing values filled according to the imputation strategy.
        """
        ...


class ForwardFillImputer(BaseImputer):
    """Imputer that fills missing values using forward fill (ffill).

    Forward fill propagates the last valid observation forward to fill gaps.
    Supports limiting the number of consecutive NaN values to fill and
    groupby operations for filling within groups.

    Attributes:
        limit: Maximum number of consecutive NaN values to fill. None for unlimited.
        group_col: Column name to group by before filling. None for no grouping.

    Example:
        ```python
        imputer = ForwardFillImputer(limit=3, group_col="area")
        df = imputer.fit_transform(df, value_col="load")
        ```
    """

    def __init__(
        self,
        limit: int | None = None,
        group_col: str | None = None,
    ) -> None:
        """Initialize the forward fill imputer.

        Args:
            limit: Maximum number of consecutive NaN values to fill.
                If None, fills all consecutive NaN values.
            group_col: Column name to group by before applying forward fill.
                If None, applies fill to entire column.
        """
        self._limit = limit
        self._group_col = group_col

    @property
    def name(self) -> str:
        """Return the imputer name."""
        return "forward_fill"

    @property
    def limit(self) -> int | None:
        """Return the fill limit."""
        return self._limit

    @property
    def group_col(self) -> str | None:
        """Return the groupby column name."""
        return self._group_col

    def fit_transform(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Fill missing values using forward fill.

        Args:
            df: Input DataFrame.
            value_col: Name of the column to impute.

        Returns:
            DataFrame with forward-filled values.

        Raises:
            KeyError: If value_col or group_col does not exist in DataFrame.
        """
        result = df.copy()

        if value_col not in result.columns:
            msg = f"Column '{value_col}' not found in DataFrame"
            raise KeyError(msg)

        if self._group_col is not None:
            if self._group_col not in result.columns:
                msg = f"Group column '{self._group_col}' not found in DataFrame"
                raise KeyError(msg)

            result[value_col] = result.groupby(self._group_col)[value_col].ffill(limit=self._limit)
        else:
            result[value_col] = result[value_col].ffill(limit=self._limit)

        na_before = df[value_col].isna().sum()
        na_after = result[value_col].isna().sum()
        filled = na_before - na_after

        logger.debug(
            "ForwardFillImputer: filled %d/%d missing values in '%s'",
            filled,
            na_before,
            value_col,
        )

        return result


class BackwardFillImputer(BaseImputer):
    """Imputer that fills missing values using backward fill (bfill).

    Backward fill propagates the next valid observation backward to fill gaps.
    Supports limiting the number of consecutive NaN values to fill and
    groupby operations for filling within groups.

    Attributes:
        limit: Maximum number of consecutive NaN values to fill. None for unlimited.
        group_col: Column name to group by before filling. None for no grouping.

    Example:
        ```python
        imputer = BackwardFillImputer(limit=3, group_col="area")
        df = imputer.fit_transform(df, value_col="load")
        ```
    """

    def __init__(
        self,
        limit: int | None = None,
        group_col: str | None = None,
    ) -> None:
        """Initialize the backward fill imputer.

        Args:
            limit: Maximum number of consecutive NaN values to fill.
                If None, fills all consecutive NaN values.
            group_col: Column name to group by before applying backward fill.
                If None, applies fill to entire column.
        """
        self._limit = limit
        self._group_col = group_col

    @property
    def name(self) -> str:
        """Return the imputer name."""
        return "backward_fill"

    @property
    def limit(self) -> int | None:
        """Return the fill limit."""
        return self._limit

    @property
    def group_col(self) -> str | None:
        """Return the groupby column name."""
        return self._group_col

    def fit_transform(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Fill missing values using backward fill.

        Args:
            df: Input DataFrame.
            value_col: Name of the column to impute.

        Returns:
            DataFrame with backward-filled values.

        Raises:
            KeyError: If value_col or group_col does not exist in DataFrame.
        """
        result = df.copy()

        if value_col not in result.columns:
            msg = f"Column '{value_col}' not found in DataFrame"
            raise KeyError(msg)

        if self._group_col is not None:
            if self._group_col not in result.columns:
                msg = f"Group column '{self._group_col}' not found in DataFrame"
                raise KeyError(msg)

            result[value_col] = result.groupby(self._group_col)[value_col].bfill(limit=self._limit)
        else:
            result[value_col] = result[value_col].bfill(limit=self._limit)

        na_before = df[value_col].isna().sum()
        na_after = result[value_col].isna().sum()
        filled = na_before - na_after

        logger.debug(
            "BackwardFillImputer: filled %d/%d missing values in '%s'",
            filled,
            na_before,
            value_col,
        )

        return result


class InterpolationImputer(BaseImputer):
    """Imputer that fills missing values using interpolation.

    Uses pandas interpolate() to fill missing values using various
    interpolation methods such as linear, polynomial, spline, etc.

    Attributes:
        method: Interpolation method to use (e.g., "linear", "polynomial", "spline").

    Example:
        ```python
        imputer = InterpolationImputer(method="linear")
        df = imputer.fit_transform(df, value_col="load")
        ```
    """

    VALID_METHODS = frozenset(
        {
            "linear",
            "time",
            "index",
            "pad",
            "nearest",
            "zero",
            "slinear",
            "quadratic",
            "cubic",
            "polynomial",
            "spline",
            "barycentric",
            "krogh",
            "pchip",
            "akima",
            "cubicspline",
        }
    )

    def __init__(self, method: str = "linear") -> None:
        """Initialize the interpolation imputer.

        Args:
            method: Interpolation method to use. Supported methods include:
                - "linear": Linear interpolation (default)
                - "time": Time-weighted interpolation for datetime index
                - "nearest": Nearest neighbor interpolation
                - "polynomial": Polynomial interpolation (requires order)
                - "spline": Spline interpolation (requires order)
                And other methods supported by pandas.interpolate().

        Raises:
            ValueError: If method is not a valid interpolation method.
        """
        if method not in self.VALID_METHODS:
            msg = f"Invalid interpolation method '{method}'. Valid methods: {sorted(self.VALID_METHODS)}"
            raise ValueError(msg)
        self._method = method

    @property
    def name(self) -> str:
        """Return the imputer name."""
        return f"interpolation_{self._method}"

    @property
    def method(self) -> str:
        """Return the interpolation method."""
        return self._method

    def fit_transform(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Fill missing values using interpolation.

        Args:
            df: Input DataFrame.
            value_col: Name of the column to impute.

        Returns:
            DataFrame with interpolated values.

        Raises:
            KeyError: If value_col does not exist in DataFrame.
        """
        result = df.copy()

        if value_col not in result.columns:
            msg = f"Column '{value_col}' not found in DataFrame"
            raise KeyError(msg)

        result[value_col] = result[value_col].interpolate(method=self._method)

        na_before = df[value_col].isna().sum()
        na_after = result[value_col].isna().sum()
        filled = na_before - na_after

        logger.debug(
            "InterpolationImputer (%s): filled %d/%d missing values in '%s'",
            self._method,
            filled,
            na_before,
            value_col,
        )

        return result


class LagFillImputer(BaseImputer):
    """Imputer that fills missing values using lagged observations.

    For time series data, this imputer fills missing values by looking back
    at values from previous periods (e.g., same hour yesterday, same hour
    two days ago). Tries each lag in sequence until a valid value is found.

    Attributes:
        lag_hours: List of lag periods in hours to try in sequence.

    Example:
        ```python
        # Try 24h lag first, then 48h lag
        imputer = LagFillImputer(lag_hours=[24, 48])
        df = imputer.fit_transform(df, value_col="load")
        ```
    """

    def __init__(self, lag_hours: list[int] | None = None) -> None:
        """Initialize the lag fill imputer.

        Args:
            lag_hours: List of lag periods in hours to try in sequence.
                Defaults to [24, 48] (same hour yesterday, then two days ago).
        """
        self._lag_hours = lag_hours if lag_hours is not None else [24, 48]

    @property
    def name(self) -> str:
        """Return the imputer name."""
        return "lag_fill"

    @property
    def lag_hours(self) -> list[int]:
        """Return the list of lag hours."""
        return self._lag_hours.copy()

    def fit_transform(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Fill missing values using lagged observations.

        For each missing value, tries to fill with the value from each lag
        period in sequence until a non-null value is found.

        Args:
            df: Input DataFrame. Must have a datetime index or timestamp column.
            value_col: Name of the column to impute.

        Returns:
            DataFrame with lag-filled values.

        Raises:
            KeyError: If value_col does not exist in DataFrame.
        """
        result = df.copy()

        if value_col not in result.columns:
            msg = f"Column '{value_col}' not found in DataFrame"
            raise KeyError(msg)

        na_before = result[value_col].isna().sum()

        for lag_h in self._lag_hours:
            # Number of periods depends on data frequency
            # Assuming 30-min data: lag_h hours = lag_h * 2 periods
            # Assuming hourly data: lag_h hours = lag_h periods
            # We try to infer from index or use a reasonable default
            lag_periods = lag_h * 2  # Assuming 30-min data (most common)

            # Create lagged series
            lagged = result[value_col].shift(lag_periods)

            # Fill only where current value is NaN and lagged value exists
            mask = result[value_col].isna() & lagged.notna()
            result.loc[mask, value_col] = lagged[mask]

            filled_this_lag = mask.sum()
            if filled_this_lag > 0:
                logger.debug(
                    "LagFillImputer: filled %d values using %dh lag",
                    filled_this_lag,
                    lag_h,
                )

            # Stop if all values are filled
            if not result[value_col].isna().any():
                break

        na_after = result[value_col].isna().sum()
        total_filled = na_before - na_after

        logger.debug(
            "LagFillImputer: total filled %d/%d missing values in '%s'",
            total_filled,
            na_before,
            value_col,
        )

        return result


class TriplePassImputer(BaseImputer):
    """Imputer implementing the R triple-pass imputation algorithm.

    This imputer matches the R implementation used in the legacy PrevCarga system.
    It performs four passes to fill missing values:

    1. Pass 1: Forward fill within same day (groupby date)
    2. Pass 2: Backward fill within same day (groupby date)
    3. Pass 3: Average interpolation using prev and next values
    4. Pass 4: Final forward fill across all data

    This approach is designed specifically for electric load data where
    intra-day patterns are important and should be preserved.

    Attributes:
        date_col: Name of the column containing timestamps or dates.

    Example:
        ```python
        imputer = TriplePassImputer(date_col="timestamp")
        df = imputer.fit_transform(df, value_col="load")
        ```
    """

    def __init__(self, date_col: str = "Data") -> None:
        """Initialize the triple pass imputer.

        Args:
            date_col: Name of the column containing timestamps or dates.
                Used for grouping by date in passes 1 and 2.
        """
        self._date_col = date_col

    @property
    def name(self) -> str:
        """Return the imputer name."""
        return "triple_pass"

    @property
    def date_col(self) -> str:
        """Return the date column name."""
        return self._date_col

    def fit_transform(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Fill missing values using the triple-pass algorithm.

        Performs four passes:
        1. Forward fill within same day
        2. Backward fill within same day
        3. Average interpolation (prev + next) / 2
        4. Final forward fill across days

        Args:
            df: Input DataFrame.
            value_col: Name of the column to impute.

        Returns:
            DataFrame with all missing values filled (if possible).

        Raises:
            KeyError: If value_col or date_col does not exist in DataFrame.
        """
        result = df.copy()

        if value_col not in result.columns:
            msg = f"Column '{value_col}' not found in DataFrame"
            raise KeyError(msg)

        if self._date_col not in result.columns:
            msg = f"Date column '{self._date_col}' not found in DataFrame"
            raise KeyError(msg)

        initial_na = result[value_col].isna().sum()
        if initial_na == 0:
            logger.info("TriplePassImputer: no missing values to fill")
            return result

        logger.info(
            "TriplePassImputer: starting with %d missing values in '%s'",
            initial_na,
            value_col,
        )

        # Extract date from timestamp for grouping
        date_series = pd.to_datetime(result[self._date_col]).dt.date

        # Pass 1: Forward fill within same day
        na_before_pass1 = result[value_col].isna().sum()
        result[value_col] = result.groupby(date_series)[value_col].ffill()
        na_after_pass1 = result[value_col].isna().sum()
        logger.info(
            "TriplePassImputer Pass 1 (forward fill by day): " "filled %d values, %d remaining",
            na_before_pass1 - na_after_pass1,
            na_after_pass1,
        )

        # Pass 2: Backward fill within same day
        na_before_pass2 = result[value_col].isna().sum()
        result[value_col] = result.groupby(date_series)[value_col].bfill()
        na_after_pass2 = result[value_col].isna().sum()
        logger.info(
            "TriplePassImputer Pass 2 (backward fill by day): " "filled %d values, %d remaining",
            na_before_pass2 - na_after_pass2,
            na_after_pass2,
        )

        # Pass 3: Average interpolation (prev + next) / 2
        na_before_pass3 = result[value_col].isna().sum()
        if na_before_pass3 > 0:
            # Get previous and next non-null values
            prev_values = result[value_col].ffill()
            next_values = result[value_col].bfill()

            # Calculate average where both exist
            avg_values = (prev_values + next_values) / 2

            # Fill NaN with average
            mask = result[value_col].isna()
            result.loc[mask, value_col] = avg_values[mask]

        na_after_pass3 = result[value_col].isna().sum()
        logger.info(
            "TriplePassImputer Pass 3 (average interpolation): " "filled %d values, %d remaining",
            na_before_pass3 - na_after_pass3,
            na_after_pass3,
        )

        # Pass 4: Final forward fill across all data
        na_before_pass4 = result[value_col].isna().sum()
        if na_before_pass4 > 0:
            result[value_col] = result[value_col].ffill()

        na_after_pass4 = result[value_col].isna().sum()
        logger.info(
            "TriplePassImputer Pass 4 (final forward fill): " "filled %d values, %d remaining",
            na_before_pass4 - na_after_pass4,
            na_after_pass4,
        )

        # Log final summary
        total_filled = initial_na - na_after_pass4
        logger.info(
            "TriplePassImputer: completed - filled %d/%d missing values " "(%d remaining)",
            total_filled,
            initial_na,
            na_after_pass4,
        )

        return result


class ImputerChain(BaseImputer):
    """Chain multiple imputers to run sequentially.

    Applies a sequence of imputers in order, with each imputer working on
    the output of the previous one. Useful for combining different
    imputation strategies.

    Attributes:
        imputers: List of imputers to apply in sequence.

    Example:
        ```python
        chain = ImputerChain([
            ForwardFillImputer(limit=2),
            BackwardFillImputer(limit=2),
            InterpolationImputer(method="linear"),
        ])
        df = chain.fit_transform(df, value_col="load")
        ```
    """

    def __init__(self, imputers: list[BaseImputer]) -> None:
        """Initialize the imputer chain.

        Args:
            imputers: List of imputer instances to apply in sequence.

        Raises:
            ValueError: If imputers list is empty.
        """
        if not imputers:
            msg = "Imputers list cannot be empty"
            raise ValueError(msg)
        self._imputers = imputers

    @property
    def name(self) -> str:
        """Return the imputer name."""
        imputer_names = "+".join(imp.name for imp in self._imputers)
        return f"chain({imputer_names})"

    @property
    def imputers(self) -> list[BaseImputer]:
        """Return the list of imputers."""
        return self._imputers.copy()

    def fit_transform(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Apply all imputers in sequence.

        Args:
            df: Input DataFrame.
            value_col: Name of the column to impute.

        Returns:
            DataFrame with values filled by all imputers in sequence.
        """
        result = df.copy()
        initial_na = result[value_col].isna().sum()

        logger.debug(
            "ImputerChain: starting with %d missing values in '%s'",
            initial_na,
            value_col,
        )

        for i, imputer in enumerate(self._imputers, 1):
            na_before = result[value_col].isna().sum()
            if na_before == 0:
                logger.debug(
                    "ImputerChain: all values filled after %d imputers",
                    i - 1,
                )
                break

            result = imputer.fit_transform(result, value_col)

            na_after = result[value_col].isna().sum()
            logger.debug(
                "ImputerChain step %d (%s): filled %d values, %d remaining",
                i,
                imputer.name,
                na_before - na_after,
                na_after,
            )

        final_na = result[value_col].isna().sum()
        logger.debug(
            "ImputerChain: completed - filled %d/%d missing values",
            initial_na - final_na,
            initial_na,
        )

        return result


class ResamplerMixin:
    """Mixin class providing resampling utilities for time series data.

    Provides methods for resampling time series data between different
    frequencies, specifically optimized for electric load data in the
    Brazilian electric system.

    Example:
        ```python
        class MyProcessor(ResamplerMixin):
            def process(self, df: pd.DataFrame) -> pd.DataFrame:
                # Upsample hourly to 30-min
                df = self.resample_to_30min(df, "load", "timestamp")
                return df
        ```
    """

    BRAZIL_TZ: ZoneInfo = ZoneInfo("America/Sao_Paulo")

    def resample_to_30min(
        self,
        df: pd.DataFrame,
        value_col: str,
        timestamp_col: str,
        use_lag_fill: bool = True,
    ) -> pd.DataFrame:
        """Resample data to 30-minute frequency.

        Upsamples data from a lower frequency (e.g., hourly) to 30-minute
        intervals. Can use lag filling to handle missing values created
        by upsampling.

        Args:
            df: Input DataFrame.
            value_col: Name of the value column.
            timestamp_col: Name of the timestamp column.
            use_lag_fill: If True, use forward fill to populate new intervals.

        Returns:
            DataFrame resampled to 30-minute frequency.

        Raises:
            KeyError: If required columns are not found.
        """
        result = df.copy()

        if timestamp_col not in result.columns:
            msg = f"Timestamp column '{timestamp_col}' not found in DataFrame"
            raise KeyError(msg)

        if value_col not in result.columns:
            msg = f"Value column '{value_col}' not found in DataFrame"
            raise KeyError(msg)

        # Ensure timestamp is datetime
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])

        # Set timestamp as index for resampling
        result = result.set_index(timestamp_col)

        # Resample to 30-minute frequency
        result = result.resample("30min").asfreq()

        # Handle missing values if requested
        if use_lag_fill:
            result[value_col] = result[value_col].ffill()

        # Reset index to bring timestamp back as column
        result = result.reset_index()

        logger.debug(
            "ResamplerMixin: resampled to 30-min frequency, %d rows",
            len(result),
        )

        return result

    def resample_to_hourly(
        self,
        df: pd.DataFrame,
        value_col: str,
        timestamp_col: str,
        agg_method: str = "mean",
    ) -> pd.DataFrame:
        """Resample data to hourly frequency.

        Downsamples data from a higher frequency (e.g., 30-minute) to hourly
        intervals using the specified aggregation method.

        Args:
            df: Input DataFrame.
            value_col: Name of the value column.
            timestamp_col: Name of the timestamp column.
            agg_method: Aggregation method for downsampling. Options:
                - "mean": Average of values in the hour (default)
                - "sum": Sum of values in the hour
                - "first": First value in the hour
                - "last": Last value in the hour
                - "min": Minimum value in the hour
                - "max": Maximum value in the hour

        Returns:
            DataFrame resampled to hourly frequency.

        Raises:
            KeyError: If required columns are not found.
            ValueError: If agg_method is not valid.
        """
        valid_methods = {"mean", "sum", "first", "last", "min", "max"}
        if agg_method not in valid_methods:
            msg = f"Invalid agg_method '{agg_method}'. Valid: {sorted(valid_methods)}"
            raise ValueError(msg)

        result = df.copy()

        if timestamp_col not in result.columns:
            msg = f"Timestamp column '{timestamp_col}' not found in DataFrame"
            raise KeyError(msg)

        if value_col not in result.columns:
            msg = f"Value column '{value_col}' not found in DataFrame"
            raise KeyError(msg)

        # Ensure timestamp is datetime
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])

        # Set timestamp as index for resampling
        result = result.set_index(timestamp_col)

        # Resample to hourly frequency with specified aggregation
        if agg_method == "mean":
            result = result.resample("h").mean()
        elif agg_method == "sum":
            result = result.resample("h").sum()
        elif agg_method == "first":
            result = result.resample("h").first()
        elif agg_method == "last":
            result = result.resample("h").last()
        elif agg_method == "min":
            result = result.resample("h").min()
        elif agg_method == "max":
            result = result.resample("h").max()

        # Reset index to bring timestamp back as column
        result = result.reset_index()

        logger.debug(
            "ResamplerMixin: resampled to hourly frequency using %s, %d rows",
            agg_method,
            len(result),
        )

        return result


class TimezoneHandler:
    """Utility class for handling timezone conversions for Brazilian data.

    Provides methods for localizing timestamps to Brazil timezone and
    adjusting timestamps for specific use cases in the electric system.

    Example:
        ```python
        handler = TimezoneHandler()

        # Localize naive timestamps to Brazil timezone
        df = handler.localize_to_brazil(df, timestamp_col="timestamp")

        # Adjust timestamps backward by 30 minutes
        df = handler.adjust_timestamp_backward(df, timestamp_col="timestamp", minutes=30)
        ```
    """

    BRAZIL_TZ: ZoneInfo = ZoneInfo("America/Sao_Paulo")

    def localize_to_brazil(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
    ) -> pd.DataFrame:
        """Localize naive timestamps to Brazil timezone.

        Converts timezone-naive timestamps to the America/Sao_Paulo timezone.
        If timestamps are already timezone-aware, converts them to Brazil time.

        Args:
            df: Input DataFrame.
            timestamp_col: Name of the timestamp column.

        Returns:
            DataFrame with timestamps localized to Brazil timezone.

        Raises:
            KeyError: If timestamp_col is not found in DataFrame.
        """
        result = df.copy()

        if timestamp_col not in result.columns:
            msg = f"Timestamp column '{timestamp_col}' not found in DataFrame"
            raise KeyError(msg)

        # Convert to datetime if needed
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])

        # Check if already timezone-aware
        if result[timestamp_col].dt.tz is None:
            # Localize naive timestamps
            result[timestamp_col] = result[timestamp_col].dt.tz_localize(self.BRAZIL_TZ)
        else:
            # Convert existing timezone to Brazil
            result[timestamp_col] = result[timestamp_col].dt.tz_convert(self.BRAZIL_TZ)

        logger.debug(
            "TimezoneHandler: localized %d timestamps to %s",
            len(result),
            self.BRAZIL_TZ,
        )

        return result

    def adjust_timestamp_backward(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        minutes: int = 30,
    ) -> pd.DataFrame:
        """Adjust timestamps backward by a specified number of minutes.

        Subtracts the specified number of minutes from all timestamps.
        Useful for aligning data where timestamps represent end-of-period
        to start-of-period.

        Args:
            df: Input DataFrame.
            timestamp_col: Name of the timestamp column.
            minutes: Number of minutes to subtract from timestamps.

        Returns:
            DataFrame with adjusted timestamps.

        Raises:
            KeyError: If timestamp_col is not found in DataFrame.
            ValueError: If minutes is negative.
        """
        if minutes < 0:
            msg = f"minutes must be non-negative, got {minutes}"
            raise ValueError(msg)

        result = df.copy()

        if timestamp_col not in result.columns:
            msg = f"Timestamp column '{timestamp_col}' not found in DataFrame"
            raise KeyError(msg)

        # Convert to datetime if needed
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])

        # Adjust backward
        result[timestamp_col] = result[timestamp_col] - pd.Timedelta(minutes=minutes)

        logger.debug(
            "TimezoneHandler: adjusted %d timestamps backward by %d minutes",
            len(result),
            minutes,
        )

        return result
