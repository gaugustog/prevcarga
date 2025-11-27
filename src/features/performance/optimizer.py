"""Memory optimization utilities for DataFrames.

This module provides utilities for optimizing DataFrame memory usage
through dtype optimization, categorical encoding, and duplicate removal.

Example:
    ```python
    from src.features.performance.optimizer import MemoryOptimizer

    optimizer = MemoryOptimizer()

    # Optimize DataFrame
    optimized_df, report = optimizer.optimize(df)
    print(f"Memory saved: {report['memory_saved_mb']:.1f}MB")
    print(f"Reduction: {report['reduction_percent']:.1f}%")
    ```
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryOptimizer:
    """Optimize DataFrame memory usage.

    Performs various memory optimization techniques:
    - Downcast numeric types (int64 -> int32, float64 -> float32)
    - Convert object columns to category dtype
    - Remove duplicate columns
    - Identify sparse columns

    Attributes:
        aggressive: Enable aggressive optimizations (may lose precision).

    Example:
        >>> optimizer = MemoryOptimizer(aggressive=False)
        >>> optimized_df, report = optimizer.optimize(df)
        >>> print(report)
    """

    def __init__(self, aggressive: bool = False):
        """Initialize memory optimizer.

        Args:
            aggressive: Enable aggressive optimizations.
                If True, uses more aggressive downcasting that may
                lose precision but saves more memory.
        """
        self.aggressive = aggressive
        logger.info(f"Initialized MemoryOptimizer (aggressive={aggressive})")

    def optimize(
        self, df: pd.DataFrame, inplace: bool = False
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Optimize DataFrame memory usage.

        Applies multiple optimization techniques and returns both
        the optimized DataFrame and a report of changes made.

        Args:
            df: DataFrame to optimize.
            inplace: Modify DataFrame in place (faster, less memory).

        Returns:
            Tuple of (optimized_df, report) where report contains:
                - original_memory_mb: Original memory usage
                - optimized_memory_mb: Optimized memory usage
                - memory_saved_mb: Memory saved
                - reduction_percent: Percentage reduction
                - optimizations_applied: List of applied optimizations

        Example:
            >>> optimized_df, report = optimizer.optimize(df)
            >>> print(f"Saved {report['memory_saved_mb']:.1f}MB")
        """
        if not inplace:
            df = df.copy()

        original_memory_mb = self._get_memory_usage(df)
        optimizations_applied = []

        # Downcast numeric columns
        df, numeric_opts = self._optimize_numeric(df)
        if numeric_opts:
            optimizations_applied.extend(numeric_opts)

        # Convert object columns to category
        df, category_opts = self._optimize_categorical(df)
        if category_opts:
            optimizations_applied.extend(category_opts)

        # Remove duplicate columns
        df, dup_count = self._remove_duplicate_columns(df)
        if dup_count > 0:
            optimizations_applied.append(
                f"Removed {dup_count} duplicate columns"
            )

        optimized_memory_mb = self._get_memory_usage(df)
        memory_saved_mb = original_memory_mb - optimized_memory_mb
        reduction_percent = (
            (memory_saved_mb / original_memory_mb * 100)
            if original_memory_mb > 0
            else 0.0
        )

        report = {
            "original_memory_mb": original_memory_mb,
            "optimized_memory_mb": optimized_memory_mb,
            "memory_saved_mb": memory_saved_mb,
            "reduction_percent": reduction_percent,
            "optimizations_applied": optimizations_applied,
        }

        logger.info(
            f"Memory optimization: {original_memory_mb:.1f}MB -> "
            f"{optimized_memory_mb:.1f}MB ({reduction_percent:.1f}% reduction)"
        )

        return df, report

    def _optimize_numeric(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Optimize numeric column dtypes.

        Args:
            df: DataFrame to optimize.

        Returns:
            Tuple of (optimized_df, optimizations_applied).
        """
        optimizations = []

        # Integer columns
        int_cols = df.select_dtypes(include=["int64"]).columns
        for col in int_cols:
            col_min = df[col].min()
            col_max = df[col].max()

            # Try to downcast to smaller integer types
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(
                np.int8
            ).max:
                df[col] = df[col].astype(np.int8)
                optimizations.append(f"{col}: int64 -> int8")
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(
                np.int16
            ).max:
                df[col] = df[col].astype(np.int16)
                optimizations.append(f"{col}: int64 -> int16")
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(
                np.int32
            ).max:
                df[col] = df[col].astype(np.int32)
                optimizations.append(f"{col}: int64 -> int32")

        # Float columns
        float_cols = df.select_dtypes(include=["float64"]).columns
        for col in float_cols:
            if self.aggressive:
                # Aggressive: always downcast to float32
                df[col] = df[col].astype(np.float32)
                optimizations.append(f"{col}: float64 -> float32")
            else:
                # Conservative: only downcast if no precision loss
                if self._safe_downcast_float(df[col]):
                    df[col] = df[col].astype(np.float32)
                    optimizations.append(f"{col}: float64 -> float32")

        return df, optimizations

    def _safe_downcast_float(self, series: pd.Series) -> bool:
        """Check if float64 series can be safely downcast to float32.

        Args:
            series: Series to check.

        Returns:
            True if safe to downcast, False otherwise.
        """
        # Check if values are within float32 range
        float32_min = np.finfo(np.float32).min
        float32_max = np.finfo(np.float32).max

        non_null = series.dropna()
        if len(non_null) == 0:
            return True

        return (non_null >= float32_min).all() and (
            non_null <= float32_max
        ).all()

    def _optimize_categorical(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Convert object columns to category dtype where beneficial.

        Args:
            df: DataFrame to optimize.

        Returns:
            Tuple of (optimized_df, optimizations_applied).
        """
        optimizations = []

        obj_cols = df.select_dtypes(include=["object"]).columns

        for col in obj_cols:
            # Convert to category if cardinality is low relative to length
            n_unique = df[col].nunique()
            n_total = len(df[col])

            # Use category if unique values are less than 50% of total
            if n_unique < n_total * 0.5:
                original_memory = df[col].memory_usage(deep=True)
                df[col] = df[col].astype("category")
                new_memory = df[col].memory_usage(deep=True)

                if new_memory < original_memory:
                    memory_saved_mb = (original_memory - new_memory) / (
                        1024 * 1024
                    )
                    optimizations.append(
                        f"{col}: object -> category "
                        f"(saved {memory_saved_mb:.1f}MB)"
                    )
                else:
                    # Revert if no benefit
                    df[col] = df[col].astype("object")

        return df, optimizations

    def _remove_duplicate_columns(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, int]:
        """Remove duplicate columns.

        Args:
            df: DataFrame to process.

        Returns:
            Tuple of (df_without_duplicates, num_duplicates_removed).
        """
        # Find duplicate columns
        duplicate_cols = []
        cols_seen = {}

        for col in df.columns:
            col_hash = pd.util.hash_pandas_object(df[col]).sum()

            if col_hash in cols_seen:
                duplicate_cols.append(col)
                logger.debug(
                    f"Found duplicate column: {col} "
                    f"(duplicate of {cols_seen[col_hash]})"
                )
            else:
                cols_seen[col_hash] = col

        if duplicate_cols:
            df = df.drop(columns=duplicate_cols)
            logger.info(f"Removed {len(duplicate_cols)} duplicate columns")

        return df, len(duplicate_cols)

    def identify_sparse_columns(
        self, df: pd.DataFrame, threshold: float = 0.95
    ) -> list[str]:
        """Identify sparse columns that could use sparse storage.

        Args:
            df: DataFrame to analyze.
            threshold: Sparsity threshold (fraction of zeros/nulls).

        Returns:
            List of column names that are sparse.

        Example:
            >>> sparse_cols = optimizer.identify_sparse_columns(df)
            >>> print(f"Sparse columns: {sparse_cols}")
        """
        sparse_columns = []

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Count zeros and nulls
                sparsity = (
                    (df[col] == 0).sum() + df[col].isna().sum()
                ) / len(df[col])

                if sparsity >= threshold:
                    sparse_columns.append(col)
                    logger.debug(
                        f"Column {col} is {sparsity:.1%} sparse "
                        f"(threshold: {threshold:.1%})"
                    )

        return sparse_columns

    @staticmethod
    def _get_memory_usage(df: pd.DataFrame) -> float:
        """Get DataFrame memory usage in MB.

        Args:
            df: DataFrame to measure.

        Returns:
            Memory usage in megabytes.
        """
        return df.memory_usage(deep=True).sum() / (1024 * 1024)

    def get_memory_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate detailed memory usage report by column.

        Args:
            df: DataFrame to analyze.

        Returns:
            DataFrame with memory usage details per column.

        Example:
            >>> report = optimizer.get_memory_report(df)
            >>> print(report.sort_values('memory_mb', ascending=False))
        """
        memory_usage = df.memory_usage(deep=True)

        report = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": df.dtypes.values,
                "memory_bytes": [memory_usage[col] for col in df.columns],
            }
        )

        report["memory_mb"] = report["memory_bytes"] / (1024 * 1024)
        report["percent"] = (
            report["memory_bytes"] / memory_usage.sum() * 100
        )

        # Add non-null count and unique count
        report["non_null_count"] = [df[col].count() for col in df.columns]
        report["unique_count"] = [df[col].nunique() for col in df.columns]

        return report.sort_values("memory_mb", ascending=False).reset_index(
            drop=True
        )
