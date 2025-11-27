"""Execution profiling for feature engineering.

This module provides execution time and memory profiling utilities
for tracking performance bottlenecks in the feature engineering pipeline.

Example:
    ```python
    from src.features.performance.profiler import ExecutionProfiler, profile_execution

    profiler = ExecutionProfiler()

    # Manual profiling
    with profiler.profile("feature_generation"):
        features = plugin.generate_features(df, config)

    # Decorator-based profiling
    @profile_execution(profiler, "compute_lags")
    def compute_lags(df):
        return df.shift(1)

    # Get profiling report
    report = profiler.get_report()
    print(report)
    ```
"""

import functools
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Generator, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionProfiler:
    """Execution time and memory profiler for feature engineering.

    Tracks execution time and memory usage for individual operations,
    enabling identification of performance bottlenecks.

    Attributes:
        results: DataFrame storing profiling results.
        enable_memory_profiling: Whether to track memory usage.

    Example:
        >>> profiler = ExecutionProfiler(enable_memory=True)
        >>>
        >>> with profiler.profile("feature_computation"):
        ...     result = expensive_operation()
        >>>
        >>> summary = profiler.get_summary()
        >>> print(summary)
    """

    def __init__(self, enable_memory: bool = True):
        """Initialize execution profiler.

        Args:
            enable_memory: Enable memory profiling (requires psutil).
        """
        self.enable_memory_profiling = enable_memory
        self.results: list[dict[str, Any]] = []

        if enable_memory and not self._check_psutil():
            logger.warning(
                "psutil not available, memory profiling disabled. "
                "Install with: pip install psutil"
            )
            self.enable_memory_profiling = False

        logger.info(
            f"Initialized ExecutionProfiler (memory profiling: "
            f"{self.enable_memory_profiling})"
        )

    @staticmethod
    def _check_psutil() -> bool:
        """Check if psutil is available.

        Returns:
            True if psutil can be imported, False otherwise.
        """
        try:
            import psutil

            return True
        except ImportError:
            return False

    @contextmanager
    def profile(
        self, operation_name: str, metadata: Optional[dict[str, Any]] = None
    ) -> Generator[None, None, None]:
        """Context manager for profiling operations.

        Args:
            operation_name: Name of the operation being profiled.
            metadata: Optional metadata to store with profiling result.

        Yields:
            None

        Example:
            >>> profiler = ExecutionProfiler()
            >>> with profiler.profile("compute_features", {"plugin": "lag"}):
            ...     features = compute_features(df)
        """
        start_time = time.perf_counter()
        start_memory_mb = self._get_memory_usage() if self.enable_memory_profiling else 0.0

        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory_mb = self._get_memory_usage() if self.enable_memory_profiling else 0.0

            execution_time = end_time - start_time
            peak_memory_mb = end_memory_mb  # Approximate
            memory_delta_mb = end_memory_mb - start_memory_mb

            result = {
                "operation": operation_name,
                "timestamp": datetime.now(),
                "execution_time_seconds": execution_time,
                "start_memory_mb": start_memory_mb,
                "end_memory_mb": end_memory_mb,
                "memory_delta_mb": memory_delta_mb,
                "peak_memory_mb": peak_memory_mb,
                **(metadata or {}),
            }

            self.results.append(result)

            logger.info(
                f"Profiled '{operation_name}': {execution_time:.3f}s, "
                f"memory: {memory_delta_mb:+.1f}MB"
            )

    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB.

        Returns:
            Memory usage in megabytes.
        """
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
        except Exception as e:
            logger.debug(f"Failed to get memory usage: {e}")
            return 0.0

    def get_results_df(self) -> pd.DataFrame:
        """Get profiling results as DataFrame.

        Returns:
            DataFrame with profiling results, or empty DataFrame if no results.

        Example:
            >>> df = profiler.get_results_df()
            >>> print(df[['operation', 'execution_time_seconds']].head())
        """
        if not self.results:
            return pd.DataFrame()

        return pd.DataFrame(self.results)

    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics for profiled operations.

        Groups results by operation name and calculates aggregate statistics.

        Returns:
            DataFrame with summary statistics (mean, total, count, min, max).

        Example:
            >>> summary = profiler.get_summary()
            >>> print(summary.sort_values('total_time_seconds', ascending=False))
        """
        if not self.results:
            logger.warning("No profiling results available")
            return pd.DataFrame()

        df = self.get_results_df()

        summary = df.groupby("operation").agg(
            {
                "execution_time_seconds": ["count", "mean", "sum", "min", "max"],
                "memory_delta_mb": ["mean", "sum", "min", "max"],
            }
        )

        # Flatten multi-index columns
        summary.columns = [
            "_".join(col).strip() for col in summary.columns.values
        ]

        # Rename for clarity
        summary = summary.rename(
            columns={
                "execution_time_seconds_count": "call_count",
                "execution_time_seconds_mean": "mean_time_seconds",
                "execution_time_seconds_sum": "total_time_seconds",
                "execution_time_seconds_min": "min_time_seconds",
                "execution_time_seconds_max": "max_time_seconds",
                "memory_delta_mb_mean": "mean_memory_delta_mb",
                "memory_delta_mb_sum": "total_memory_delta_mb",
                "memory_delta_mb_min": "min_memory_delta_mb",
                "memory_delta_mb_max": "max_memory_delta_mb",
            }
        )

        return summary.sort_values("total_time_seconds", ascending=False)

    def get_report(self, top_n: int = 10) -> str:
        """Generate human-readable profiling report.

        Args:
            top_n: Number of top operations to include in report.

        Returns:
            Formatted report string.

        Example:
            >>> report = profiler.get_report(top_n=5)
            >>> print(report)
        """
        if not self.results:
            return "No profiling results available."

        summary = self.get_summary()

        lines = [
            "=" * 80,
            "EXECUTION PROFILING REPORT",
            "=" * 80,
            f"Total operations profiled: {len(self.results)}",
            f"Unique operations: {len(summary)}",
            "",
            f"Top {min(top_n, len(summary))} operations by total time:",
            "-" * 80,
        ]

        for idx, (operation, row) in enumerate(
            summary.head(top_n).iterrows(), 1
        ):
            lines.append(
                f"{idx}. {operation}:\n"
                f"   Total time: {row['total_time_seconds']:.3f}s\n"
                f"   Mean time: {row['mean_time_seconds']:.3f}s\n"
                f"   Calls: {int(row['call_count'])}\n"
                f"   Memory delta: {row['mean_memory_delta_mb']:+.1f}MB (mean)\n"
            )

        lines.append("=" * 80)

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all profiling results.

        Example:
            >>> profiler.reset()
        """
        self.results.clear()
        logger.info("Reset profiler results")

    def save_results(self, path: str) -> None:
        """Save profiling results to file.

        Args:
            path: Path to save results (supports .csv, .parquet, .json).

        Example:
            >>> profiler.save_results("profiling_results.csv")
        """
        if not self.results:
            logger.warning("No results to save")
            return

        df = self.get_results_df()

        if path.endswith(".csv"):
            df.to_csv(path, index=False)
        elif path.endswith(".parquet"):
            df.to_parquet(path, index=False)
        elif path.endswith(".json"):
            df.to_json(path, orient="records", date_format="iso")
        else:
            raise ValueError(
                f"Unsupported file format: {path}. "
                f"Supported: .csv, .parquet, .json"
            )

        logger.info(f"Saved profiling results to {path}")


def profile_execution(
    profiler: ExecutionProfiler,
    operation_name: Optional[str] = None,
    **metadata: Any,
) -> Callable:
    """Decorator for automatic execution profiling.

    Args:
        profiler: ExecutionProfiler instance to use.
        operation_name: Name for the operation (defaults to function name).
        **metadata: Additional metadata to store with profiling result.

    Returns:
        Decorated function.

    Example:
        >>> profiler = ExecutionProfiler()
        >>>
        >>> @profile_execution(profiler, "compute_lags")
        >>> def compute_lags(df):
        ...     return df.shift(1)
        >>>
        >>> result = compute_lags(data)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            op_name = operation_name or func.__name__

            with profiler.profile(op_name, metadata=metadata):
                return func(*args, **kwargs)

        return wrapper

    return decorator
