"""Tests for memory optimizer."""

import numpy as np
import pandas as pd
import pytest

from src.features.performance.optimizer import MemoryOptimizer


@pytest.fixture
def sample_df():
    """Create sample DataFrame with various dtypes."""
    return pd.DataFrame(
        {
            "int64_col": np.array([1, 2, 3, 4, 5], dtype=np.int64),
            "int64_small": np.array([1, 2, 3, 4, 5], dtype=np.int64),  # Can be int8
            "float64_col": np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64),
            "object_col": pd.Series(["a", "b", "c", "a", "b"], dtype=object),
            "high_cardinality": pd.Series(
                ["x1", "x2", "x3", "x4", "x5"], dtype=object
            ),
        }
    )


def test_optimizer_basic():
    """Test basic memory optimization."""
    optimizer = MemoryOptimizer(aggressive=False)

    df = pd.DataFrame(
        {
            "int_col": np.array([1, 2, 3], dtype=np.int64),
            "float_col": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        }
    )

    original_memory = df.memory_usage(deep=True).sum()

    optimized_df, report = optimizer.optimize(df)

    assert report["original_memory_mb"] > 0
    assert report["optimized_memory_mb"] <= report["original_memory_mb"]
    assert report["memory_saved_mb"] >= 0
    assert 0 <= report["reduction_percent"] <= 100


def test_optimizer_int_downcast():
    """Test integer downcasting."""
    optimizer = MemoryOptimizer(aggressive=False)

    df = pd.DataFrame({"small_int": np.array([1, 2, 3, 4, 5], dtype=np.int64)})

    optimized_df, report = optimizer.optimize(df)

    # Should be downcasted to int8
    assert optimized_df["small_int"].dtype == np.int8
    assert "small_int" in str(report["optimizations_applied"])


def test_optimizer_categorical_conversion():
    """Test categorical conversion."""
    optimizer = MemoryOptimizer(aggressive=False)

    # Create DataFrame with repeated strings
    df = pd.DataFrame({"category_col": ["a", "b", "c"] * 100})

    optimized_df, report = optimizer.optimize(df)

    # Should be converted to category
    assert optimized_df["category_col"].dtype.name == "category"


def test_optimizer_no_categorical_high_cardinality():
    """Test that high cardinality columns don't become categorical."""
    optimizer = MemoryOptimizer(aggressive=False)

    # Create DataFrame with unique strings (high cardinality)
    df = pd.DataFrame({"unique_col": [f"value_{i}" for i in range(100)]})

    optimized_df, report = optimizer.optimize(df)

    # Should remain object (not category)
    assert optimized_df["unique_col"].dtype == object


def test_optimizer_aggressive_mode():
    """Test aggressive optimization mode."""
    optimizer_aggressive = MemoryOptimizer(aggressive=True)
    optimizer_conservative = MemoryOptimizer(aggressive=False)

    df = pd.DataFrame({"float_col": np.array([1.0, 2.0, 3.0], dtype=np.float64)})

    optimized_aggressive, report_agg = optimizer_aggressive.optimize(df.copy())
    optimized_conservative, report_cons = optimizer_conservative.optimize(
        df.copy()
    )

    # Aggressive should always downcast floats
    assert optimized_aggressive["float_col"].dtype == np.float32


def test_optimizer_inplace():
    """Test in-place optimization."""
    optimizer = MemoryOptimizer(aggressive=False)

    df = pd.DataFrame({"int_col": np.array([1, 2, 3], dtype=np.int64)})

    df_id_before = id(df)

    optimized_df, report = optimizer.optimize(df, inplace=True)

    # Should be same object
    assert id(df) == df_id_before


def test_optimizer_duplicate_column_removal():
    """Test duplicate column removal."""
    optimizer = MemoryOptimizer(aggressive=False)

    df = pd.DataFrame(
        {"col1": [1, 2, 3], "col2": [1, 2, 3], "col3": [4, 5, 6]}  # Duplicate of col1
    )

    optimized_df, report = optimizer.optimize(df)

    # Should have removed one duplicate
    assert len(optimized_df.columns) == 2
    assert "duplicate" in str(report["optimizations_applied"]).lower()


def test_optimizer_identify_sparse_columns():
    """Test sparse column identification."""
    optimizer = MemoryOptimizer(aggressive=False)

    df = pd.DataFrame(
        {
            "sparse_col": [0] * 95 + [1] * 5,  # 95% zeros
            "dense_col": list(range(100)),
        }
    )

    sparse_cols = optimizer.identify_sparse_columns(df, threshold=0.9)

    assert "sparse_col" in sparse_cols
    assert "dense_col" not in sparse_cols


def test_optimizer_memory_report():
    """Test memory usage report generation."""
    optimizer = MemoryOptimizer(aggressive=False)

    df = pd.DataFrame(
        {
            "col1": np.random.randn(1000),
            "col2": np.random.randint(0, 100, 1000),
            "col3": ["a", "b", "c"] * 333 + ["a"],
        }
    )

    report = optimizer.get_memory_report(df)

    assert len(report) == 3
    assert "column" in report.columns
    assert "dtype" in report.columns
    assert "memory_mb" in report.columns
    assert "percent" in report.columns
    assert "unique_count" in report.columns

    # Should be sorted by memory usage
    assert report["memory_mb"].iloc[0] >= report["memory_mb"].iloc[-1]


def test_optimizer_preserves_data():
    """Test that optimization preserves data values."""
    optimizer = MemoryOptimizer(aggressive=False)

    df = pd.DataFrame(
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "str_col": ["a", "b", "c", "d", "e"],
        }
    )

    optimized_df, report = optimizer.optimize(df)

    # Values should be preserved
    assert (optimized_df["int_col"].values == df["int_col"].values).all()
    assert np.allclose(
        optimized_df["float_col"].values, df["float_col"].values
    )
    assert (optimized_df["str_col"].values == df["str_col"].values).all()


def test_optimizer_empty_dataframe():
    """Test optimizer with empty DataFrame."""
    optimizer = MemoryOptimizer(aggressive=False)

    df = pd.DataFrame()

    optimized_df, report = optimizer.optimize(df)

    assert len(optimized_df) == 0
    assert report["memory_saved_mb"] == 0


def test_optimizer_with_nulls():
    """Test optimizer handles null values."""
    optimizer = MemoryOptimizer(aggressive=False)

    df = pd.DataFrame(
        {
            "int_with_nulls": [1, 2, None, 4, 5],
            "float_with_nulls": [1.1, None, 3.3, None, 5.5],
            "str_with_nulls": ["a", None, "c", "a", None],
        }
    )

    optimized_df, report = optimizer.optimize(df)

    # Should handle nulls without errors
    assert optimized_df["int_with_nulls"].isna().sum() == 1
    assert optimized_df["float_with_nulls"].isna().sum() == 2
    assert optimized_df["str_with_nulls"].isna().sum() == 2
