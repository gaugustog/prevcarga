"""Tests for seasonal strength calculator.

This module tests the SeasonalStrengthCalculator which computes metrics
for evaluating seasonal and trend components in time series decomposition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.metrics.seasonal_strength import SeasonalStrengthCalculator


@pytest.fixture
def calculator():
    """Create calculator instance."""
    return SeasonalStrengthCalculator()


class TestSeasonalStrengthCalculator:
    """Tests for SeasonalStrengthCalculator."""

    def test_strong_seasonality(self, calculator):
        """Test with strong seasonality."""
        # Strong seasonal component with small residual
        seasonal = pd.Series(100 * np.sin(2 * np.pi * np.arange(100) / 10))
        residual = pd.Series(np.random.RandomState(42).randn(100) * 5)

        strength = calculator.calculate_seasonal_strength(seasonal, residual)

        # Should be high (> 0.6)
        assert strength > 0.6
        assert calculator.classify_strength(strength) == "strong"

    def test_moderate_seasonality(self, calculator):
        """Test with moderate seasonality."""
        # Moderate seasonal component
        seasonal = pd.Series(50 * np.sin(2 * np.pi * np.arange(100) / 10))
        residual = pd.Series(np.random.RandomState(42).randn(100) * 40)

        strength = calculator.calculate_seasonal_strength(seasonal, residual)

        # Should be moderate (0.3 to 0.6)
        assert 0.3 < strength < 0.6
        assert calculator.classify_strength(strength) == "moderate"

    def test_weak_seasonality(self, calculator):
        """Test with weak seasonality."""
        # Weak seasonal component with large residual
        seasonal = pd.Series(np.sin(2 * np.pi * np.arange(100) / 10))
        residual = pd.Series(np.random.RandomState(42).randn(100) * 100)

        strength = calculator.calculate_seasonal_strength(seasonal, residual)

        # Should be low (< 0.3)
        assert strength < 0.5
        assert calculator.classify_strength(strength) in ["weak", "moderate"]

    def test_no_seasonality(self, calculator):
        """Test with no seasonality."""
        # No seasonal component
        seasonal = pd.Series(np.zeros(100))
        residual = pd.Series(np.random.RandomState(42).randn(100))

        strength = calculator.calculate_seasonal_strength(seasonal, residual)

        # Should be very low or zero
        assert strength < 0.1
        assert calculator.classify_strength(strength) == "weak"

    def test_perfect_seasonality(self, calculator):
        """Test with perfect seasonality (no residual)."""
        # Perfect seasonal component with no residual
        seasonal = pd.Series(100 * np.sin(2 * np.pi * np.arange(100) / 10))
        residual = pd.Series(np.zeros(100))

        strength = calculator.calculate_seasonal_strength(seasonal, residual)

        # Should be 1.0 (perfect)
        assert strength == pytest.approx(1.0, abs=1e-10)
        assert calculator.classify_strength(strength) == "strong"

    def test_constant_series(self, calculator):
        """Test with constant series (zero variance)."""
        # Constant seasonal component and residual
        seasonal = pd.Series(np.ones(100) * 10)
        residual = pd.Series(np.zeros(100))

        strength = calculator.calculate_seasonal_strength(seasonal, residual)

        # Should return 0 to avoid division by zero
        assert strength == 0.0

    def test_zero_variance_detrended(self, calculator):
        """Test with zero variance in detrended series."""
        # Edge case: seasonal + residual has zero variance
        seasonal = pd.Series(np.ones(100) * 5)
        residual = pd.Series(np.ones(100) * -5)

        strength = calculator.calculate_seasonal_strength(seasonal, residual)

        # Should return 0 to avoid division by zero
        assert strength == 0.0


class TestTrendStrengthCalculator:
    """Tests for trend strength calculation."""

    def test_strong_trend(self, calculator):
        """Test with strong trend."""
        # Strong linear trend with small residual
        trend = pd.Series(np.linspace(0, 100, 100))
        residual = pd.Series(np.random.RandomState(42).randn(100) * 2)

        strength = calculator.calculate_trend_strength(trend, residual)

        # Should be high (> 0.6)
        assert strength > 0.6
        assert calculator.classify_strength(strength) == "strong"

    def test_moderate_trend(self, calculator):
        """Test with moderate trend."""
        # Moderate trend
        trend = pd.Series(np.linspace(0, 50, 100))
        residual = pd.Series(np.random.RandomState(42).randn(100) * 30)

        strength = calculator.calculate_trend_strength(trend, residual)

        # Should be moderate (0.3 to 0.6)
        assert 0.2 < strength < 0.7
        assert calculator.classify_strength(strength) in ["moderate", "weak", "strong"]

    def test_weak_trend(self, calculator):
        """Test with weak trend."""
        # Weak trend with large residual
        trend = pd.Series(np.linspace(0, 10, 100))
        residual = pd.Series(np.random.RandomState(42).randn(100) * 100)

        strength = calculator.calculate_trend_strength(trend, residual)

        # Should be low (< 0.3)
        assert strength < 0.5

    def test_no_trend(self, calculator):
        """Test with no trend."""
        # No trend (constant)
        trend = pd.Series(np.ones(100) * 50)
        residual = pd.Series(np.random.RandomState(42).randn(100))

        strength = calculator.calculate_trend_strength(trend, residual)

        # Should be very low
        assert strength < 0.1
        assert calculator.classify_strength(strength) == "weak"

    def test_perfect_trend(self, calculator):
        """Test with perfect trend (no residual)."""
        # Perfect trend with no residual
        trend = pd.Series(np.linspace(0, 100, 100))
        residual = pd.Series(np.zeros(100))

        strength = calculator.calculate_trend_strength(trend, residual)

        # Should be 1.0 (perfect)
        assert strength == pytest.approx(1.0, abs=1e-10)
        assert calculator.classify_strength(strength) == "strong"


class TestStrengthClassification:
    """Tests for strength classification."""

    def test_classify_strong(self, calculator):
        """Test classification of strong strength."""
        assert calculator.classify_strength(0.61) == "strong"
        assert calculator.classify_strength(0.8) == "strong"
        assert calculator.classify_strength(1.0) == "strong"

    def test_classify_moderate(self, calculator):
        """Test classification of moderate strength."""
        assert calculator.classify_strength(0.31) == "moderate"
        assert calculator.classify_strength(0.45) == "moderate"
        assert calculator.classify_strength(0.6) == "moderate"

    def test_classify_weak(self, calculator):
        """Test classification of weak strength."""
        assert calculator.classify_strength(0.0) == "weak"
        assert calculator.classify_strength(0.15) == "weak"
        assert calculator.classify_strength(0.3) == "weak"

    def test_boundary_values(self, calculator):
        """Test boundary values."""
        # At 0.6 boundary
        assert calculator.classify_strength(0.6) == "moderate"
        assert calculator.classify_strength(0.60001) == "strong"

        # At 0.3 boundary
        assert calculator.classify_strength(0.3) == "weak"
        assert calculator.classify_strength(0.30001) == "moderate"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_series(self, calculator):
        """Test with empty series."""
        seasonal = pd.Series(dtype=float)
        residual = pd.Series(dtype=float)

        # Should handle gracefully by returning 0
        strength = calculator.calculate_seasonal_strength(seasonal, residual)
        assert strength == 0.0

    def test_single_value(self, calculator):
        """Test with single value."""
        seasonal = pd.Series([1.0])
        residual = pd.Series([0.0])

        # Variance of single value is 0
        strength = calculator.calculate_seasonal_strength(seasonal, residual)
        assert strength == 0.0

    def test_negative_values(self, calculator):
        """Test with negative values."""
        # Seasonal and residual can be negative
        seasonal = pd.Series([-10, -5, 0, 5, 10])
        residual = pd.Series([-1, 0, 1, 0, -1])

        # Should work correctly
        strength = calculator.calculate_seasonal_strength(seasonal, residual)
        assert 0 <= strength <= 1

    def test_large_values(self, calculator):
        """Test with large values."""
        seasonal = pd.Series(np.arange(1000) * 1e6)
        residual = pd.Series(np.random.RandomState(42).randn(1000) * 1e3)

        # Should work correctly with large values
        strength = calculator.calculate_seasonal_strength(seasonal, residual)
        assert 0 <= strength <= 1

    def test_identical_components(self, calculator):
        """Test when seasonal and residual are identical."""
        data = pd.Series(np.random.RandomState(42).randn(100))
        seasonal = data.copy()
        residual = data.copy()

        strength = calculator.calculate_seasonal_strength(seasonal, residual)

        # var(residual) / var(seasonal + residual)
        # = var(data) / var(2*data)
        # = var(data) / (4*var(data))
        # = 1/4
        # strength = 1 - 1/4 = 0.75
        assert strength == pytest.approx(0.75, abs=0.01)
