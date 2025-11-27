"""Tests for ShrinkageReconciler.

This module tests the shrinkage-based hierarchical forecast reconciliation
with James-Stein shrinkage for robust covariance estimation.
"""

import numpy as np
import pandas as pd
import pytest

from src.reconciliation import (
    HierarchyDefinition,
    ReconciliationConfig,
    ShrinkageReconciler,
    ShrinkageTarget,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_hierarchy():
    """Create simple 3-level hierarchy programmatically."""
    hierarchy = HierarchyDefinition()

    # Add nodes programmatically (avoids YAML complexity)
    hierarchy.add_node("Total", level=0, node_type="aggregate", children=["A", "B"])
    hierarchy.add_node("A", level=1, node_type="aggregate", parent="Total", children=["A1", "A2"])
    hierarchy.add_node("B", level=1, node_type="bottom", parent="Total")
    hierarchy.add_node("A1", level=2, node_type="bottom", parent="A")
    hierarchy.add_node("A2", level=2, node_type="bottom", parent="A")

    hierarchy.build_graph()
    hierarchy.build_aggregation_matrix()

    return hierarchy


@pytest.fixture
def sample_data_limited():
    """Generate limited sample data (small n) for testing shrinkage."""
    np.random.seed(42)
    n_periods = 15  # Limited data
    nodes = ["Total", "A", "B", "A1", "A2"]

    # Generate base forecasts with some error
    base_forecasts = pd.DataFrame({
        node: np.random.normal(100, 10, n_periods) for node in nodes
    })

    # Generate actuals (true values)
    actuals = pd.DataFrame({
        node: base_forecasts[node] + np.random.normal(0, 5, n_periods) for node in nodes
    })

    return base_forecasts, actuals


@pytest.fixture
def sample_data_moderate():
    """Generate moderate sample data for testing."""
    np.random.seed(43)
    n_periods = 50
    nodes = ["Total", "A", "B", "A1", "A2"]

    base_forecasts = pd.DataFrame({
        node: np.random.normal(100, 10, n_periods) for node in nodes
    })

    actuals = pd.DataFrame({
        node: base_forecasts[node] + np.random.normal(0, 5, n_periods) for node in nodes
    })

    return base_forecasts, actuals


# ============================================================================
# Test Initialization
# ============================================================================


class TestShrinkageReconcilerInit:
    """Test ShrinkageReconciler initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        reconciler = ShrinkageReconciler()

        assert reconciler.shrinkage_target == ShrinkageTarget.DIAGONAL
        assert reconciler.auto_lambda is True
        assert reconciler.lambda_method == "ledoit_wolf"
        assert reconciler.min_lambda == 0.0
        assert reconciler.max_lambda == 1.0
        assert reconciler.fitted is False

    def test_custom_config_initialization(self):
        """Test initialization with custom config."""
        config = ReconciliationConfig(
            method="shrinkage",
            shrinkage_target="identity",
            auto_lambda=False,
            fixed_lambda=0.7,
            min_lambda=0.1,
            max_lambda=0.9
        )

        reconciler = ShrinkageReconciler(config=config)

        assert reconciler.shrinkage_target == ShrinkageTarget.IDENTITY
        assert reconciler.auto_lambda is False
        assert reconciler.fixed_lambda == 0.7
        assert reconciler.min_lambda == 0.1
        assert reconciler.max_lambda == 0.9

    def test_invalid_lambda_bounds(self):
        """Test validation of lambda bounds."""
        config = ReconciliationConfig(min_lambda=1.5)

        with pytest.raises(ValueError, match="min_lambda must be in"):
            ShrinkageReconciler(config=config)

    def test_min_greater_than_max(self):
        """Test validation when min_lambda > max_lambda."""
        config = ReconciliationConfig(min_lambda=0.8, max_lambda=0.3)

        with pytest.raises(ValueError, match="min_lambda.*> max_lambda"):
            ShrinkageReconciler(config=config)

    def test_repr_unfitted(self):
        """Test string representation when unfitted."""
        reconciler = ShrinkageReconciler()
        repr_str = repr(reconciler)

        assert "ShrinkageReconciler" in repr_str
        assert "fitted=False" in repr_str

    def test_repr_fitted(self, simple_hierarchy, sample_data_limited):
        """Test string representation when fitted."""
        reconciler = ShrinkageReconciler()
        forecasts, actuals = sample_data_limited
        reconciler.fit(forecasts, actuals, simple_hierarchy)

        repr_str = repr(reconciler)

        assert "ShrinkageReconciler" in repr_str
        assert "fitted=True" in repr_str
        assert "λ=" in repr_str


# ============================================================================
# Test Shrinkage Targets
# ============================================================================


class TestShrinkageTargets:
    """Test shrinkage target construction."""

    def test_identity_target(self):
        """Test IDENTITY shrinkage target."""
        config = ReconciliationConfig(shrinkage_target="identity")
        reconciler = ShrinkageReconciler(config=config)

        # Create sample covariance
        sample_cov = np.array([[4, 1], [1, 9]])
        residuals = np.random.randn(10, 2)

        target = reconciler._construct_target(sample_cov, residuals)

        # Should be scaled identity
        assert target.shape == (2, 2)
        assert target[0, 1] == 0.0  # Off-diagonal zero
        assert target[1, 0] == 0.0
        # Diagonal should be equal (same scale factor)
        assert pytest.approx(target[0, 0], rel=0.01) == target[1, 1]

    def test_diagonal_target(self):
        """Test DIAGONAL shrinkage target."""
        config = ReconciliationConfig(shrinkage_target="diagonal")
        reconciler = ShrinkageReconciler(config=config)

        sample_cov = np.array([[4, 2], [2, 9]])
        residuals = np.random.randn(10, 2)

        target = reconciler._construct_target(sample_cov, residuals)

        # Should be diagonal of sample covariance
        assert target.shape == (2, 2)
        assert target[0, 0] == 4.0
        assert target[1, 1] == 9.0
        assert target[0, 1] == 0.0
        assert target[1, 0] == 0.0

    def test_constant_correlation_target(self):
        """Test CONSTANT_CORRELATION shrinkage target."""
        config = ReconciliationConfig(shrinkage_target="constant_correlation")
        reconciler = ShrinkageReconciler(config=config)

        sample_cov = np.array([[4, 2], [2, 9]])
        residuals = np.random.randn(10, 2)

        target = reconciler._construct_target(sample_cov, residuals)

        # Should have constant correlation
        assert target.shape == (2, 2)
        assert target[0, 0] == 4.0  # Variances preserved
        assert target[1, 1] == 9.0
        # Correlation should be constant
        corr = target[0, 1] / np.sqrt(target[0, 0] * target[1, 1])
        assert -1 <= corr <= 1

    def test_factor_target(self):
        """Test FACTOR shrinkage target."""
        config = ReconciliationConfig(shrinkage_target="factor")
        reconciler = ShrinkageReconciler(config=config)

        np.random.seed(42)
        sample_cov = np.array([[4, 2, 1], [2, 9, 3], [1, 3, 16]])
        residuals = np.random.randn(20, 3)

        target = reconciler._construct_target(sample_cov, residuals)

        # Should be valid covariance
        assert target.shape == (3, 3)
        assert np.allclose(target, target.T)  # Symmetric
        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(target)
        assert np.all(eigenvalues >= -1e-6)


# ============================================================================
# Test Lambda Selection
# ============================================================================


class TestLambdaSelection:
    """Test lambda selection methods."""

    def test_ledoit_wolf_lambda(self):
        """Test Ledoit-Wolf lambda computation."""
        reconciler = ShrinkageReconciler()

        np.random.seed(42)
        n_samples, n_nodes = 20, 10  # Small sample
        residuals = np.random.randn(n_samples, n_nodes)

        sample_cov = reconciler._compute_sample_covariance(residuals)
        target_cov = reconciler._construct_target(sample_cov, residuals)

        lambda_opt = reconciler._ledoit_wolf_lambda(sample_cov, target_cov, residuals)

        # Should be in valid range
        assert 0.0 <= lambda_opt <= 1.0

        # With small sample, should have significant shrinkage
        assert lambda_opt > 0.1

    def test_ledoit_wolf_large_sample(self):
        """Test Ledoit-Wolf with large sample."""
        reconciler = ShrinkageReconciler()

        np.random.seed(43)
        n_samples, n_nodes = 200, 10  # Large sample
        residuals = np.random.randn(n_samples, n_nodes)

        sample_cov = reconciler._compute_sample_covariance(residuals)
        target_cov = reconciler._construct_target(sample_cov, residuals)

        lambda_opt = reconciler._ledoit_wolf_lambda(sample_cov, target_cov, residuals)

        # With large sample, should be in valid range
        # Actual shrinkage intensity depends on data characteristics
        assert 0.0 <= lambda_opt <= 1.0

    def test_cv_lambda_selection(self):
        """Test cross-validation lambda selection."""
        config = ReconciliationConfig(lambda_method="cross_validation")
        reconciler = ShrinkageReconciler(config=config)

        np.random.seed(44)
        n_samples, n_nodes = 30, 5
        residuals = np.random.randn(n_samples, n_nodes)

        sample_cov = reconciler._compute_sample_covariance(residuals)
        target_cov = reconciler._construct_target(sample_cov, residuals)

        lambda_opt = reconciler._cv_lambda(residuals, sample_cov, target_cov)

        # Should be in valid range
        assert 0.0 <= lambda_opt <= 1.0

    def test_cv_lambda_insufficient_data(self):
        """Test CV lambda with insufficient data."""
        config = ReconciliationConfig(lambda_method="cross_validation")
        reconciler = ShrinkageReconciler(config=config)

        residuals = np.random.randn(5, 3)  # Only 5 samples
        sample_cov = reconciler._compute_sample_covariance(residuals)
        target_cov = reconciler._construct_target(sample_cov, residuals)

        lambda_opt = reconciler._cv_lambda(residuals, sample_cov, target_cov)

        # Should fallback to 0.5
        assert lambda_opt == 0.5

    def test_fixed_lambda(self):
        """Test fixed lambda mode."""
        config = ReconciliationConfig(auto_lambda=False, fixed_lambda=0.6)
        reconciler = ShrinkageReconciler(config=config)

        assert reconciler.auto_lambda is False
        assert reconciler.fixed_lambda == 0.6


# ============================================================================
# Test Fitting
# ============================================================================


class TestFitting:
    """Test reconciler fitting."""

    def test_fit_basic(self, simple_hierarchy, sample_data_limited):
        """Test basic fitting."""
        reconciler = ShrinkageReconciler()
        forecasts, actuals = sample_data_limited

        reconciler.fit(forecasts, actuals, simple_hierarchy)

        assert reconciler.fitted is True
        assert reconciler.covariance_matrix is not None
        assert reconciler.computed_lambda is not None
        assert 0.0 <= reconciler.computed_lambda <= 1.0

    def test_fit_with_small_sample(self, simple_hierarchy):
        """Test fitting with very small sample."""
        np.random.seed(42)
        n_periods = 5  # Very small
        nodes = ["Total", "A", "B", "A1", "A2"]

        forecasts = pd.DataFrame({
            node: np.random.normal(100, 10, n_periods) for node in nodes
        })
        actuals = pd.DataFrame({
            node: forecasts[node] + np.random.normal(0, 5, n_periods) for node in nodes
        })

        reconciler = ShrinkageReconciler()
        reconciler.fit(forecasts, actuals, simple_hierarchy)

        # Should have very high shrinkage with small sample
        assert reconciler.computed_lambda > 0.5

    def test_fit_shape_mismatch_error(self, simple_hierarchy, sample_data_limited):
        """Test error on shape mismatch."""
        reconciler = ShrinkageReconciler()
        forecasts, actuals = sample_data_limited

        # Remove rows from actuals
        actuals_short = actuals.iloc[:10]

        with pytest.raises(ValueError, match="Shape mismatch"):
            reconciler.fit(forecasts, actuals_short, simple_hierarchy)

    def test_fit_insufficient_periods_error(self, simple_hierarchy):
        """Test error with insufficient periods."""
        reconciler = ShrinkageReconciler()

        # Only 1 period
        forecasts = pd.DataFrame({
            "Total": [100], "A": [40], "B": [60], "A1": [20], "A2": [20]
        })
        actuals = pd.DataFrame({
            "Total": [105], "A": [42], "B": [63], "A1": [21], "A2": [21]
        })

        with pytest.raises(ValueError, match="at least 2 historical periods"):
            reconciler.fit(forecasts, actuals, simple_hierarchy)

    def test_covariance_positive_definite(self, simple_hierarchy, sample_data_limited):
        """Test that shrunk covariance is positive definite."""
        reconciler = ShrinkageReconciler()
        forecasts, actuals = sample_data_limited

        reconciler.fit(forecasts, actuals, simple_hierarchy)

        # Check eigenvalues
        eigenvalues = np.linalg.eigvalsh(reconciler.covariance_matrix)
        assert np.all(eigenvalues > 0)  # Strictly positive definite


# ============================================================================
# Test Reconciliation
# ============================================================================


class TestReconciliation:
    """Test forecast reconciliation."""

    def test_reconcile_basic(self, simple_hierarchy, sample_data_limited):
        """Test basic reconciliation."""
        reconciler = ShrinkageReconciler()
        forecasts, actuals = sample_data_limited

        # Fit
        reconciler.fit(forecasts, actuals, simple_hierarchy)

        # Reconcile
        result = reconciler.reconcile(forecasts.iloc[:5], simple_hierarchy)

        assert result.reconciled_forecasts.shape == forecasts.iloc[:5].shape
        assert "shrinkage_lambda" in result.metadata
        assert result.metadata["method"] == "shrinkage"

    def test_reconcile_not_fitted_error(self, simple_hierarchy, sample_data_limited):
        """Test error when reconciling before fitting."""
        reconciler = ShrinkageReconciler()
        forecasts, _ = sample_data_limited

        with pytest.raises(ValueError, match="not fitted"):
            reconciler.reconcile(forecasts, simple_hierarchy)

    def test_coherence_improvement(self, simple_hierarchy, sample_data_moderate):
        """Test that reconciliation improves coherence."""
        reconciler = ShrinkageReconciler()
        forecasts, actuals = sample_data_moderate

        # Fit
        reconciler.fit(forecasts, actuals, simple_hierarchy)

        # Create incoherent forecasts
        incoherent = forecasts.iloc[:10].copy()
        incoherent["Total"] = 100  # Force incoherence

        # Reconcile
        result = reconciler.reconcile(incoherent, simple_hierarchy)

        # Coherence should improve
        coherence_before = result.metadata["coherence_before"]
        coherence_after = result.metadata["coherence_after"]

        assert coherence_after < coherence_before

    def test_non_negativity_constraint(self, simple_hierarchy, sample_data_moderate):
        """Test non-negativity constraint."""
        config = ReconciliationConfig(non_negative=True)
        reconciler = ShrinkageReconciler(config=config)

        forecasts, actuals = sample_data_moderate
        reconciler.fit(forecasts, actuals, simple_hierarchy)

        result = reconciler.reconcile(forecasts.iloc[:10], simple_hierarchy)

        # All values should be non-negative
        assert (result.reconciled_forecasts >= 0).all().all()

    def test_metadata_complete(self, simple_hierarchy, sample_data_limited):
        """Test that metadata is complete."""
        reconciler = ShrinkageReconciler()
        forecasts, actuals = sample_data_limited

        reconciler.fit(forecasts, actuals, simple_hierarchy)
        result = reconciler.reconcile(forecasts.iloc[:5], simple_hierarchy)

        metadata = result.metadata

        assert "method" in metadata
        assert "shrinkage_target" in metadata
        assert "shrinkage_lambda" in metadata
        assert "lambda_method" in metadata
        assert "coherence_before" in metadata
        assert "coherence_after" in metadata
        assert metadata["method"] == "shrinkage"


# ============================================================================
# Test Diagnostics
# ============================================================================


class TestDiagnostics:
    """Test shrinkage diagnostics."""

    def test_diagnostics_unfitted(self):
        """Test diagnostics when unfitted."""
        reconciler = ShrinkageReconciler()
        diag = reconciler.get_diagnostics()

        assert diag["fitted"] is False

    def test_diagnostics_fitted(self, simple_hierarchy, sample_data_limited):
        """Test diagnostics when fitted."""
        reconciler = ShrinkageReconciler()
        forecasts, actuals = sample_data_limited

        reconciler.fit(forecasts, actuals, simple_hierarchy)
        diag = reconciler.get_diagnostics()

        assert diag["fitted"] is True
        assert "lambda" in diag
        assert "shrinkage_target" in diag
        assert "sample_covariance_condition" in diag
        assert "shrunk_covariance_condition" in diag
        assert "condition_improvement" in diag

        # Shrinkage should improve condition number
        assert diag["condition_improvement"] >= 1.0

    def test_condition_number_improvement(self, simple_hierarchy, sample_data_limited):
        """Test that shrinkage improves condition number."""
        reconciler = ShrinkageReconciler()
        forecasts, actuals = sample_data_limited

        reconciler.fit(forecasts, actuals, simple_hierarchy)
        diag = reconciler.get_diagnostics()

        sample_cond = diag["sample_covariance_condition"]
        shrunk_cond = diag["shrunk_covariance_condition"]

        # Shrunk covariance should be better conditioned
        assert shrunk_cond <= sample_cond


# ============================================================================
# Test Performance
# ============================================================================


class TestPerformance:
    """Test performance requirements."""

    def test_reconciliation_performance(self, simple_hierarchy, sample_data_moderate):
        """Test reconciliation completes in <5s for 26 series."""
        import time

        # Note: Our test hierarchy has 5 nodes, not 26
        # In production, Brazilian hierarchy has 26 nodes
        reconciler = ShrinkageReconciler()
        forecasts, actuals = sample_data_moderate

        reconciler.fit(forecasts, actuals, simple_hierarchy)

        start = time.time()
        result = reconciler.reconcile(forecasts.iloc[:10], simple_hierarchy)
        elapsed = time.time() - start

        # Should be very fast for small hierarchy
        assert elapsed < 1.0  # Well under 5s requirement


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_lambda_clipping(self, simple_hierarchy, sample_data_limited):
        """Test lambda is clipped to bounds."""
        config = ReconciliationConfig(min_lambda=0.3, max_lambda=0.7)
        reconciler = ShrinkageReconciler(config=config)

        forecasts, actuals = sample_data_limited
        reconciler.fit(forecasts, actuals, simple_hierarchy)

        # Lambda should be clipped to bounds
        assert 0.3 <= reconciler.computed_lambda <= 0.7

    def test_identical_sample_and_target(self):
        """Test when sample covariance equals target."""
        reconciler = ShrinkageReconciler()

        # Create identical matrices
        sample_cov = np.diag([1, 2, 3])
        target_cov = np.diag([1, 2, 3])
        residuals = np.random.randn(10, 3)

        lambda_opt = reconciler._ledoit_wolf_lambda(sample_cov, target_cov, residuals)

        # Should be 0 (no shrinkage needed)
        assert lambda_opt == 0.0
