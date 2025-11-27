# ruff: noqa: NPY002
"""Tests for WLS (Weighted Least Squares) reconciler.

Tests cover:
- Weight calculation with different schemes
- WLS reconciliation matrix computation
- Fit method for learning variances
- Fallback mechanisms
- Weight diagnostics
- Save/load persistence
- Edge cases
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reconciliation.data_structures import ReconciliationConfig
from src.reconciliation.hierarchy import HierarchyDefinition
from src.reconciliation.wls_reconciler import (
    VarianceWeightCalculator,
    WeightDiagnostics,
    WeightingScheme,
    WLSConfig,
    WLSReconciler,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_hierarchy(tmp_path):
    """Create a sample hierarchy for testing."""
    yaml_content = """
hierarchy:
  national:
    name: "Total"
    children: ["North", "South"]

  subsystems:
    North:
      name: "North"
      children: ["City_A", "City_B"]
    South:
      name: "South"
      children: ["City_C", "City_D"]

  areas:
    City_A:
      name: "City_A"
      type: "consumption"
    City_B:
      name: "City_B"
      type: "consumption"
    City_C:
      name: "City_C"
      type: "consumption"
    City_D:
      name: "City_D"
      type: "consumption"
"""
    yaml_file = tmp_path / "test_hierarchy.yaml"
    yaml_file.write_text(yaml_content)
    return HierarchyDefinition(str(yaml_file))


@pytest.fixture
def sample_forecasts(sample_hierarchy):
    """Create sample forecasts matching the hierarchy."""
    np.random.seed(42)
    n_timesteps = 10
    nodes = sample_hierarchy.get_node_names_sorted()

    # Create forecasts for bottom nodes
    city_a = 100 + np.random.randn(n_timesteps) * 10
    city_b = 150 + np.random.randn(n_timesteps) * 15
    city_c = 80 + np.random.randn(n_timesteps) * 8
    city_d = 120 + np.random.randn(n_timesteps) * 12

    # Aggregate (making them slightly incoherent)
    north = city_a + city_b + np.random.randn(n_timesteps) * 5
    south = city_c + city_d + np.random.randn(n_timesteps) * 5
    total = north + south + np.random.randn(n_timesteps) * 10

    data = {
        "Total": total,
        "North": north,
        "South": south,
        "City_A": city_a,
        "City_B": city_b,
        "City_C": city_c,
        "City_D": city_d,
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_variances():
    """Create sample variance estimates."""
    return {
        "Total": np.array([100.0]),
        "North": np.array([50.0]),
        "South": np.array([40.0]),
        "City_A": np.array([20.0]),
        "City_B": np.array([30.0]),
        "City_C": np.array([15.0]),
        "City_D": np.array([25.0]),
    }


@pytest.fixture
def sample_actuals(sample_forecasts):
    """Create sample actual values (slightly different from forecasts)."""
    np.random.seed(43)
    actuals = sample_forecasts.copy()
    for col in actuals.columns:
        actuals[col] += np.random.randn(len(actuals)) * 5
    return actuals


# =============================================================================
# WeightingScheme Tests
# =============================================================================


class TestWeightingScheme:
    """Tests for WeightingScheme enum."""

    def test_all_schemes_defined(self):
        """Test that all expected schemes are defined."""
        assert WeightingScheme.INVERSE_VARIANCE.value == "inverse_variance"
        assert WeightingScheme.PREDICTION_INTERVAL.value == "prediction_interval"
        assert WeightingScheme.MODEL_CONFIDENCE.value == "model_confidence"
        assert WeightingScheme.ENSEMBLE_VARIANCE.value == "ensemble_variance"
        assert WeightingScheme.EQUAL.value == "equal"

    def test_scheme_from_string(self):
        """Test creating scheme from string."""
        scheme = WeightingScheme("inverse_variance")
        assert scheme == WeightingScheme.INVERSE_VARIANCE


# =============================================================================
# VarianceWeightCalculator Tests
# =============================================================================


class TestVarianceWeightCalculator:
    """Tests for VarianceWeightCalculator class."""

    def test_inverse_variance_weights(self, sample_variances):
        """Test inverse variance weight calculation."""
        calculator = VarianceWeightCalculator(scheme=WeightingScheme.INVERSE_VARIANCE)
        node_order = list(sample_variances.keys())

        weights = calculator.calculate_weights(
            variances=sample_variances,
            node_order=node_order,
        )

        assert len(weights) == len(node_order)
        assert all(w > 0 for w in weights)

        # Lower variance should mean higher weight
        # City_C has variance 15, City_A has variance 20
        city_c_idx = node_order.index("City_C")
        city_a_idx = node_order.index("City_A")
        assert weights[city_c_idx] > weights[city_a_idx]

    def test_equal_weights(self):
        """Test equal weight calculation."""
        calculator = VarianceWeightCalculator(scheme=WeightingScheme.EQUAL)
        node_order = ["A", "B", "C"]

        weights = calculator.calculate_weights(node_order=node_order)

        assert len(weights) == 3
        assert all(w == 1.0 for w in weights)

    def test_prediction_interval_weights(self):
        """Test prediction interval weight calculation."""
        calculator = VarianceWeightCalculator(scheme=WeightingScheme.PREDICTION_INTERVAL)

        prediction_intervals = {
            "A": (np.array([90, 95]), np.array([110, 105])),  # Width ~15
            "B": (np.array([80, 85]), np.array([120, 115])),  # Width ~35
        }
        node_order = ["A", "B"]

        weights = calculator.calculate_weights(
            prediction_intervals=prediction_intervals,
            node_order=node_order,
        )

        # Narrower interval should have higher weight
        assert weights[0] > weights[1]

    def test_ensemble_variance_weights(self):
        """Test ensemble variance weight calculation."""
        calculator = VarianceWeightCalculator(scheme=WeightingScheme.ENSEMBLE_VARIANCE)

        ensemble_forecasts = {
            "A": [np.array([100, 101, 99]), np.array([101, 100, 100])],  # Low variance
            "B": [np.array([100, 110, 90]), np.array([105, 95, 100])],  # High variance
        }
        node_order = ["A", "B"]

        weights = calculator.calculate_weights(
            ensemble_forecasts=ensemble_forecasts,
            node_order=node_order,
        )

        # Lower ensemble variance should have higher weight
        assert weights[0] > weights[1]

    def test_confidence_weights(self):
        """Test model confidence weight calculation."""
        calculator = VarianceWeightCalculator(scheme=WeightingScheme.MODEL_CONFIDENCE)

        confidence_scores = {"A": 0.9, "B": 0.5, "C": 0.7}
        node_order = ["A", "B", "C"]

        weights = calculator.calculate_weights(
            confidence_scores=confidence_scores,
            node_order=node_order,
        )

        # Higher confidence should have higher weight
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]

    def test_weight_normalization(self, sample_variances):
        """Test weight normalization."""
        calculator = VarianceWeightCalculator(
            scheme=WeightingScheme.INVERSE_VARIANCE,
            normalize=True,
        )
        node_order = list(sample_variances.keys())

        weights = calculator.calculate_weights(
            variances=sample_variances,
            node_order=node_order,
        )

        assert abs(sum(weights) - 1.0) < 1e-10

    def test_weight_clipping(self):
        """Test weight clipping to bounds."""
        calculator = VarianceWeightCalculator(
            scheme=WeightingScheme.INVERSE_VARIANCE,
            min_weight=0.1,
            max_weight=10.0,
        )

        # Create extreme variances
        variances = {
            "A": np.array([1e-20]),  # Should clip to max_weight
            "B": np.array([1e20]),  # Should clip to min_weight
        }
        node_order = ["A", "B"]

        weights = calculator.calculate_weights(
            variances=variances,
            node_order=node_order,
        )

        assert weights[0] == 10.0  # Clipped to max
        assert weights[1] == 0.1  # Clipped to min

    def test_missing_variance_fallback(self):
        """Test fallback for missing variance."""
        calculator = VarianceWeightCalculator(scheme=WeightingScheme.INVERSE_VARIANCE)

        variances = {"A": np.array([50.0])}  # Missing B
        node_order = ["A", "B"]

        weights = calculator.calculate_weights(
            variances=variances,
            node_order=node_order,
        )

        # B should get fallback weight of 1.0
        assert weights[1] == 1.0

    def test_missing_required_data_raises(self):
        """Test that missing required data raises error."""
        calculator = VarianceWeightCalculator(scheme=WeightingScheme.INVERSE_VARIANCE)

        with pytest.raises(ValueError, match="Variances required"):
            calculator.calculate_weights(node_order=["A", "B"])

    def test_handle_nan_weights(self):
        """Test handling of NaN weights."""
        calculator = VarianceWeightCalculator(scheme=WeightingScheme.INVERSE_VARIANCE)

        variances = {
            "A": np.array([np.nan]),
            "B": np.array([50.0]),
        }
        node_order = ["A", "B"]

        weights = calculator.calculate_weights(
            variances=variances,
            node_order=node_order,
        )

        # NaN should be replaced with 1.0
        assert np.isfinite(weights[0])


# =============================================================================
# WLSReconciler Tests
# =============================================================================


class TestWLSReconciler:
    """Tests for WLSReconciler class."""

    def test_initialization(self):
        """Test reconciler initialization."""
        reconciler = WLSReconciler()

        assert reconciler.weighting_scheme == WeightingScheme.INVERSE_VARIANCE
        assert not reconciler.fitted

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = ReconciliationConfig(
            method="wls",
            weighting_scheme="prediction_interval",
            normalize_weights=True,
            regularization=1e-4,
        )
        reconciler = WLSReconciler(config=config)

        assert reconciler.weighting_scheme == WeightingScheme.PREDICTION_INTERVAL
        assert reconciler.normalize_weights is True
        assert reconciler.regularization == 1e-4

    def test_fit(self, sample_hierarchy, sample_forecasts, sample_actuals):
        """Test fitting reconciler to historical data."""
        reconciler = WLSReconciler()

        reconciler.fit(sample_forecasts, sample_actuals, sample_hierarchy)

        assert reconciler.fitted
        assert reconciler._learned_variances is not None
        assert len(reconciler._learned_variances) == sample_hierarchy.n_nodes

    def test_reconcile_with_provided_variances(
        self, sample_hierarchy, sample_forecasts, sample_variances
    ):
        """Test reconciliation with provided variances."""
        reconciler = WLSReconciler()

        result = reconciler.reconcile(
            sample_forecasts,
            sample_hierarchy,
            forecast_variances=sample_variances,
        )

        assert result.reconciled_forecasts is not None
        assert result.reconciled_forecasts.shape == sample_forecasts.shape
        assert "weighting_scheme" in result.metadata
        assert result.metadata["weighting_scheme"] == "inverse_variance"

    def test_reconcile_with_fitted_variances(
        self, sample_hierarchy, sample_forecasts, sample_actuals
    ):
        """Test reconciliation using learned variances."""
        reconciler = WLSReconciler()

        # Fit to learn variances
        reconciler.fit(sample_forecasts, sample_actuals, sample_hierarchy)

        # Reconcile without providing variances
        result = reconciler.reconcile(sample_forecasts, sample_hierarchy)

        assert result.reconciled_forecasts is not None
        assert reconciler._computed_weights is not None

    def test_reconcile_coherence_improvement(
        self, sample_hierarchy, sample_forecasts, sample_variances
    ):
        """Test that reconciliation improves coherence."""
        reconciler = WLSReconciler()

        # Check coherence before
        coherence_before = reconciler.check_coherence(sample_forecasts, sample_hierarchy)

        # Reconcile
        result = reconciler.reconcile(
            sample_forecasts,
            sample_hierarchy,
            forecast_variances=sample_variances,
        )

        # Check coherence after
        coherence_after = reconciler.check_coherence(
            result.reconciled_forecasts, sample_hierarchy
        )

        # Coherence should improve (error should decrease)
        assert coherence_after < coherence_before

    def test_fallback_equal_weights(self, sample_hierarchy, sample_forecasts):
        """Test fallback to equal weights when variances unavailable."""
        reconciler = WLSReconciler(
            config=ReconciliationConfig(
                method="wls",
                fallback_equal_weights=True,
            )
        )

        # Reconcile without providing variances and without fitting
        result = reconciler.reconcile(sample_forecasts, sample_hierarchy)

        assert result.reconciled_forecasts is not None
        assert result.metadata["fallback_used"] is True

    def test_no_fallback_raises(self, sample_hierarchy, sample_forecasts):
        """Test that error is raised when fallback is disabled."""
        reconciler = WLSReconciler(
            config=ReconciliationConfig(
                method="wls",
                fallback_equal_weights=False,
            )
        )

        with pytest.raises(ValueError, match="Weight calculation failed"):
            reconciler.reconcile(sample_forecasts, sample_hierarchy)

    def test_weight_diagnostics(self, sample_hierarchy, sample_forecasts, sample_variances):
        """Test weight diagnostics."""
        reconciler = WLSReconciler()

        # Reconcile to compute weights
        reconciler.reconcile(
            sample_forecasts,
            sample_hierarchy,
            forecast_variances=sample_variances,
        )

        diagnostics = reconciler.get_weight_diagnostics()

        assert isinstance(diagnostics, WeightDiagnostics)
        assert diagnostics.n_series == sample_hierarchy.n_nodes
        assert diagnostics.min_weight > 0
        assert diagnostics.max_weight > diagnostics.min_weight
        assert diagnostics.weight_ratio > 1

    def test_diagnostics_before_reconcile(self):
        """Test diagnostics before reconciliation."""
        reconciler = WLSReconciler()

        diagnostics = reconciler.get_weight_diagnostics()

        assert "error" in diagnostics

    def test_get_weights_by_node(self, sample_hierarchy, sample_forecasts, sample_variances):
        """Test getting weights mapped to node names."""
        reconciler = WLSReconciler()

        reconciler.reconcile(
            sample_forecasts,
            sample_hierarchy,
            forecast_variances=sample_variances,
        )

        weights_by_node = reconciler.get_weights_by_node()

        assert weights_by_node is not None
        assert len(weights_by_node) == sample_hierarchy.n_nodes
        assert all(isinstance(v, float) for v in weights_by_node.values())

    def test_non_negativity_constraint(self, sample_hierarchy):
        """Test non-negativity constraint application."""
        config = ReconciliationConfig(method="wls", non_negative=True)
        reconciler = WLSReconciler(config=config)

        # Create forecasts with some negative base forecasts
        forecasts = pd.DataFrame({
            "Total": [100, 90, 80],
            "North": [60, 50, 40],
            "South": [40, 40, 40],
            "City_A": [30, 25, 20],
            "City_B": [30, 25, 20],
            "City_C": [20, 20, 20],
            "City_D": [20, 20, 20],
        })

        variances = {node: np.array([10.0]) for node in forecasts.columns}

        result = reconciler.reconcile(
            forecasts,
            sample_hierarchy,
            forecast_variances=variances,
        )

        # All values should be non-negative
        assert (result.reconciled_forecasts >= 0).all().all()

    def test_different_weighting_schemes(self, sample_hierarchy, sample_forecasts):
        """Test reconciliation with different weighting schemes."""
        for scheme in ["inverse_variance", "equal"]:
            config = ReconciliationConfig(method="wls", weighting_scheme=scheme)
            reconciler = WLSReconciler(config=config)

            variances = {node: np.array([10.0]) for node in sample_forecasts.columns}

            result = reconciler.reconcile(
                sample_forecasts,
                sample_hierarchy,
                forecast_variances=variances,
            )

            assert result.reconciled_forecasts is not None
            assert result.metadata["weighting_scheme"] == scheme


# =============================================================================
# Save/Load Tests
# =============================================================================


class TestWLSSaveLoad:
    """Tests for save/load functionality."""

    def test_save_and_load_unfitted(self):
        """Test saving and loading unfitted reconciler."""
        config = ReconciliationConfig(
            method="wls",
            weighting_scheme="prediction_interval",
            normalize_weights=True,
        )
        reconciler = WLSReconciler(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "wls_reconciler.json"
            reconciler.save(filepath)

            loaded = WLSReconciler.load(filepath)

            assert loaded.weighting_scheme == WeightingScheme.PREDICTION_INTERVAL
            assert loaded.normalize_weights is True
            assert not loaded.fitted

    def test_save_and_load_fitted(
        self, sample_hierarchy, sample_forecasts, sample_actuals
    ):
        """Test saving and loading fitted reconciler."""
        reconciler = WLSReconciler()
        reconciler.fit(sample_forecasts, sample_actuals, sample_hierarchy)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "wls_reconciler.json"
            reconciler.save(filepath)

            loaded = WLSReconciler.load(filepath)

            assert loaded.fitted
            assert loaded._learned_variances is not None
            assert len(loaded._learned_variances) == sample_hierarchy.n_nodes

    def test_loaded_reconciler_can_reconcile(
        self, sample_hierarchy, sample_forecasts, sample_actuals
    ):
        """Test that loaded reconciler can perform reconciliation."""
        reconciler = WLSReconciler()
        reconciler.fit(sample_forecasts, sample_actuals, sample_hierarchy)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "wls_reconciler.json"
            reconciler.save(filepath)

            loaded = WLSReconciler.load(filepath)

            # Reconcile with loaded reconciler
            result = loaded.reconcile(sample_forecasts, sample_hierarchy)

            assert result.reconciled_forecasts is not None


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_timestep(self, sample_hierarchy, sample_variances):
        """Test reconciliation with single timestep."""
        forecasts = pd.DataFrame({
            "Total": [100.0],
            "North": [60.0],
            "South": [40.0],
            "City_A": [30.0],
            "City_B": [30.0],
            "City_C": [20.0],
            "City_D": [20.0],
        })

        reconciler = WLSReconciler()
        result = reconciler.reconcile(
            forecasts,
            sample_hierarchy,
            forecast_variances=sample_variances,
        )

        assert result.reconciled_forecasts is not None
        assert len(result.reconciled_forecasts) == 1

    def test_extreme_variance_differences(self, sample_hierarchy, sample_forecasts):
        """Test with extreme variance differences."""
        variances = {
            "Total": np.array([1e-10]),  # Very low variance
            "North": np.array([1e10]),  # Very high variance
            "South": np.array([1.0]),
            "City_A": np.array([1.0]),
            "City_B": np.array([1.0]),
            "City_C": np.array([1.0]),
            "City_D": np.array([1.0]),
        }

        reconciler = WLSReconciler()
        result = reconciler.reconcile(
            sample_forecasts,
            sample_hierarchy,
            forecast_variances=variances,
        )

        # Should still complete successfully
        assert result.reconciled_forecasts is not None

        # Diagnostics should show extreme ratio
        diagnostics = reconciler.get_weight_diagnostics()
        assert diagnostics.weight_ratio > 1000

    def test_all_equal_variances(self, sample_hierarchy, sample_forecasts):
        """Test with all equal variances (equivalent to OLS)."""
        variances = {node: np.array([100.0]) for node in sample_forecasts.columns}

        reconciler = WLSReconciler()
        result = reconciler.reconcile(
            sample_forecasts,
            sample_hierarchy,
            forecast_variances=variances,
        )

        assert result.reconciled_forecasts is not None

        # All weights should be equal
        weights_by_node = reconciler.get_weights_by_node()
        weights = list(weights_by_node.values())
        assert all(abs(w - weights[0]) < 1e-10 for w in weights)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for WLS reconciler."""

    def test_full_workflow(self, sample_hierarchy, sample_forecasts, sample_actuals):
        """Test complete workflow."""
        # Create reconciler
        config = ReconciliationConfig(
            method="wls",
            weighting_scheme="inverse_variance",
            normalize_weights=True,
            non_negative=True,
        )
        reconciler = WLSReconciler(config=config)

        # Fit to historical data
        reconciler.fit(sample_forecasts, sample_actuals, sample_hierarchy)
        assert reconciler.fitted

        # Reconcile new forecasts
        result = reconciler.reconcile(sample_forecasts, sample_hierarchy)
        assert result.reconciled_forecasts is not None

        # Check coherence
        coherence_after = reconciler.check_coherence(
            result.reconciled_forecasts, sample_hierarchy
        )
        assert coherence_after < 1e-6

        # Get diagnostics
        diagnostics = reconciler.get_weight_diagnostics()
        assert isinstance(diagnostics, WeightDiagnostics)

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "wls_test.json"
            reconciler.save(filepath)

            loaded = WLSReconciler.load(filepath)
            assert loaded.fitted

            # Reconcile with loaded
            result2 = loaded.reconcile(sample_forecasts, sample_hierarchy)
            assert result2.reconciled_forecasts is not None

    def test_compare_with_equal_weights(
        self, sample_hierarchy, sample_forecasts, sample_variances
    ):
        """Test that variance weighting differs from equal weighting."""
        # WLS with inverse variance
        wls_reconciler = WLSReconciler(
            config=ReconciliationConfig(
                method="wls",
                weighting_scheme="inverse_variance",
            )
        )
        wls_result = wls_reconciler.reconcile(
            sample_forecasts,
            sample_hierarchy,
            forecast_variances=sample_variances,
        )

        # WLS with equal weights
        equal_reconciler = WLSReconciler(
            config=ReconciliationConfig(
                method="wls",
                weighting_scheme="equal",
            )
        )
        equal_result = equal_reconciler.reconcile(
            sample_forecasts,
            sample_hierarchy,
            forecast_variances=sample_variances,
        )

        # Results should be different (unless variances are all equal)
        diff = (wls_result.reconciled_forecasts - equal_result.reconciled_forecasts).abs()
        assert diff.max().max() > 0.01  # Should have some difference
