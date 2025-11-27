"""Tests for base reconciler abstract class."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reconciliation import (
    BaseReconciler,
    ReconciliationConfig,
    ReconciliationResult,
)
from src.reconciliation.hierarchy import HierarchyDefinition


# Concrete test implementation


class ConcreteReconciler(BaseReconciler):
    """Concrete implementation for testing BaseReconciler.

    Implements simple OLS reconciliation for testing purposes.
    """

    def reconcile(
        self,
        base_forecasts: pd.DataFrame,
        hierarchy: HierarchyDefinition,
    ) -> ReconciliationResult:
        """Reconcile using OLS method."""
        # Validate
        self.validate_forecasts(base_forecasts, hierarchy)

        # Compute coherence before
        coherence_before = self.check_coherence(base_forecasts, hierarchy)

        # Compute reconciliation matrix
        G = self._compute_reconciliation_matrix(hierarchy)

        # Apply reconciliation
        all_nodes = hierarchy.get_node_names_sorted()
        y_base = base_forecasts[all_nodes].values.T  # (n_total, T)
        y_reconciled = (G @ y_base).T  # (T, n_total)

        # Create DataFrame
        reconciled_df = pd.DataFrame(
            y_reconciled,
            index=base_forecasts.index,
            columns=all_nodes,
        )

        # Apply non-negativity if configured
        if self.config.non_negative:
            reconciled_df = self.apply_non_negativity(reconciled_df)

        # Compute coherence after
        coherence_after = self.check_coherence(reconciled_df, hierarchy)

        # Create result
        metadata = {
            "method": "ols",
            "coherence_before": coherence_before,
            "coherence_after": coherence_after,
        }

        return ReconciliationResult(
            reconciled_forecasts=reconciled_df,
            metadata=metadata,
            base_forecasts=base_forecasts,
            reconciliation_matrix=G,
        )

    def _compute_reconciliation_matrix(
        self,
        hierarchy: HierarchyDefinition,
        **kwargs,
    ) -> np.ndarray:
        """Compute OLS reconciliation matrix: G = S(S'S)^{-1}S'."""
        S = hierarchy.aggregation_matrix

        # G = S @ (S.T @ S)^{-1} @ S.T
        S_T_S = S.T @ S
        S_T_S_inv = np.linalg.inv(S_T_S)
        G = S @ S_T_S_inv @ S.T

        return G


# Fixtures


@pytest.fixture
def config():
    """Create test configuration."""
    return ReconciliationConfig(
        method="ols",
        non_negative=True,
        check_coherence=True,
    )


@pytest.fixture
def reconciler(config):
    """Create concrete reconciler instance."""
    return ConcreteReconciler(config=config)


@pytest.fixture
def simple_hierarchy():
    """Create simple hierarchy for testing."""
    hierarchy = HierarchyDefinition()
    hierarchy.add_node("National", level=0, node_type="aggregate", children=["A", "B"])
    hierarchy.add_node("A", level=1, node_type="bottom", parent="National")
    hierarchy.add_node("B", level=1, node_type="bottom", parent="National")
    hierarchy.build_graph()
    hierarchy.build_aggregation_matrix()
    return hierarchy


@pytest.fixture
def sample_forecasts(simple_hierarchy):
    """Create sample incoherent forecasts."""
    return pd.DataFrame(
        {
            "National": [350.0, 450.0, 550.0],
            "A": [100.0, 150.0, 200.0],
            "B": [200.0, 250.0, 300.0],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="h"),
    )


@pytest.fixture
def coherent_forecasts(simple_hierarchy):
    """Create coherent forecasts."""
    return pd.DataFrame(
        {
            "National": [300.0, 400.0, 500.0],
            "A": [100.0, 150.0, 200.0],
            "B": [200.0, 250.0, 300.0],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="h"),
    )


@pytest.fixture
def brazil_hierarchy():
    """Load Brazil hierarchy."""
    return HierarchyDefinition("config/hierarchy_brazil.yaml")


# Tests for BaseReconciler initialization


class TestBaseReconcilerInit:
    """Tests for BaseReconciler initialization."""

    def test_init_with_config(self, config):
        """Test initialization with config."""
        reconciler = ConcreteReconciler(config=config)

        assert reconciler.config == config
        assert reconciler.fitted is False
        assert reconciler.logger is not None

    def test_init_without_config(self):
        """Test initialization without config uses defaults."""
        reconciler = ConcreteReconciler()

        assert reconciler.config is not None
        assert reconciler.config.method == "ols"
        assert reconciler.fitted is False

    def test_cannot_instantiate_abstract(self):
        """Test cannot instantiate abstract BaseReconciler."""
        with pytest.raises(TypeError, match="abstract"):
            BaseReconciler()


# Tests for validate_forecasts


class TestValidateForecasts:
    """Tests for validate_forecasts method."""

    def test_validate_valid_forecasts(self, reconciler, sample_forecasts, simple_hierarchy):
        """Test validation passes for valid forecasts."""
        # Should not raise
        reconciler.validate_forecasts(sample_forecasts, simple_hierarchy)

    def test_validate_empty_raises(self, reconciler, simple_hierarchy):
        """Test validation fails for empty DataFrame."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            reconciler.validate_forecasts(empty_df, simple_hierarchy)

    def test_validate_nan_raises(self, reconciler, sample_forecasts, simple_hierarchy):
        """Test validation fails for NaN values."""
        forecasts_with_nan = sample_forecasts.copy()
        forecasts_with_nan.loc[0, "A"] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            reconciler.validate_forecasts(forecasts_with_nan, simple_hierarchy)

    def test_validate_missing_nodes_raises(self, reconciler, simple_hierarchy):
        """Test validation fails for missing nodes."""
        incomplete_forecasts = pd.DataFrame(
            {
                "National": [300.0, 400.0],
                "A": [100.0, 150.0],
                # Missing B
            },
            index=[0, 1],
        )

        with pytest.raises(ValueError, match="Missing nodes"):
            reconciler.validate_forecasts(incomplete_forecasts, simple_hierarchy)

    def test_validate_warns_extra_nodes(self, reconciler, sample_forecasts, simple_hierarchy):
        """Test validation warns about extra nodes."""
        forecasts_with_extra = sample_forecasts.copy()
        forecasts_with_extra["Extra"] = [100.0, 200.0, 300.0]

        # Should not raise, but logs warning
        reconciler.validate_forecasts(forecasts_with_extra, simple_hierarchy)


# Tests for check_coherence


class TestCheckCoherence:
    """Tests for check_coherence method."""

    def test_check_coherent_forecasts(self, reconciler, coherent_forecasts, simple_hierarchy):
        """Test coherence check for perfectly coherent forecasts."""
        error = reconciler.check_coherence(coherent_forecasts, simple_hierarchy)

        assert error < 1e-10

    def test_check_incoherent_forecasts(self, reconciler, sample_forecasts, simple_hierarchy):
        """Test coherence check for incoherent forecasts."""
        error = reconciler.check_coherence(sample_forecasts, simple_hierarchy)

        # National should be 300, 400, 500 but is 350, 450, 550
        # Normalized error should be positive (indicating incoherence)
        assert error > 0.05  # Relaxed threshold for normalized error

    def test_check_no_aggregation_matrix_raises(self, reconciler, sample_forecasts):
        """Test coherence check fails without aggregation matrix."""
        hierarchy = HierarchyDefinition()
        hierarchy.add_node("National", level=0, children=["A", "B"])
        hierarchy.add_node("A", level=1, parent="National")
        hierarchy.add_node("B", level=1, parent="National")
        hierarchy.build_graph()
        # Don't build aggregation matrix

        with pytest.raises(ValueError, match="aggregation matrix"):
            reconciler.check_coherence(sample_forecasts, hierarchy)

    def test_check_missing_nodes_raises(self, reconciler, simple_hierarchy):
        """Test coherence check fails with missing nodes."""
        incomplete = pd.DataFrame(
            {
                "National": [300.0],
                # Missing A and B
            },
            index=[0],
        )

        with pytest.raises(ValueError, match="Missing nodes"):
            reconciler.check_coherence(incomplete, simple_hierarchy)


# Tests for extract_bottom_level


class TestExtractBottomLevel:
    """Tests for extract_bottom_level method."""

    def test_extract_bottom(self, reconciler, sample_forecasts, simple_hierarchy):
        """Test extracting bottom-level forecasts."""
        bottom = reconciler.extract_bottom_level(sample_forecasts, simple_hierarchy)

        assert list(bottom.columns) == ["A", "B"]
        assert len(bottom) == 3
        pd.testing.assert_series_equal(bottom["A"], sample_forecasts["A"])
        pd.testing.assert_series_equal(bottom["B"], sample_forecasts["B"])

    def test_extract_missing_bottom_raises(self, reconciler, simple_hierarchy):
        """Test extraction fails with missing bottom nodes."""
        incomplete = pd.DataFrame(
            {
                "National": [300.0],
                "A": [100.0],
                # Missing B
            },
            index=[0],
        )

        with pytest.raises(ValueError, match="Missing bottom-level"):
            reconciler.extract_bottom_level(incomplete, simple_hierarchy)

    def test_extract_returns_copy(self, reconciler, sample_forecasts, simple_hierarchy):
        """Test extraction returns a copy, not view."""
        bottom = reconciler.extract_bottom_level(sample_forecasts, simple_hierarchy)

        # Modify extracted using proper indexing
        first_idx = bottom.index[0]
        bottom.loc[first_idx, "A"] = 999.0

        # Original should be unchanged
        assert sample_forecasts.loc[first_idx, "A"] != 999.0


# Tests for aggregate_to_top


class TestAggregateToTop:
    """Tests for aggregate_to_top method."""

    def test_aggregate_bottom(self, reconciler, simple_hierarchy):
        """Test aggregating bottom-level to all levels."""
        bottom = pd.DataFrame(
            {
                "A": [100.0, 150.0, 200.0],
                "B": [200.0, 250.0, 300.0],
            },
            index=[0, 1, 2],
        )

        all_forecasts = reconciler.aggregate_to_top(bottom, simple_hierarchy)

        # Check shape
        assert all_forecasts.shape == (3, 3)
        assert set(all_forecasts.columns) == {"National", "A", "B"}

        # Check aggregation
        assert all_forecasts.loc[0, "National"] == 300.0  # 100 + 200
        assert all_forecasts.loc[1, "National"] == 400.0  # 150 + 250
        assert all_forecasts.loc[2, "National"] == 500.0  # 200 + 300

        # Check bottom level preserved
        pd.testing.assert_series_equal(all_forecasts["A"], bottom["A"])
        pd.testing.assert_series_equal(all_forecasts["B"], bottom["B"])

    def test_aggregate_missing_bottom_raises(self, reconciler, simple_hierarchy):
        """Test aggregation fails with missing bottom nodes."""
        incomplete = pd.DataFrame(
            {
                "A": [100.0, 150.0],
                # Missing B
            },
            index=[0, 1],
        )

        with pytest.raises(ValueError, match="Missing bottom-level"):
            reconciler.aggregate_to_top(incomplete, simple_hierarchy)

    def test_aggregate_no_aggregation_matrix_raises(self, reconciler):
        """Test aggregation fails without aggregation matrix."""
        hierarchy = HierarchyDefinition()
        hierarchy.add_node("National", level=0, children=["A", "B"])
        hierarchy.add_node("A", level=1, parent="National")
        hierarchy.add_node("B", level=1, parent="National")
        hierarchy.build_graph()
        # Don't build aggregation matrix

        bottom = pd.DataFrame({"A": [100.0], "B": [200.0]}, index=[0])

        with pytest.raises(ValueError, match="aggregation matrix"):
            reconciler.aggregate_to_top(bottom, hierarchy)


# Tests for apply_non_negativity


class TestApplyNonNegativity:
    """Tests for apply_non_negativity method."""

    def test_apply_non_negativity_enabled(self, config):
        """Test non-negativity constraint when enabled."""
        config.non_negative = True
        reconciler = ConcreteReconciler(config=config)

        forecasts = pd.DataFrame(
            {
                "A": [100.0, -50.0, 200.0],
                "B": [-10.0, 150.0, -5.0],
            },
            index=[0, 1, 2],
        )

        result = reconciler.apply_non_negativity(forecasts)

        # Negative values should be clipped to 0
        assert result.loc[1, "A"] == 0.0
        assert result.loc[0, "B"] == 0.0
        assert result.loc[2, "B"] == 0.0

        # Positive values unchanged
        assert result.loc[0, "A"] == 100.0
        assert result.loc[1, "B"] == 150.0

    def test_apply_non_negativity_disabled(self, config):
        """Test non-negativity not applied when disabled."""
        config.non_negative = False
        reconciler = ConcreteReconciler(config=config)

        forecasts = pd.DataFrame(
            {
                "A": [100.0, -50.0, 200.0],
                "B": [-10.0, 150.0, -5.0],
            },
            index=[0, 1, 2],
        )

        result = reconciler.apply_non_negativity(forecasts)

        # Should be unchanged
        pd.testing.assert_frame_equal(result, forecasts)


# Tests for fit method


class TestFit:
    """Tests for fit method."""

    def test_fit_default_implementation(self, reconciler, sample_forecasts, simple_hierarchy):
        """Test default fit implementation does nothing."""
        actuals = sample_forecasts * 0.95

        # Should not raise
        reconciler.fit(sample_forecasts, actuals, simple_hierarchy)

        # Should not be marked as fitted by default
        assert reconciler.fitted is False


# Tests for reconcile method


class TestReconcile:
    """Tests for reconcile method (using concrete implementation)."""

    def test_reconcile_incoherent_forecasts(self, reconciler, sample_forecasts, simple_hierarchy):
        """Test reconciliation of incoherent forecasts."""
        result = reconciler.reconcile(sample_forecasts, simple_hierarchy)

        # Check result structure
        assert isinstance(result, ReconciliationResult)
        assert result.reconciled_forecasts is not None
        assert result.base_forecasts is not None

        # Check metadata
        assert result.metadata["method"] == "ols"
        assert "coherence_before" in result.metadata
        assert "coherence_after" in result.metadata

        # Coherence should improve
        assert result.metadata["coherence_after"] < result.metadata["coherence_before"]

        # Result should be coherent
        assert result.validate_coherence(simple_hierarchy, tolerance=1e-6)

    def test_reconcile_coherent_forecasts(self, reconciler, coherent_forecasts, simple_hierarchy):
        """Test reconciliation of already coherent forecasts."""
        result = reconciler.reconcile(coherent_forecasts, simple_hierarchy)

        # Should remain coherent
        assert result.validate_coherence(simple_hierarchy, tolerance=1e-6)
        assert result.metadata["coherence_after"] < 1e-10

    def test_reconcile_applies_non_negativity(self, config, simple_hierarchy):
        """Test reconciliation applies non-negativity constraint."""
        config.non_negative = True
        reconciler = ConcreteReconciler(config=config)

        # Create forecasts that might reconcile to negative values
        forecasts = pd.DataFrame(
            {
                "National": [10.0, 20.0, 30.0],
                "A": [50.0, 60.0, 70.0],
                "B": [-30.0, -30.0, -30.0],
            },
            index=[0, 1, 2],
        )

        result = reconciler.reconcile(forecasts, simple_hierarchy)

        # No negative values in result
        assert (result.reconciled_forecasts >= 0).all().all()

    def test_reconcile_invalid_forecasts_raises(self, reconciler, simple_hierarchy):
        """Test reconciliation fails with invalid forecasts."""
        invalid = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            reconciler.reconcile(invalid, simple_hierarchy)


# Tests for save/load


class TestSaveLoad:
    """Tests for save/load methods."""

    def test_save_config(self, reconciler, tmp_path):
        """Test saving reconciler configuration."""
        config_path = tmp_path / "reconciler_config.json"

        reconciler.save(config_path)

        assert config_path.exists()

        # Load and check
        with open(config_path) as f:
            data = json.load(f)

        assert data["method"] == "ols"
        assert data["non_negative"] is True
        assert data["fitted"] is False
        assert data["reconciler_class"] == "ConcreteReconciler"

    def test_load_config(self, reconciler, tmp_path):
        """Test loading reconciler configuration."""
        config_path = tmp_path / "reconciler_config.json"

        # Save
        reconciler.save(config_path)

        # Load
        loaded = ConcreteReconciler.load(config_path)

        # Check config matches
        assert loaded.config.method == reconciler.config.method
        assert loaded.config.non_negative == reconciler.config.non_negative
        assert loaded.config.check_coherence == reconciler.config.check_coherence

    def test_save_creates_directory(self, reconciler, tmp_path):
        """Test save creates parent directories."""
        config_path = tmp_path / "subdir" / "configs" / "reconciler.json"

        reconciler.save(config_path)

        assert config_path.exists()
        assert config_path.parent.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError, match="not found"):
            ConcreteReconciler.load(tmp_path / "missing.json")

    def test_save_load_fitted_flag(self, reconciler, tmp_path):
        """Test save/load preserves fitted flag."""
        config_path = tmp_path / "reconciler.json"

        # Mark as fitted
        reconciler.fitted = True
        reconciler.save(config_path)

        # Load
        loaded = ConcreteReconciler.load(config_path)

        assert loaded.fitted is True


# Tests for __repr__


class TestRepr:
    """Tests for string representation."""

    def test_repr(self, reconciler):
        """Test string representation."""
        repr_str = repr(reconciler)

        assert "ConcreteReconciler" in repr_str
        assert "method='ols'" in repr_str
        assert "fitted=False" in repr_str

    def test_repr_fitted(self, reconciler):
        """Test repr with fitted flag."""
        reconciler.fitted = True

        repr_str = repr(reconciler)

        assert "fitted=True" in repr_str


# Integration tests


class TestIntegrationWithBrazilHierarchy:
    """Integration tests with Brazil hierarchy."""

    def test_reconcile_brazil_hierarchy(self, brazil_hierarchy):
        """Test reconciliation with full Brazil hierarchy."""
        # Get nodes
        bottom_nodes = sorted(brazil_hierarchy.get_bottom_level_nodes())
        all_nodes = brazil_hierarchy.get_node_names_sorted()

        # Create incoherent forecasts from bottom-up then perturb top levels
        np.random.seed(42)
        n_timesteps = 10

        # Start with coherent forecasts
        y_bottom = np.random.rand(n_timesteps, len(bottom_nodes)) * 1000
        S = brazil_hierarchy.aggregation_matrix
        y_all = (S @ y_bottom.T).T

        # Perturb top-level forecasts to create incoherence
        forecasts = pd.DataFrame(y_all, columns=all_nodes)
        forecasts.iloc[:, :5] *= 1.1  # Perturb top-level nodes

        # Create reconciler
        config = ReconciliationConfig(method="ols", non_negative=True)
        reconciler = ConcreteReconciler(config=config)

        # Check coherence before
        error_before = reconciler.check_coherence(forecasts, brazil_hierarchy)
        assert error_before > 0  # Should be incoherent

        # Reconcile
        result = reconciler.reconcile(forecasts, brazil_hierarchy)

        # Check result
        assert result.reconciled_forecasts.shape == (n_timesteps, len(all_nodes))
        assert result.metadata["coherence_before"] > 0
        # OLS reconciliation should significantly reduce error
        assert result.metadata["coherence_after"] < error_before

        # Validate coherence (use slightly relaxed tolerance for numerical stability)
        assert result.validate_coherence(brazil_hierarchy, tolerance=1e-5)

    def test_extract_aggregate_brazil(self, brazil_hierarchy):
        """Test extract and aggregate with Brazil hierarchy."""
        # Get nodes
        bottom_nodes = sorted(brazil_hierarchy.get_bottom_level_nodes())
        all_nodes = brazil_hierarchy.get_node_names_sorted()

        # Create full forecasts
        np.random.seed(42)
        forecasts = pd.DataFrame(
            np.random.rand(5, len(all_nodes)) * 1000,
            columns=all_nodes,
        )

        # Create reconciler
        reconciler = ConcreteReconciler()

        # Extract bottom
        bottom = reconciler.extract_bottom_level(forecasts, brazil_hierarchy)
        assert bottom.shape == (5, 17)  # 17 areas

        # Aggregate back
        aggregated = reconciler.aggregate_to_top(bottom, brazil_hierarchy)
        assert aggregated.shape == (5, len(all_nodes))

        # Aggregated should be coherent
        error = reconciler.check_coherence(aggregated, brazil_hierarchy)
        assert error < 1e-10


class TestAbstractMethodEnforcement:
    """Test that abstract methods must be implemented."""

    def test_missing_reconcile_raises(self):
        """Test that missing reconcile method raises TypeError."""

        class IncompleteReconciler(BaseReconciler):
            def _compute_reconciliation_matrix(self, hierarchy, **kwargs):
                return np.eye(3)

        with pytest.raises(TypeError, match="abstract"):
            IncompleteReconciler()

    def test_missing_compute_matrix_raises(self):
        """Test that missing _compute_reconciliation_matrix raises TypeError."""

        class IncompleteReconciler(BaseReconciler):
            def reconcile(self, base_forecasts, hierarchy):
                return ReconciliationResult(reconciled_forecasts=base_forecasts)

        with pytest.raises(TypeError, match="abstract"):
            IncompleteReconciler()
