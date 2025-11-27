"""Tests for MinT (Minimum Trace) reconciler.

This module provides comprehensive tests for the MintReconciler class,
including covariance estimation, reconciliation accuracy, coherence validation,
and numerical stability.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reconciliation import (
    HierarchyDefinition,
    MintReconciler,
    ReconciliationConfig,
    ReconciliationResult,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config_sample():
    """Create test configuration with sample covariance."""
    return ReconciliationConfig(
        method="mint",
        non_negative=True,
        check_coherence=True,
        tolerance=1e-6,
        covariance_method="sample",
    )


@pytest.fixture
def config_shrinkage():
    """Create test configuration with shrinkage covariance."""
    return ReconciliationConfig(
        method="mint",
        non_negative=True,
        check_coherence=True,
        tolerance=1e-6,
        covariance_method="shrinkage",
    )


@pytest.fixture
def config_diagonal():
    """Create test configuration with diagonal covariance."""
    return ReconciliationConfig(
        method="mint",
        non_negative=True,
        check_coherence=True,
        tolerance=1e-6,
        covariance_method="diagonal",
    )


@pytest.fixture
def config_identity():
    """Create test configuration with identity covariance (OLS)."""
    return ReconciliationConfig(
        method="mint",
        non_negative=True,
        check_coherence=True,
        tolerance=1e-6,
        covariance_method="identity",
    )


@pytest.fixture
def config_structural():
    """Create test configuration with structural covariance."""
    return ReconciliationConfig(
        method="mint",
        non_negative=True,
        check_coherence=True,
        tolerance=1e-6,
        covariance_method="structural",
    )


@pytest.fixture
def simple_hierarchy():
    """Create simple 3-level hierarchy for testing.

    Structure:
        Total (root)
        ├── A (aggregate)
        │   ├── A1 (bottom)
        │   └── A2 (bottom)
        └── B (aggregate)
            ├── B1 (bottom)
            └── B2 (bottom)
    """
    hierarchy = HierarchyDefinition()

    # Root
    hierarchy.add_node("Total", level=0, node_type="aggregate")

    # Aggregates
    hierarchy.add_node("A", level=1, node_type="aggregate", parent="Total")
    hierarchy.add_node("B", level=1, node_type="aggregate", parent="Total")

    # Bottom level
    hierarchy.add_node("A1", level=2, node_type="bottom", parent="A")
    hierarchy.add_node("A2", level=2, node_type="bottom", parent="A")
    hierarchy.add_node("B1", level=2, node_type="bottom", parent="B")
    hierarchy.add_node("B2", level=2, node_type="bottom", parent="B")

    # Update children
    hierarchy.nodes["Total"].children = ["A", "B"]
    hierarchy.nodes["A"].children = ["A1", "A2"]
    hierarchy.nodes["B"].children = ["B1", "B2"]

    # Build graph and aggregation matrix
    hierarchy.build_graph()
    hierarchy.build_aggregation_matrix()

    return hierarchy


@pytest.fixture
def brazil_hierarchy():
    """Load Brazilian grid hierarchy from config file."""
    config_path = Path(__file__).parent.parent.parent / "config" / "hierarchy_brazil.yaml"
    if not config_path.exists():
        pytest.skip(f"Brazil hierarchy config not found: {config_path}")

    hierarchy = HierarchyDefinition(config_path)
    return hierarchy


@pytest.fixture
def simple_historical_data(simple_hierarchy):
    """Create historical forecasts and actuals for simple hierarchy."""
    np.random.seed(42)

    # Get nodes
    all_nodes = simple_hierarchy.get_node_names_sorted()

    # Generate 30 periods of historical data
    n_periods = 30

    # Generate bottom-level actuals
    bottom_actuals = {
        "A1": 100 + np.random.randn(n_periods) * 5,
        "A2": 80 + np.random.randn(n_periods) * 4,
        "B1": 120 + np.random.randn(n_periods) * 6,
        "B2": 90 + np.random.randn(n_periods) * 4.5,
    }

    # Aggregate actuals
    actuals_dict = {
        "A1": bottom_actuals["A1"],
        "A2": bottom_actuals["A2"],
        "B1": bottom_actuals["B1"],
        "B2": bottom_actuals["B2"],
        "A": bottom_actuals["A1"] + bottom_actuals["A2"],
        "B": bottom_actuals["B1"] + bottom_actuals["B2"],
        "Total": (
            bottom_actuals["A1"]
            + bottom_actuals["A2"]
            + bottom_actuals["B1"]
            + bottom_actuals["B2"]
        ),
    }

    actuals = pd.DataFrame(actuals_dict)[all_nodes]

    # Generate forecasts with errors
    forecasts_dict = {}
    for node in all_nodes:
        # Add forecast errors (heteroskedastic)
        if node == "Total":
            error_std = 10
        elif node in ["A", "B"]:
            error_std = 7
        else:  # Bottom level
            error_std = 5

        forecasts_dict[node] = actuals_dict[node] + np.random.randn(n_periods) * error_std

    forecasts = pd.DataFrame(forecasts_dict)[all_nodes]

    return forecasts, actuals


@pytest.fixture
def simple_base_forecasts(simple_hierarchy):
    """Create incoherent base forecasts for testing."""
    all_nodes = simple_hierarchy.get_node_names_sorted()

    # Create incoherent forecasts (violate aggregation constraints)
    data = {
        "Total": [400, 410],
        "A": [180, 185],
        "B": [210, 215],
        "A1": [100, 102],
        "A2": [82, 84],
        "B1": [118, 120],
        "B2": [92, 93],
    }

    return pd.DataFrame(data)[all_nodes]


# ============================================================================
# Initialization Tests
# ============================================================================


def test_mint_reconciler_init(config_sample):
    """Test MinT reconciler initialization."""
    reconciler = MintReconciler(config=config_sample)

    assert reconciler.config.method == "mint"
    assert reconciler.covariance_method == "sample"
    assert reconciler.covariance_matrix is None
    assert not reconciler.fitted
    assert reconciler.regularization > 0


def test_mint_reconciler_init_default_config():
    """Test MinT reconciler with default config."""
    reconciler = MintReconciler()

    assert reconciler.config.method == "ols"  # Default in ReconciliationConfig
    assert reconciler.covariance_method == "shrinkage"  # Default for MinT
    assert not reconciler.fitted


def test_mint_reconciler_init_custom_regularization():
    """Test MinT reconciler with custom regularization."""
    config = ReconciliationConfig(
        method="mint",
        regularization=1e-6,
    )
    reconciler = MintReconciler(config=config)

    assert reconciler.regularization == 1e-6


# ============================================================================
# Fit Method Tests
# ============================================================================


def test_fit_sample_covariance(config_sample, simple_hierarchy, simple_historical_data):
    """Test fit() with sample covariance estimation."""
    forecasts, actuals = simple_historical_data

    reconciler = MintReconciler(config=config_sample)
    reconciler.fit(forecasts, actuals, simple_hierarchy)

    assert reconciler.fitted
    assert reconciler.covariance_matrix is not None
    assert reconciler.covariance_matrix.shape == (7, 7)  # 7 nodes

    # Check positive definiteness
    eigenvalues = np.linalg.eigvals(reconciler.covariance_matrix)
    assert all(eigenvalues > 0), "Covariance matrix should be positive definite"


def test_fit_shrinkage_covariance(config_shrinkage, simple_hierarchy, simple_historical_data):
    """Test fit() with Ledoit-Wolf shrinkage estimation."""
    forecasts, actuals = simple_historical_data

    reconciler = MintReconciler(config=config_shrinkage)
    reconciler.fit(forecasts, actuals, simple_hierarchy)

    assert reconciler.fitted
    assert reconciler.covariance_matrix is not None

    # Shrinkage should produce positive definite matrix
    eigenvalues = np.linalg.eigvals(reconciler.covariance_matrix)
    assert all(eigenvalues > 0)


def test_fit_diagonal_covariance(config_diagonal, simple_hierarchy, simple_historical_data):
    """Test fit() with diagonal covariance estimation."""
    forecasts, actuals = simple_historical_data

    reconciler = MintReconciler(config=config_diagonal)
    reconciler.fit(forecasts, actuals, simple_hierarchy)

    assert reconciler.fitted
    assert reconciler.covariance_matrix is not None

    # Check that covariance is diagonal
    off_diagonal = reconciler.covariance_matrix - np.diag(np.diag(reconciler.covariance_matrix))
    assert np.allclose(off_diagonal, 0), "Diagonal covariance should have zero off-diagonal"


def test_fit_identity_covariance(config_identity, simple_hierarchy, simple_historical_data):
    """Test fit() with identity covariance (OLS fallback)."""
    forecasts, actuals = simple_historical_data

    reconciler = MintReconciler(config=config_identity)
    reconciler.fit(forecasts, actuals, simple_hierarchy)

    assert reconciler.fitted
    assert reconciler.covariance_matrix is not None

    # Check that covariance is identity
    assert np.allclose(reconciler.covariance_matrix, np.eye(7))


def test_fit_structural_covariance(config_structural, simple_hierarchy, simple_historical_data):
    """Test fit() with structural covariance estimation."""
    forecasts, actuals = simple_historical_data

    reconciler = MintReconciler(config=config_structural)
    reconciler.fit(forecasts, actuals, simple_hierarchy)

    assert reconciler.fitted
    assert reconciler.covariance_matrix is not None

    # Check that covariance is diagonal
    off_diagonal = reconciler.covariance_matrix - np.diag(np.diag(reconciler.covariance_matrix))
    assert np.allclose(off_diagonal, 0)


def test_fit_insufficient_data(config_sample, simple_hierarchy):
    """Test fit() with insufficient historical data."""
    # Only 1 period (need at least 2)
    forecasts = pd.DataFrame(
        {
            "Total": [400],
            "A": [180],
            "B": [210],
            "A1": [100],
            "A2": [80],
            "B1": [120],
            "B2": [90],
        }
    )
    actuals = forecasts.copy()

    reconciler = MintReconciler(config=config_sample)

    with pytest.raises(ValueError, match="at least 2 historical periods"):
        reconciler.fit(forecasts, actuals, simple_hierarchy)


def test_fit_shape_mismatch(config_sample, simple_hierarchy, simple_historical_data):
    """Test fit() with mismatched forecasts and actuals shapes."""
    forecasts, actuals = simple_historical_data

    # Drop one row from actuals
    actuals_short = actuals.iloc[:-1]

    reconciler = MintReconciler(config=config_sample)

    with pytest.raises(ValueError, match="Shape mismatch"):
        reconciler.fit(forecasts, actuals_short, simple_hierarchy)


def test_fit_missing_nodes(config_sample, simple_hierarchy, simple_historical_data):
    """Test fit() with missing nodes in data."""
    forecasts, actuals = simple_historical_data

    # Drop a column
    forecasts_incomplete = forecasts.drop(columns=["A1"])

    reconciler = MintReconciler(config=config_sample)

    with pytest.raises(ValueError, match="Missing nodes"):
        reconciler.fit(forecasts_incomplete, actuals, simple_hierarchy)


# ============================================================================
# Reconcile Method Tests
# ============================================================================


def test_reconcile_basic(
    config_shrinkage, simple_hierarchy, simple_historical_data, simple_base_forecasts
):
    """Test basic reconciliation workflow."""
    forecasts, actuals = simple_historical_data

    # Fit
    reconciler = MintReconciler(config=config_shrinkage)
    reconciler.fit(forecasts, actuals, simple_hierarchy)

    # Reconcile
    result = reconciler.reconcile(simple_base_forecasts, simple_hierarchy)

    assert isinstance(result, ReconciliationResult)
    assert result.reconciled_forecasts.shape == simple_base_forecasts.shape
    assert "coherence_before" in result.metadata
    assert "coherence_after" in result.metadata
    assert result.metadata["method"] == "mint"


def test_reconcile_coherence_improvement(
    config_shrinkage, simple_hierarchy, simple_historical_data, simple_base_forecasts
):
    """Test that reconciliation improves coherence."""
    forecasts, actuals = simple_historical_data

    reconciler = MintReconciler(config=config_shrinkage)
    reconciler.fit(forecasts, actuals, simple_hierarchy)

    result = reconciler.reconcile(simple_base_forecasts, simple_hierarchy)

    coherence_before = result.metadata["coherence_before"]
    coherence_after = result.metadata["coherence_after"]

    # Coherence should improve (decrease)
    assert coherence_after < coherence_before
    # After reconciliation, coherence should be near zero
    assert coherence_after < 1e-5


def test_reconcile_validates_coherence(
    config_shrinkage, simple_hierarchy, simple_historical_data, simple_base_forecasts
):
    """Test that reconciled forecasts satisfy coherence constraints."""
    forecasts, actuals = simple_historical_data

    reconciler = MintReconciler(config=config_shrinkage)
    reconciler.fit(forecasts, actuals, simple_hierarchy)

    result = reconciler.reconcile(simple_base_forecasts, simple_hierarchy)

    # Validate coherence using result method
    is_coherent = result.validate_coherence(simple_hierarchy, tolerance=1e-6)
    assert is_coherent


def test_reconcile_without_fit(config_shrinkage, simple_hierarchy, simple_base_forecasts):
    """Test reconciliation without fitting (should fallback to identity)."""
    reconciler = MintReconciler(config=config_shrinkage)

    # Don't call fit()
    result = reconciler.reconcile(simple_base_forecasts, simple_hierarchy)

    assert isinstance(result, ReconciliationResult)
    assert result.metadata["fitted"] is False


def test_reconcile_non_negativity(simple_hierarchy, simple_historical_data):
    """Test non-negativity constraint is applied."""
    forecasts, actuals = simple_historical_data

    # Create base forecasts with some negative values
    base_forecasts = pd.DataFrame(
        {
            "Total": [400, 410],
            "A": [180, 185],
            "B": [210, 215],
            "A1": [-5, 102],  # Negative value
            "A2": [82, 84],
            "B1": [118, 120],
            "B2": [92, 93],
        }
    )

    config = ReconciliationConfig(
        method="mint",
        non_negative=True,
        covariance_method="shrinkage",
    )

    reconciler = MintReconciler(config=config)
    reconciler.fit(forecasts, actuals, simple_hierarchy)

    result = reconciler.reconcile(base_forecasts, simple_hierarchy)

    # All values should be non-negative
    assert (result.reconciled_forecasts >= 0).all().all()


def test_reconcile_preserves_index(config_shrinkage, simple_hierarchy, simple_historical_data):
    """Test that reconciliation preserves DataFrame index."""
    forecasts, actuals = simple_historical_data

    # Create base forecasts with datetime index
    dates = pd.date_range("2025-01-01", periods=2, freq="D")
    all_nodes = simple_hierarchy.get_node_names_sorted()

    base_forecasts = pd.DataFrame(
        {
            "Total": [400, 410],
            "A": [180, 185],
            "B": [210, 215],
            "A1": [100, 102],
            "A2": [82, 84],
            "B1": [118, 120],
            "B2": [92, 93],
        },
        index=dates,
    )[all_nodes]

    reconciler = MintReconciler(config=config_shrinkage)
    reconciler.fit(forecasts, actuals, simple_hierarchy)

    result = reconciler.reconcile(base_forecasts, simple_hierarchy)

    # Check index is preserved
    pd.testing.assert_index_equal(result.reconciled_forecasts.index, base_forecasts.index)


# ============================================================================
# Covariance Estimation Tests
# ============================================================================


def test_estimate_covariance_sample(config_sample):
    """Test sample covariance estimation."""
    np.random.seed(42)
    residuals = np.random.randn(100, 5)

    reconciler = MintReconciler(config=config_sample)
    cov = reconciler._estimate_covariance(residuals, method="sample")

    assert cov.shape == (5, 5)
    # Sample covariance should be symmetric
    assert np.allclose(cov, cov.T)


def test_estimate_covariance_shrinkage(config_shrinkage):
    """Test shrinkage covariance estimation."""
    np.random.seed(42)
    residuals = np.random.randn(50, 5)

    reconciler = MintReconciler(config=config_shrinkage)
    cov = reconciler._estimate_covariance(residuals, method="shrinkage")

    assert cov.shape == (5, 5)
    assert np.allclose(cov, cov.T)


def test_estimate_covariance_diagonal(config_diagonal):
    """Test diagonal covariance estimation."""
    np.random.seed(42)
    residuals = np.random.randn(100, 5)

    reconciler = MintReconciler(config=config_diagonal)
    cov = reconciler._estimate_covariance(residuals, method="diagonal")

    assert cov.shape == (5, 5)

    # Check diagonal structure
    off_diagonal = cov - np.diag(np.diag(cov))
    assert np.allclose(off_diagonal, 0)


def test_estimate_covariance_identity(config_identity):
    """Test identity covariance estimation."""
    np.random.seed(42)
    residuals = np.random.randn(100, 5)

    reconciler = MintReconciler(config=config_identity)
    cov = reconciler._estimate_covariance(residuals, method="identity")

    assert cov.shape == (5, 5)
    assert np.allclose(cov, np.eye(5))


def test_estimate_covariance_structural(config_structural):
    """Test structural covariance estimation."""
    np.random.seed(42)
    residuals = np.random.randn(100, 5)

    reconciler = MintReconciler(config=config_structural)
    cov = reconciler._estimate_covariance(residuals, method="structural")

    assert cov.shape == (5, 5)

    # Check diagonal structure
    off_diagonal = cov - np.diag(np.diag(cov))
    assert np.allclose(off_diagonal, 0)


def test_estimate_covariance_unknown_method(config_sample):
    """Test error handling for unknown covariance method."""
    residuals = np.random.randn(100, 5)

    reconciler = MintReconciler(config=config_sample)

    with pytest.raises(ValueError, match="Unknown covariance method"):
        reconciler._estimate_covariance(residuals, method="unknown")


# ============================================================================
# Positive Definiteness Tests
# ============================================================================


def test_ensure_positive_definite_already_pd(config_sample):
    """Test _ensure_positive_definite with already PD matrix."""
    # Create positive definite matrix
    A = np.random.randn(5, 5)
    cov = A.T @ A + np.eye(5) * 0.1

    reconciler = MintReconciler(config=config_sample)
    cov_pd = reconciler._ensure_positive_definite(cov)

    # Should be unchanged (or minimally changed)
    assert np.allclose(cov_pd, cov, atol=1e-6)


def test_ensure_positive_definite_negative_eigenvalue(config_sample):
    """Test _ensure_positive_definite with negative eigenvalues."""
    # Create matrix with small negative eigenvalue (more realistic)
    # Start with a correlation matrix
    cov = np.array(
        [
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ]
    )
    # Make it slightly non-positive definite
    cov[0, 0] = 0.5  # This will create a small negative eigenvalue

    reconciler = MintReconciler(config=config_sample)
    cov_pd = reconciler._ensure_positive_definite(cov)

    # Should now be positive definite
    eigenvalues = np.linalg.eigvals(cov_pd)
    # After adding regularization, all eigenvalues should be positive
    assert all(eigenvalues > 0), f"Expected all positive eigenvalues, got {eigenvalues}"


def test_ensure_positive_definite_custom_epsilon(config_sample):
    """Test _ensure_positive_definite with custom regularization."""
    cov = np.eye(5)

    reconciler = MintReconciler(config=config_sample)
    cov_pd = reconciler._ensure_positive_definite(cov, epsilon=1e-3)

    # Check that matrix is positive definite
    eigenvalues = np.linalg.eigvals(cov_pd)
    assert all(eigenvalues > 0)

    # For an identity matrix with large epsilon, regularization should be added
    # Min eigenvalue should be at least epsilon (since starting eigenvalue is 1)
    assert np.min(eigenvalues) >= 1.0


# ============================================================================
# Reconciliation Matrix Tests
# ============================================================================


def test_compute_reconciliation_matrix(config_shrinkage, simple_hierarchy, simple_historical_data):
    """Test _compute_reconciliation_matrix computation."""
    forecasts, actuals = simple_historical_data

    reconciler = MintReconciler(config=config_shrinkage)
    reconciler.fit(forecasts, actuals, simple_hierarchy)

    G = reconciler._compute_reconciliation_matrix(simple_hierarchy)

    # Check shape
    n_nodes = len(simple_hierarchy.nodes)
    assert G.shape == (n_nodes, n_nodes)

    # Check that G @ S = S (coherence property)
    S = simple_hierarchy.aggregation_matrix
    GS = G @ S
    assert np.allclose(GS, S, atol=1e-10), "G should satisfy GS = S"


def test_compute_reconciliation_matrix_without_fit(config_shrinkage, simple_hierarchy):
    """Test _compute_reconciliation_matrix without fitting."""
    reconciler = MintReconciler(config=config_shrinkage)

    with pytest.raises(ValueError, match="Covariance matrix not fitted"):
        reconciler._compute_reconciliation_matrix(simple_hierarchy)


def test_compute_reconciliation_matrix_no_aggregation_matrix(
    config_shrinkage, simple_historical_data
):
    """Test _compute_reconciliation_matrix with missing aggregation matrix."""
    forecasts, actuals = simple_historical_data

    # Create hierarchy without aggregation matrix
    hierarchy = HierarchyDefinition()
    hierarchy.add_node("Total", level=0, node_type="aggregate")
    hierarchy.add_node("A", level=1, node_type="bottom", parent="Total")

    reconciler = MintReconciler(config=config_shrinkage)

    # Manually set covariance to bypass fit
    reconciler.covariance_matrix = np.eye(2)
    reconciler.fitted = True

    with pytest.raises(ValueError, match="aggregation matrix"):
        reconciler._compute_reconciliation_matrix(hierarchy)


# ============================================================================
# Save/Load Tests
# ============================================================================


def test_save_and_load(config_shrinkage, simple_hierarchy, simple_historical_data, tmp_path):
    """Test saving and loading reconciler."""
    forecasts, actuals = simple_historical_data

    # Create and fit reconciler
    reconciler = MintReconciler(config=config_shrinkage)
    reconciler.fit(forecasts, actuals, simple_hierarchy)

    # Save
    save_path = tmp_path / "mint_reconciler.json"
    reconciler.save(save_path)

    assert save_path.exists()
    assert (tmp_path / "mint_reconciler.cov.npy").exists()

    # Load
    loaded = MintReconciler.load(save_path)

    assert loaded.fitted
    assert loaded.covariance_method == "shrinkage"
    assert loaded.covariance_matrix is not None
    assert np.allclose(loaded.covariance_matrix, reconciler.covariance_matrix)


def test_save_without_fit(config_shrinkage, tmp_path):
    """Test saving reconciler without fitting."""
    reconciler = MintReconciler(config=config_shrinkage)

    save_path = tmp_path / "mint_reconciler_unfitted.json"
    reconciler.save(save_path)

    assert save_path.exists()

    # Load
    loaded = MintReconciler.load(save_path)
    assert not loaded.fitted
    assert loaded.covariance_matrix is None


def test_load_nonexistent_file(tmp_path):
    """Test loading from non-existent file."""
    with pytest.raises(FileNotFoundError):
        MintReconciler.load(tmp_path / "nonexistent.json")


# ============================================================================
# Brazil Hierarchy Tests
# ============================================================================


@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / "config" / "hierarchy_brazil.yaml").exists(),
    reason="Brazil hierarchy config not found",
)
def test_reconcile_brazil_hierarchy(brazil_hierarchy):
    """Test reconciliation with full Brazilian hierarchy."""
    np.random.seed(42)

    # Get all nodes
    all_nodes = brazil_hierarchy.get_node_names_sorted()
    n_nodes = len(all_nodes)

    # Generate historical data (50 periods)
    n_periods = 50
    bottom_nodes = brazil_hierarchy.get_bottom_level_nodes()

    # Generate bottom-level data
    historical_bottom = {}
    for node in bottom_nodes:
        # Random walk
        historical_bottom[node] = 100 + np.cumsum(np.random.randn(n_periods) * 5)

    # Aggregate to upper levels
    historical_actuals = {}
    for node in all_nodes:
        node_obj = brazil_hierarchy.nodes[node]
        if node_obj.node_type == "bottom":
            historical_actuals[node] = historical_bottom[node]
        elif node_obj.node_type == "aggregate":
            # Sum children
            descendants = brazil_hierarchy._get_bottom_descendants(node)
            historical_actuals[node] = sum(
                historical_bottom[desc] for desc in descendants if desc in historical_bottom
            )
        else:
            # Calculated nodes - set to zero for this test
            historical_actuals[node] = np.zeros(n_periods)

    actuals = pd.DataFrame(historical_actuals)[all_nodes]

    # Generate forecasts with errors
    forecasts_dict = {}
    for node in all_nodes:
        error_std = 10 if brazil_hierarchy.nodes[node].level == 0 else 5
        forecasts_dict[node] = actuals[node] + np.random.randn(n_periods) * error_std

    forecasts = pd.DataFrame(forecasts_dict)[all_nodes]

    # Configure and fit
    config = ReconciliationConfig(
        method="mint",
        covariance_method="shrinkage",
        check_coherence=True,
    )

    reconciler = MintReconciler(config=config)
    reconciler.fit(forecasts, actuals, brazil_hierarchy)

    # Create base forecasts for reconciliation (2 periods)
    base_forecasts_dict = {}
    for node in all_nodes:
        base_forecasts_dict[node] = forecasts[node].iloc[:2].values

    base_forecasts = pd.DataFrame(base_forecasts_dict)[all_nodes]

    # Reconcile
    result = reconciler.reconcile(base_forecasts, brazil_hierarchy)

    # Validate
    assert result.reconciled_forecasts.shape == base_forecasts.shape
    assert result.metadata["coherence_after"] < 1e-5
    assert result.validate_coherence(brazil_hierarchy, tolerance=1e-6)


# ============================================================================
# Edge Cases and Numerical Stability
# ============================================================================


def test_perfect_forecasts(config_shrinkage, simple_hierarchy):
    """Test with perfect forecasts (zero residuals)."""
    np.random.seed(42)

    # Get nodes
    all_nodes = simple_hierarchy.get_node_names_sorted()

    # Create coherent data (no errors)
    bottom_data = {
        "A1": [100] * 30,
        "A2": [80] * 30,
        "B1": [120] * 30,
        "B2": [90] * 30,
    }

    data_dict = {
        "A1": bottom_data["A1"],
        "A2": bottom_data["A2"],
        "B1": bottom_data["B1"],
        "B2": bottom_data["B2"],
        "A": [180] * 30,
        "B": [210] * 30,
        "Total": [390] * 30,
    }

    forecasts = pd.DataFrame(data_dict)[all_nodes]
    actuals = forecasts.copy()

    # Fit (residuals will be zero)
    reconciler = MintReconciler(config=config_shrinkage)
    reconciler.fit(forecasts, actuals, simple_hierarchy)

    # Covariance should still be positive definite (due to regularization)
    assert reconciler.covariance_matrix is not None
    eigenvalues = np.linalg.eigvals(reconciler.covariance_matrix)
    assert all(eigenvalues > 0)


def test_identical_forecasts(config_shrinkage, simple_hierarchy, simple_historical_data):
    """Test with identical forecasts across time."""
    forecasts, actuals = simple_historical_data

    reconciler = MintReconciler(config=config_shrinkage)
    reconciler.fit(forecasts, actuals, simple_hierarchy)

    # Create base forecasts where all timesteps are identical
    all_nodes = simple_hierarchy.get_node_names_sorted()
    base_forecasts = pd.DataFrame({node: [100] * 5 for node in all_nodes})[all_nodes]

    result = reconciler.reconcile(base_forecasts, simple_hierarchy)

    assert result.reconciled_forecasts.shape == base_forecasts.shape


def test_large_scale_factors(config_shrinkage, simple_hierarchy, simple_historical_data):
    """Test with very large forecast values."""
    forecasts, actuals = simple_historical_data

    # Scale up by 1e6
    forecasts_large = forecasts * 1e6
    actuals_large = actuals * 1e6

    reconciler = MintReconciler(config=config_shrinkage)
    reconciler.fit(forecasts_large, actuals_large, simple_hierarchy)

    # Create base forecasts
    all_nodes = simple_hierarchy.get_node_names_sorted()
    base_forecasts = forecasts_large.iloc[:2]

    result = reconciler.reconcile(base_forecasts, simple_hierarchy)

    # Should still achieve coherence
    assert result.metadata["coherence_after"] < 1e-5


def test_repr(config_shrinkage):
    """Test string representation."""
    reconciler = MintReconciler(config=config_shrinkage)

    repr_str = repr(reconciler)

    assert "MintReconciler" in repr_str
    assert "shrinkage" in repr_str
    assert "fitted=False" in repr_str


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_workflow_sample_method(
    simple_hierarchy, simple_historical_data, simple_base_forecasts
):
    """Test complete workflow with sample covariance method."""
    forecasts, actuals = simple_historical_data

    config = ReconciliationConfig(
        method="mint",
        covariance_method="sample",
        non_negative=True,
        check_coherence=True,
    )

    reconciler = MintReconciler(config=config)
    reconciler.fit(forecasts, actuals, simple_hierarchy)
    result = reconciler.reconcile(simple_base_forecasts, simple_hierarchy)

    assert result.validate_coherence(simple_hierarchy)


def test_full_workflow_diagonal_method(
    simple_hierarchy, simple_historical_data, simple_base_forecasts
):
    """Test complete workflow with diagonal covariance method."""
    forecasts, actuals = simple_historical_data

    config = ReconciliationConfig(
        method="mint",
        covariance_method="diagonal",
        non_negative=True,
        check_coherence=True,
    )

    reconciler = MintReconciler(config=config)
    reconciler.fit(forecasts, actuals, simple_hierarchy)
    result = reconciler.reconcile(simple_base_forecasts, simple_hierarchy)

    assert result.validate_coherence(simple_hierarchy)


def test_comparison_with_ols(simple_hierarchy, simple_historical_data, simple_base_forecasts):
    """Test that MinT with identity covariance matches OLS."""
    forecasts, actuals = simple_historical_data

    # MinT with identity covariance
    config_mint = ReconciliationConfig(
        method="mint",
        covariance_method="identity",
    )
    reconciler_mint = MintReconciler(config=config_mint)
    reconciler_mint.fit(forecasts, actuals, simple_hierarchy)
    result_mint = reconciler_mint.reconcile(simple_base_forecasts, simple_hierarchy)

    # For identity covariance, MinT should be equivalent to OLS
    # G_mint = S(S'S)^{-1}S' = G_ols
    S = simple_hierarchy.aggregation_matrix
    G_ols = S @ np.linalg.inv(S.T @ S) @ S.T

    G_mint = reconciler_mint._compute_reconciliation_matrix(simple_hierarchy)

    assert np.allclose(G_mint, G_ols, atol=1e-10)
