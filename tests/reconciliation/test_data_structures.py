"""Tests for reconciliation data structures."""

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.reconciliation import ReconciliationConfig, ReconciliationResult
from src.reconciliation.hierarchy import HierarchyDefinition


# Fixtures


@pytest.fixture
def sample_forecasts():
    """Create sample forecast DataFrame."""
    return pd.DataFrame(
        {
            "BR_National": [1000.0, 1100.0, 1200.0],
            "SE": [400.0, 440.0, 480.0],
            "S": [300.0, 330.0, 360.0],
            "Area_1": [200.0, 220.0, 240.0],
            "Area_2": [200.0, 220.0, 240.0],
            "Area_3": [300.0, 330.0, 360.0],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="h"),
    )


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
def brazil_hierarchy():
    """Load Brazil hierarchy."""
    return HierarchyDefinition("config/hierarchy_brazil.yaml")


# Tests for ReconciliationConfig


class TestReconciliationConfigCreation:
    """Tests for ReconciliationConfig creation."""

    def test_default_creation(self):
        """Test creating config with defaults."""
        config = ReconciliationConfig()

        assert config.method == "ols"
        assert config.bottom_up is False
        assert config.non_negative is True
        assert config.check_coherence is True
        assert config.hierarchy_path is None
        assert config.tolerance == 1e-6
        assert config.min_variance == 1e-8

    def test_custom_creation(self):
        """Test creating config with custom values."""
        config = ReconciliationConfig(
            method="mint",
            bottom_up=True,
            non_negative=False,
            check_coherence=False,
            hierarchy_path="config/hierarchy_brazil.yaml",
            tolerance=1e-4,
            min_variance=1e-6,
        )

        assert config.method == "mint"
        assert config.bottom_up is True
        assert config.non_negative is False
        assert config.check_coherence is False
        assert config.hierarchy_path == "config/hierarchy_brazil.yaml"
        assert config.tolerance == 1e-4
        assert config.min_variance == 1e-6

    def test_extra_fields(self):
        """Test config with extra fields."""
        config = ReconciliationConfig(
            method="wls",
            custom_param="value",
            weight_method="variance",
        )

        assert config.method == "wls"
        assert config.model_extra is not None
        assert config.get_extra_field("custom_param") == "value"
        assert config.get_extra_field("weight_method") == "variance"


class TestReconciliationConfigValidation:
    """Tests for ReconciliationConfig validation."""

    def test_method_validation_lowercase(self):
        """Test method name is converted to lowercase."""
        config = ReconciliationConfig(method="MINT")

        assert config.method == "mint"

    def test_method_validation_unknown(self):
        """Test warning for unknown method (but allows it)."""
        # Should not raise, but logs warning
        config = ReconciliationConfig(method="unknown_method")

        assert config.method == "unknown_method"

    def test_tolerance_positive(self):
        """Test tolerance must be positive."""
        with pytest.raises(ValidationError, match="tolerance"):
            ReconciliationConfig(tolerance=0)

        with pytest.raises(ValidationError, match="tolerance"):
            ReconciliationConfig(tolerance=-1e-6)

    def test_min_variance_positive(self):
        """Test min_variance must be positive."""
        with pytest.raises(ValidationError, match="min_variance"):
            ReconciliationConfig(min_variance=0)

        with pytest.raises(ValidationError, match="min_variance"):
            ReconciliationConfig(min_variance=-1e-8)

    def test_tolerance_validation_range(self):
        """Test tolerance validation for large values."""
        # Should work but log warning
        config = ReconciliationConfig(tolerance=2.0)
        assert config.tolerance == 2.0


class TestReconciliationConfigMethods:
    """Tests for ReconciliationConfig methods."""

    def test_get_extra_field(self):
        """Test getting extra field values."""
        config = ReconciliationConfig(method="mint", custom_param=42)

        assert config.get_extra_field("custom_param") == 42
        assert config.get_extra_field("missing") is None
        assert config.get_extra_field("missing", default=100) == 100

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ReconciliationConfig(
            method="ols",
            non_negative=True,
            tolerance=1e-5,
            custom_field="test",
        )

        d = config.to_dict()

        assert d["method"] == "ols"
        assert d["non_negative"] is True
        assert d["tolerance"] == 1e-5
        assert d["custom_field"] == "test"


# Tests for ReconciliationResult


class TestReconciliationResultCreation:
    """Tests for ReconciliationResult creation."""

    def test_basic_creation(self, sample_forecasts):
        """Test creating result with minimal fields."""
        result = ReconciliationResult(
            reconciled_forecasts=sample_forecasts,
        )

        assert result.reconciled_forecasts is not None
        assert result.metadata["method"] == "unknown"
        assert "timestamp" in result.metadata
        assert result.base_forecasts is None
        assert result.residual_covariance is None

    def test_full_creation(self, sample_forecasts):
        """Test creating result with all fields."""
        metadata = {
            "method": "mint",
            "timestamp": "2025-11-25T10:00:00",
            "coherence_before": 15.3,
            "coherence_after": 0.0,
            "elapsed_time": 0.123,
        }

        result = ReconciliationResult(
            reconciled_forecasts=sample_forecasts,
            metadata=metadata,
            base_forecasts=sample_forecasts * 1.1,
            residual_covariance=np.eye(6),
            reconciliation_matrix=np.eye(6),
        )

        assert result.reconciled_forecasts is not None
        assert result.metadata["method"] == "mint"
        assert result.metadata["coherence_after"] == 0.0
        assert result.base_forecasts is not None
        assert result.residual_covariance is not None
        assert result.reconciliation_matrix is not None

    def test_empty_forecasts_raises(self):
        """Test empty forecasts raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ReconciliationResult(reconciled_forecasts=pd.DataFrame())

    def test_none_forecasts_raises(self):
        """Test None forecasts raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ReconciliationResult(reconciled_forecasts=None)

    def test_auto_timestamp(self, sample_forecasts):
        """Test timestamp is auto-generated if not provided."""
        result = ReconciliationResult(reconciled_forecasts=sample_forecasts)

        assert "timestamp" in result.metadata
        # Should be valid ISO format
        datetime.fromisoformat(result.metadata["timestamp"])


class TestReconciliationResultValidateCoherence:
    """Tests for validate_coherence method."""

    def test_validate_coherent_forecasts(self, simple_hierarchy):
        """Test validation of perfectly coherent forecasts."""
        # Create coherent forecasts (National = A + B)
        forecasts = pd.DataFrame(
            {
                "National": [300.0, 400.0],
                "A": [100.0, 150.0],
                "B": [200.0, 250.0],
            },
            index=[0, 1],
        )

        result = ReconciliationResult(reconciled_forecasts=forecasts)
        is_coherent = result.validate_coherence(simple_hierarchy, tolerance=1e-6)

        assert is_coherent is True
        assert result.metadata["coherence_error_validated"] < 1e-6

    def test_validate_incoherent_forecasts(self, simple_hierarchy):
        """Test validation of incoherent forecasts."""
        # Create incoherent forecasts (National != A + B)
        forecasts = pd.DataFrame(
            {
                "National": [400.0, 500.0],  # Should be 300, 400
                "A": [100.0, 150.0],
                "B": [200.0, 250.0],
            },
            index=[0, 1],
        )

        result = ReconciliationResult(reconciled_forecasts=forecasts)
        is_coherent = result.validate_coherence(simple_hierarchy, tolerance=1e-6)

        assert is_coherent is False
        assert result.metadata["coherence_error_validated"] > 1e-6

    def test_validate_with_tolerance(self, simple_hierarchy):
        """Test validation with different tolerance levels."""
        # Slightly incoherent forecasts
        forecasts = pd.DataFrame(
            {
                "National": [300.001, 400.001],
                "A": [100.0, 150.0],
                "B": [200.0, 250.0],
            },
            index=[0, 1],
        )

        result = ReconciliationResult(reconciled_forecasts=forecasts)

        # Should fail with strict tolerance
        assert result.validate_coherence(simple_hierarchy, tolerance=1e-6) is False

        # Should pass with loose tolerance
        assert result.validate_coherence(simple_hierarchy, tolerance=1e-2) is True

    def test_validate_missing_bottom_nodes(self, simple_hierarchy):
        """Test validation with missing bottom nodes."""
        forecasts = pd.DataFrame(
            {
                "National": [300.0],
                "A": [100.0],
                # Missing B
            },
            index=[0],
        )

        result = ReconciliationResult(reconciled_forecasts=forecasts)
        is_coherent = result.validate_coherence(simple_hierarchy)

        assert is_coherent is False

    def test_validate_no_aggregation_matrix(self, simple_hierarchy):
        """Test validation fails without aggregation matrix."""
        simple_hierarchy.aggregation_matrix = None

        forecasts = pd.DataFrame(
            {
                "National": [300.0],
                "A": [100.0],
                "B": [200.0],
            },
            index=[0],
        )

        result = ReconciliationResult(reconciled_forecasts=forecasts)

        with pytest.raises(ValueError, match="aggregation matrix"):
            result.validate_coherence(simple_hierarchy)


class TestReconciliationResultSerialization:
    """Tests for serialization methods."""

    def test_to_dict(self, sample_forecasts):
        """Test converting result to dictionary."""
        metadata = {"method": "ols", "timestamp": "2025-11-25T10:00:00"}
        result = ReconciliationResult(
            reconciled_forecasts=sample_forecasts,
            metadata=metadata,
            base_forecasts=sample_forecasts * 0.9,
            residual_covariance=np.eye(6),
            reconciliation_matrix=np.eye(6),
        )

        d = result.to_dict()

        assert "reconciled_forecasts" in d
        assert "metadata" in d
        assert "base_forecasts" in d
        assert "residual_covariance" in d
        assert "reconciliation_matrix" in d

        # Check DataFrame conversion
        assert isinstance(d["reconciled_forecasts"], dict)
        assert "data" in d["reconciled_forecasts"]
        assert "index" in d["reconciled_forecasts"]
        assert "columns" in d["reconciled_forecasts"]

        # Check array conversion
        assert isinstance(d["residual_covariance"], list)
        assert isinstance(d["reconciliation_matrix"], list)

    def test_to_dict_minimal(self, sample_forecasts):
        """Test to_dict with minimal fields."""
        result = ReconciliationResult(reconciled_forecasts=sample_forecasts)

        d = result.to_dict()

        assert "reconciled_forecasts" in d
        assert "metadata" in d
        assert d["base_forecasts"] is None
        assert d["residual_covariance"] is None

    def test_from_dict(self, sample_forecasts):
        """Test creating result from dictionary."""
        original = ReconciliationResult(
            reconciled_forecasts=sample_forecasts,
            metadata={"method": "mint"},
            base_forecasts=sample_forecasts * 1.1,
        )

        d = original.to_dict()
        restored = ReconciliationResult.from_dict(d)

        # Check forecasts (check_freq=False to handle datetime index frequency)
        pd.testing.assert_frame_equal(
            restored.reconciled_forecasts, original.reconciled_forecasts, check_freq=False
        )

        # Check metadata
        assert restored.metadata["method"] == "mint"

        # Check base forecasts
        assert restored.base_forecasts is not None
        pd.testing.assert_frame_equal(
            restored.base_forecasts, original.base_forecasts, check_freq=False
        )

    def test_from_dict_with_arrays(self, sample_forecasts):
        """Test from_dict with covariance and reconciliation matrices."""
        original = ReconciliationResult(
            reconciled_forecasts=sample_forecasts,
            residual_covariance=np.eye(6) * 2,
            reconciliation_matrix=np.eye(6) * 0.5,
        )

        d = original.to_dict()
        restored = ReconciliationResult.from_dict(d)

        # Check arrays
        assert restored.residual_covariance is not None
        np.testing.assert_array_almost_equal(
            restored.residual_covariance, original.residual_covariance
        )

        assert restored.reconciliation_matrix is not None
        np.testing.assert_array_almost_equal(
            restored.reconciliation_matrix, original.reconciliation_matrix
        )

    def test_roundtrip_serialization(self, sample_forecasts):
        """Test full roundtrip: result -> dict -> result."""
        original = ReconciliationResult(
            reconciled_forecasts=sample_forecasts,
            metadata={"method": "mint", "coherence": 0.001},
            base_forecasts=sample_forecasts * 0.95,
            residual_covariance=np.random.rand(6, 6),
            reconciliation_matrix=np.random.rand(6, 6),
        )

        # Roundtrip
        d = original.to_dict()
        restored = ReconciliationResult.from_dict(d)

        # Compare (check_freq=False to handle datetime index frequency)
        pd.testing.assert_frame_equal(
            restored.reconciled_forecasts, original.reconciled_forecasts, check_freq=False
        )
        assert restored.metadata == original.metadata
        pd.testing.assert_frame_equal(
            restored.base_forecasts, original.base_forecasts, check_freq=False
        )
        np.testing.assert_array_almost_equal(
            restored.residual_covariance, original.residual_covariance
        )
        np.testing.assert_array_almost_equal(
            restored.reconciliation_matrix, original.reconciliation_matrix
        )


class TestReconciliationResultPersistence:
    """Tests for save/load methods."""

    def test_save_and_load(self, sample_forecasts, tmp_path):
        """Test saving and loading result."""
        result_path = tmp_path / "result.pkl"

        # Create and save result
        original = ReconciliationResult(
            reconciled_forecasts=sample_forecasts,
            metadata={"method": "ols", "coherence": 0.0},
        )
        original.save(result_path)

        assert result_path.exists()

        # Load result
        loaded = ReconciliationResult.load(result_path)

        # Compare (check_freq=False to handle datetime index frequency)
        pd.testing.assert_frame_equal(
            loaded.reconciled_forecasts, original.reconciled_forecasts, check_freq=False
        )
        assert loaded.metadata == original.metadata

    def test_save_creates_directory(self, sample_forecasts, tmp_path):
        """Test save creates parent directories."""
        result_path = tmp_path / "subdir" / "results" / "result.pkl"

        result = ReconciliationResult(reconciled_forecasts=sample_forecasts)
        result.save(result_path)

        assert result_path.exists()
        assert result_path.parent.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError, match="not found"):
            ReconciliationResult.load(tmp_path / "missing.pkl")

    def test_save_load_full_result(self, sample_forecasts, tmp_path):
        """Test save/load with all fields populated."""
        result_path = tmp_path / "full_result.pkl"

        original = ReconciliationResult(
            reconciled_forecasts=sample_forecasts,
            metadata={"method": "mint", "time": 0.5},
            base_forecasts=sample_forecasts * 1.05,
            residual_covariance=np.eye(6) * 2,
            reconciliation_matrix=np.random.rand(6, 6),
        )

        original.save(result_path)
        loaded = ReconciliationResult.load(result_path)

        # Compare all fields (check_freq=False to handle datetime index frequency)
        pd.testing.assert_frame_equal(
            loaded.reconciled_forecasts, original.reconciled_forecasts, check_freq=False
        )
        assert loaded.metadata == original.metadata
        pd.testing.assert_frame_equal(
            loaded.base_forecasts, original.base_forecasts, check_freq=False
        )
        np.testing.assert_array_almost_equal(
            loaded.residual_covariance, original.residual_covariance
        )
        np.testing.assert_array_almost_equal(
            loaded.reconciliation_matrix, original.reconciliation_matrix
        )


class TestReconciliationResultRepr:
    """Tests for string representation."""

    def test_repr(self, sample_forecasts):
        """Test string representation."""
        result = ReconciliationResult(
            reconciled_forecasts=sample_forecasts,
            metadata={"method": "mint"},
        )

        repr_str = repr(result)

        assert "ReconciliationResult" in repr_str
        assert "method='mint'" in repr_str
        assert "timesteps=3" in repr_str
        assert "nodes=6" in repr_str


class TestIntegrationWithHierarchy:
    """Integration tests with real hierarchy."""

    def test_validate_with_brazil_hierarchy(self, brazil_hierarchy):
        """Test validation with Brazil hierarchy."""
        # Get nodes
        bottom_nodes = sorted(brazil_hierarchy.get_bottom_level_nodes())
        all_nodes = brazil_hierarchy.get_node_names_sorted()

        # Create coherent forecasts from bottom-up
        n_timesteps = 5
        y_bottom = np.random.rand(n_timesteps, len(bottom_nodes)) * 100

        # Aggregate to get all forecasts
        S = brazil_hierarchy.aggregation_matrix
        y_all = (S @ y_bottom.T).T

        forecasts = pd.DataFrame(y_all, columns=all_nodes)

        # Create result
        result = ReconciliationResult(reconciled_forecasts=forecasts)

        # Should be coherent
        is_coherent = result.validate_coherence(brazil_hierarchy, tolerance=1e-6)
        assert is_coherent is True
        assert result.metadata["coherence_error_validated"] < 1e-6

    def test_validate_incoherent_brazil(self, brazil_hierarchy):
        """Test detection of incoherent forecasts for Brazil."""
        # Get nodes
        all_nodes = brazil_hierarchy.get_node_names_sorted()

        # Create random (likely incoherent) forecasts
        n_timesteps = 5
        y_random = np.random.rand(n_timesteps, len(all_nodes)) * 100

        forecasts = pd.DataFrame(y_random, columns=all_nodes)

        # Create result
        result = ReconciliationResult(reconciled_forecasts=forecasts)

        # Likely to be incoherent
        is_coherent = result.validate_coherence(brazil_hierarchy, tolerance=1e-6)

        # Check that coherence error is computed
        assert "coherence_error_validated" in result.metadata
        assert result.metadata["coherence_error_validated"] >= 0
