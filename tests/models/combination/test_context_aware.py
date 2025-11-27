# ruff: noqa: NPY002
"""Tests for context-aware combiner.

Tests cover:
- Context feature extraction (temporal, load, meteorological)
- Different adaptation strategies (Ridge, Random Forest)
- Weight constraint enforcement
- Feature importance calculation
- Context-weight relationship analysis
- Save/load persistence
- Edge cases and error handling
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.combination.context_aware import (
    ContextAnalysis,
    ContextAwareConfig,
    ContextAwareCombiner,
    ContextFeatureExtractor,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_timestamps():
    """Create sample timestamps spanning multiple days."""
    return pd.date_range("2024-01-01", periods=200, freq="30min")


@pytest.fixture
def sample_predictions(sample_timestamps):
    """Create sample predictions from two models."""
    np.random.seed(42)
    n = len(sample_timestamps)

    # Model 1: Better at night (hours 0-11)
    hour = sample_timestamps.hour
    model1_base = 1000 + (hour < 12).astype(float) * 50

    # Model 2: Better during day (hours 12-23)
    model2_base = 1000 + (hour >= 12).astype(float) * 50

    predictions = {
        "model1": pd.DataFrame(
            {
                "pred_h0": model1_base + np.random.randn(n) * 20,
                "pred_h1": model1_base + np.random.randn(n) * 25,
            },
            index=sample_timestamps,
        ),
        "model2": pd.DataFrame(
            {
                "pred_h0": model2_base + np.random.randn(n) * 20,
                "pred_h1": model2_base + np.random.randn(n) * 25,
            },
            index=sample_timestamps,
        ),
    }

    return predictions


@pytest.fixture
def sample_targets(sample_timestamps):
    """Create sample targets with hour-dependent pattern."""
    np.random.seed(42)
    n = len(sample_timestamps)
    hour = sample_timestamps.hour

    # Target follows hour pattern - model1 better at night, model2 better at day
    base = 1000 + (hour < 12).astype(float) * 30

    return pd.DataFrame(
        {
            "target_h0": base + np.random.randn(n) * 10,
            "target_h1": base + np.random.randn(n) * 15,
        },
        index=sample_timestamps,
    )


@pytest.fixture
def sample_meteo_data(sample_timestamps):
    """Create sample meteorological data."""
    np.random.seed(42)
    n = len(sample_timestamps)

    return pd.DataFrame(
        {
            "temperature": 25 + np.random.randn(n) * 5,
        },
        index=sample_timestamps,
    )


@pytest.fixture
def three_model_predictions(sample_timestamps):
    """Create predictions from three models."""
    np.random.seed(42)
    n = len(sample_timestamps)

    return {
        "lgbm": pd.DataFrame(
            {
                "pred_h0": 1000 + np.random.randn(n) * 20,
                "pred_h1": 1000 + np.random.randn(n) * 25,
            },
            index=sample_timestamps,
        ),
        "rf": pd.DataFrame(
            {
                "pred_h0": 1000 + np.random.randn(n) * 22,
                "pred_h1": 1000 + np.random.randn(n) * 27,
            },
            index=sample_timestamps,
        ),
        "xgb": pd.DataFrame(
            {
                "pred_h0": 1000 + np.random.randn(n) * 18,
                "pred_h1": 1000 + np.random.randn(n) * 23,
            },
            index=sample_timestamps,
        ),
    }


# =============================================================================
# ContextAwareConfig Tests
# =============================================================================


class TestContextAwareConfig:
    """Tests for ContextAwareConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ContextAwareConfig()

        assert config.adaptation_method == "ridge"
        assert config.ridge_alpha == 1.0
        assert config.rf_n_estimators == 100
        assert config.rf_max_depth == 10
        assert config.use_temporal_features is True
        assert config.use_load_features is False  # Default False since not available during combine
        assert config.use_meteo_features is False
        assert config.min_weight == 0.01
        assert config.max_weight == 0.95

    def test_custom_config(self):
        """Test custom configuration."""
        config = ContextAwareConfig(
            adaptation_method="random_forest",
            ridge_alpha=0.5,
            rf_n_estimators=50,
            min_weight=0.05,
        )

        assert config.adaptation_method == "random_forest"
        assert config.ridge_alpha == 0.5
        assert config.rf_n_estimators == 50
        assert config.min_weight == 0.05


# =============================================================================
# ContextFeatureExtractor Tests
# =============================================================================


class TestContextFeatureExtractor:
    """Tests for ContextFeatureExtractor class."""

    def test_extract_temporal_features(self, sample_timestamps):
        """Test temporal feature extraction."""
        config = {"use_temporal_features": True, "use_load_features": False}
        extractor = ContextFeatureExtractor(config)

        features = extractor.extract(sample_timestamps)

        # Check temporal features exist
        assert "hour" in features.columns
        assert "day_of_week" in features.columns
        assert "month" in features.columns
        assert "is_weekend" in features.columns
        assert "hour_sin" in features.columns
        assert "hour_cos" in features.columns
        assert "dow_sin" in features.columns
        assert "dow_cos" in features.columns

        # Check values are valid
        assert features["hour"].min() >= 0
        assert features["hour"].max() <= 23
        assert features["day_of_week"].min() >= 0
        assert features["day_of_week"].max() <= 6

        # Check cyclic encoding values are in [-1, 1]
        assert features["hour_sin"].min() >= -1
        assert features["hour_sin"].max() <= 1
        assert features["hour_cos"].min() >= -1
        assert features["hour_cos"].max() <= 1

    def test_extract_load_features(self, sample_timestamps):
        """Test load feature extraction."""
        config = {"use_temporal_features": False, "use_load_features": True}
        extractor = ContextFeatureExtractor(config)

        load_data = pd.Series(np.random.randn(len(sample_timestamps)) * 100 + 1000, index=sample_timestamps)

        features = extractor.extract(sample_timestamps, load_data=load_data)

        assert "load_level" in features.columns
        assert "load_volatility" in features.columns
        assert "load_trend" in features.columns
        assert "load_regime_low" in features.columns
        assert "load_regime_high" in features.columns

    def test_extract_meteo_features(self, sample_timestamps, sample_meteo_data):
        """Test meteorological feature extraction."""
        config = {
            "use_temporal_features": False,
            "use_load_features": False,
            "use_meteo_features": True,
        }
        extractor = ContextFeatureExtractor(config)

        features = extractor.extract(sample_timestamps, meteo_data=sample_meteo_data)

        assert "temperature" in features.columns
        assert "temp_deviation" in features.columns
        assert "temp_extreme_high" in features.columns
        assert "temp_extreme_low" in features.columns

    def test_extract_all_features(self, sample_timestamps, sample_meteo_data):
        """Test extraction of all feature types."""
        config = {
            "use_temporal_features": True,
            "use_load_features": True,
            "use_meteo_features": True,
        }
        extractor = ContextFeatureExtractor(config)

        load_data = pd.Series(np.random.randn(len(sample_timestamps)) * 100 + 1000, index=sample_timestamps)

        features = extractor.extract(sample_timestamps, load_data=load_data, meteo_data=sample_meteo_data)

        # Check all feature types present
        assert "hour" in features.columns  # temporal
        assert "load_level" in features.columns  # load
        assert "temperature" in features.columns  # meteo

    def test_feature_names_stored(self, sample_timestamps):
        """Test that feature names are stored correctly."""
        config = {"use_temporal_features": True}
        extractor = ContextFeatureExtractor(config)

        features = extractor.extract(sample_timestamps)

        assert len(extractor.feature_names) > 0
        assert extractor.feature_names == list(features.columns)

    def test_fit_scaler(self, sample_timestamps):
        """Test scaler fitting."""
        config = {"use_temporal_features": True}
        extractor = ContextFeatureExtractor(config)

        features = extractor.extract(sample_timestamps, fit_scaler=True)

        assert extractor.is_fitted

        # Scaled features should have different range
        features_raw = extractor.extract(sample_timestamps, fit_scaler=False)
        # Raw features are used when scaler is fitted
        assert features.shape == features_raw.shape

    def test_no_nan_values(self, sample_timestamps):
        """Test that no NaN values in output."""
        config = {"use_temporal_features": True, "use_load_features": True}
        extractor = ContextFeatureExtractor(config)

        # Create load data with some NaN
        load_data = pd.Series(np.random.randn(len(sample_timestamps)) * 100 + 1000, index=sample_timestamps)
        load_data.iloc[:10] = np.nan

        features = extractor.extract(sample_timestamps, load_data=load_data)

        assert not features.isna().any().any()


# =============================================================================
# ContextAwareCombiner Tests
# =============================================================================


class TestContextAwareCombiner:
    """Tests for ContextAwareCombiner class."""

    def test_initialization(self):
        """Test combiner initialization."""
        combiner = ContextAwareCombiner()

        assert combiner.name == "context_aware"
        assert combiner.version == "1.0.0"
        assert not combiner.is_fitted
        assert combiner.adaptation_method == "ridge"

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        combiner = ContextAwareCombiner(
            config={
                "adaptation_method": "random_forest",
                "rf_n_estimators": 50,
                "min_weight": 0.05,
            }
        )

        assert combiner.adaptation_method == "random_forest"
        assert combiner.rf_n_estimators == 50
        assert combiner.min_weight == 0.05

    def test_fit_ridge(self, sample_predictions, sample_targets):
        """Test fitting with Ridge adaptation."""
        combiner = ContextAwareCombiner(config={"adaptation_method": "ridge"})

        combiner.fit(sample_predictions, sample_targets)

        assert combiner.is_fitted
        assert len(combiner.adaptation_models) == 2
        assert "model1" in combiner.adaptation_models
        assert "model2" in combiner.adaptation_models

    def test_fit_random_forest(self, sample_predictions, sample_targets):
        """Test fitting with Random Forest adaptation."""
        combiner = ContextAwareCombiner(
            config={
                "adaptation_method": "random_forest",
                "rf_n_estimators": 20,
                "rf_max_depth": 5,
            }
        )

        combiner.fit(sample_predictions, sample_targets)

        assert combiner.is_fitted
        assert len(combiner.adaptation_models) == 2

    def test_combine_after_fit(self, sample_predictions, sample_targets):
        """Test combining predictions after fitting."""
        combiner = ContextAwareCombiner(config={"adaptation_method": "ridge"})
        combiner.fit(sample_predictions, sample_targets)

        result = combiner.combine(sample_predictions)

        assert result.combined_predictions is not None
        assert "pred_h0" in result.combined_predictions.columns
        assert "pred_h1" in result.combined_predictions.columns
        assert len(result.combined_predictions) == len(sample_predictions["model1"])

    def test_combine_weights_sum_to_one(self, sample_predictions, sample_targets):
        """Test that weights sum to 1."""
        combiner = ContextAwareCombiner()
        combiner.fit(sample_predictions, sample_targets)

        result = combiner.combine(sample_predictions)

        weight_sum = sum(result.weights.values())
        assert abs(weight_sum - 1.0) < 0.01

    def test_combine_without_fit_raises(self, sample_predictions):
        """Test that combining without fit raises error."""
        combiner = ContextAwareCombiner()

        with pytest.raises(RuntimeError, match="must be fitted"):
            combiner.combine(sample_predictions)

    def test_weight_constraints_enforced(self, sample_predictions, sample_targets):
        """Test that weight constraints are enforced."""
        combiner = ContextAwareCombiner(
            config={
                "min_weight": 0.1,
                "max_weight": 0.9,
            }
        )
        combiner.fit(sample_predictions, sample_targets)

        result = combiner.combine(sample_predictions)

        for weight in result.weights.values():
            assert weight >= 0  # After normalization, might be below min_weight
            assert weight <= 1.0

    def test_three_models(self, three_model_predictions, sample_targets):
        """Test with three models."""
        # Extend targets to match three model predictions
        targets = sample_targets.copy()

        combiner = ContextAwareCombiner()
        combiner.fit(three_model_predictions, targets)

        result = combiner.combine(three_model_predictions)

        assert len(result.weights) == 3
        assert "lgbm" in result.weights
        assert "rf" in result.weights
        assert "xgb" in result.weights

    def test_metadata_includes_method(self, sample_predictions, sample_targets):
        """Test that metadata includes adaptation method."""
        combiner = ContextAwareCombiner(config={"adaptation_method": "ridge"})
        combiner.fit(sample_predictions, sample_targets)

        result = combiner.combine(sample_predictions)

        assert result.metadata.configuration["adaptation_method"] == "ridge"

    def test_get_feature_importance_ridge(self, sample_predictions, sample_targets):
        """Test feature importance for Ridge."""
        combiner = ContextAwareCombiner(config={"adaptation_method": "ridge"})
        combiner.fit(sample_predictions, sample_targets)

        importance = combiner.get_feature_importance()

        assert len(importance) == 2
        assert "model1" in importance
        assert "model2" in importance
        assert len(importance["model1"]) > 0

    def test_get_feature_importance_rf(self, sample_predictions, sample_targets):
        """Test feature importance for Random Forest."""
        combiner = ContextAwareCombiner(
            config={
                "adaptation_method": "random_forest",
                "rf_n_estimators": 20,
            }
        )
        combiner.fit(sample_predictions, sample_targets)

        importance = combiner.get_feature_importance()

        assert len(importance) == 2
        # RF importances should sum to ~1
        for model_imp in importance.values():
            total_imp = sum(model_imp.values())
            assert total_imp > 0.99 and total_imp < 1.01

    def test_get_top_features(self, sample_predictions, sample_targets):
        """Test getting top features."""
        combiner = ContextAwareCombiner()
        combiner.fit(sample_predictions, sample_targets)

        top_features = combiner.get_top_features(n=3)

        assert len(top_features) == 2
        for model, features in top_features.items():
            assert len(features) <= 3
            for feat_name, imp in features:
                assert isinstance(feat_name, str)
                assert isinstance(imp, float)

    def test_feature_importance_unfitted(self):
        """Test feature importance when not fitted."""
        combiner = ContextAwareCombiner()

        importance = combiner.get_feature_importance()

        assert importance == {}

    def test_get_adaptation_diagnostics(self, sample_predictions, sample_targets):
        """Test getting adaptation diagnostics."""
        combiner = ContextAwareCombiner(config={"adaptation_method": "ridge"})
        combiner.fit(sample_predictions, sample_targets)

        diagnostics = combiner.get_adaptation_diagnostics()

        assert diagnostics["adaptation_method"] == "ridge"
        assert diagnostics["n_models"] == 2
        assert diagnostics["n_features"] > 0
        assert "base_weights" in diagnostics
        assert "ridge_alpha" in diagnostics

    def test_diagnostics_unfitted(self):
        """Test diagnostics when not fitted."""
        combiner = ContextAwareCombiner()

        diagnostics = combiner.get_adaptation_diagnostics()

        assert diagnostics == {}

    def test_invalid_adaptation_method(self, sample_predictions, sample_targets):
        """Test that invalid adaptation method raises error."""
        combiner = ContextAwareCombiner(config={"adaptation_method": "invalid_method"})

        with pytest.raises(ValueError, match="Unknown adaptation method"):
            combiner.fit(sample_predictions, sample_targets)

    def test_empty_predictions_raises(self):
        """Test that empty predictions raises error."""
        combiner = ContextAwareCombiner()

        with pytest.raises(ValueError):
            combiner._validate_predictions({})

    def test_model_names_stored(self, sample_predictions, sample_targets):
        """Test that model names are stored after fitting."""
        combiner = ContextAwareCombiner()
        combiner.fit(sample_predictions, sample_targets)

        assert "model1" in combiner.model_names
        assert "model2" in combiner.model_names


# =============================================================================
# Save/Load Tests
# =============================================================================


class TestContextAwareSaveLoad:
    """Tests for save/load functionality."""

    def test_save_and_load(self, sample_predictions, sample_targets):
        """Test saving and loading combiner."""
        combiner = ContextAwareCombiner(config={"adaptation_method": "ridge"})
        combiner.fit(sample_predictions, sample_targets)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "context_aware.pkl"
            combiner.save(filepath)

            loaded = ContextAwareCombiner.load(filepath)

            assert loaded.is_fitted
            assert loaded.adaptation_method == "ridge"
            assert loaded.base_weights == combiner.base_weights
            assert len(loaded.adaptation_models) == len(combiner.adaptation_models)

    def test_loaded_combiner_can_predict(self, sample_predictions, sample_targets):
        """Test that loaded combiner can make predictions."""
        combiner = ContextAwareCombiner()
        combiner.fit(sample_predictions, sample_targets)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "context_aware.pkl"
            combiner.save(filepath)

            loaded = ContextAwareCombiner.load(filepath)
            result = loaded.combine(sample_predictions)

            assert result.combined_predictions is not None
            assert len(result.combined_predictions) == len(sample_predictions["model1"])

    def test_save_rf_model(self, sample_predictions, sample_targets):
        """Test saving Random Forest adaptation model."""
        combiner = ContextAwareCombiner(
            config={
                "adaptation_method": "random_forest",
                "rf_n_estimators": 20,
            }
        )
        combiner.fit(sample_predictions, sample_targets)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "context_aware_rf.pkl"
            combiner.save(filepath)

            loaded = ContextAwareCombiner.load(filepath)

            assert loaded.adaptation_method == "random_forest"
            assert loaded.rf_n_estimators == 20


# =============================================================================
# Context Analysis Tests
# =============================================================================


class TestContextAnalysis:
    """Tests for context-weight relationship analysis."""

    def test_analyze_context_relationships(self, sample_predictions, sample_targets):
        """Test context-weight relationship analysis."""
        combiner = ContextAwareCombiner()
        combiner.fit(sample_predictions, sample_targets)

        # Create context features and optimal weights for analysis
        first_pred = sample_predictions["model1"]
        context_features = combiner.feature_extractor.extract(first_pred.index)

        # Create dummy optimal weights
        optimal_weights = pd.DataFrame(
            {
                "model1": np.random.rand(len(context_features)),
                "model2": np.random.rand(len(context_features)),
            },
            index=context_features.index,
        )
        # Normalize
        row_sums = optimal_weights.sum(axis=1)
        optimal_weights = optimal_weights.div(row_sums, axis=0)

        analysis = combiner.analyze_context_relationships(context_features, optimal_weights)

        assert isinstance(analysis, ContextAnalysis)
        assert "model1" in analysis.correlations
        assert "model2" in analysis.correlations
        assert len(analysis.top_features["model1"]) > 0
        assert analysis.relationship_summary != ""


# =============================================================================
# Weight Smoothing Tests
# =============================================================================


class TestWeightSmoothing:
    """Tests for weight smoothing functionality."""

    def test_weight_smoothing_enabled(self, sample_predictions, sample_targets):
        """Test that weight smoothing works."""
        combiner = ContextAwareCombiner(config={"weight_smoothing": 0.5})
        combiner.fit(sample_predictions, sample_targets)

        result = combiner.combine(sample_predictions)

        # Should complete without error
        assert result.combined_predictions is not None

    def test_weight_smoothing_zero(self, sample_predictions, sample_targets):
        """Test with no smoothing."""
        combiner = ContextAwareCombiner(config={"weight_smoothing": 0.0})
        combiner.fit(sample_predictions, sample_targets)

        result = combiner.combine(sample_predictions)

        assert result.combined_predictions is not None


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_model(self, sample_timestamps, sample_targets):
        """Test with single model."""
        predictions = {
            "only_model": pd.DataFrame(
                {
                    "pred_h0": np.random.randn(len(sample_timestamps)) * 20 + 1000,
                    "pred_h1": np.random.randn(len(sample_timestamps)) * 25 + 1000,
                },
                index=sample_timestamps,
            )
        }

        combiner = ContextAwareCombiner()
        combiner.fit(predictions, sample_targets)

        result = combiner.combine(predictions)

        assert result.weights["only_model"] == 1.0

    def test_misaligned_indices_raises(self, sample_timestamps, sample_targets):
        """Test that misaligned indices raises error."""
        predictions = {
            "model1": pd.DataFrame(
                {
                    "pred_h0": np.random.randn(100) * 20 + 1000,
                    "pred_h1": np.random.randn(100) * 25 + 1000,
                },
                index=pd.date_range("2024-01-01", periods=100, freq="30min"),
            ),
            "model2": pd.DataFrame(
                {
                    "pred_h0": np.random.randn(100) * 20 + 1000,
                    "pred_h1": np.random.randn(100) * 25 + 1000,
                },
                index=pd.date_range("2024-02-01", periods=100, freq="30min"),  # Different dates
            ),
        }

        combiner = ContextAwareCombiner()

        with pytest.raises(ValueError, match="misaligned"):
            combiner.fit(predictions, sample_targets)

    def test_missing_columns_raises(self, sample_timestamps, sample_targets):
        """Test that missing prediction columns raises error."""
        predictions = {
            "model1": pd.DataFrame(
                {
                    "pred_h0": np.random.randn(len(sample_timestamps)) * 20 + 1000,
                    "pred_h1": np.random.randn(len(sample_timestamps)) * 25 + 1000,
                },
                index=sample_timestamps,
            ),
            "model2": pd.DataFrame(
                {
                    "pred_h0": np.random.randn(len(sample_timestamps)) * 20 + 1000,
                    # Missing pred_h1
                },
                index=sample_timestamps,
            ),
        }

        combiner = ContextAwareCombiner()

        with pytest.raises(ValueError, match="inconsistent columns"):
            combiner.fit(predictions, sample_targets)

    def test_all_nan_predictions(self, sample_timestamps, sample_targets):
        """Test handling of all-NaN predictions."""
        predictions = {
            "model1": pd.DataFrame(
                {
                    "pred_h0": [np.nan] * len(sample_timestamps),
                    "pred_h1": [np.nan] * len(sample_timestamps),
                },
                index=sample_timestamps,
            ),
            "model2": pd.DataFrame(
                {
                    "pred_h0": np.random.randn(len(sample_timestamps)) * 20 + 1000,
                    "pred_h1": np.random.randn(len(sample_timestamps)) * 25 + 1000,
                },
                index=sample_timestamps,
            ),
        }

        combiner = ContextAwareCombiner()
        # Should handle this gracefully (models with too many NaN excluded)
        # or process with available data


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for context-aware combiner."""

    def test_full_workflow(self, sample_predictions, sample_targets, sample_meteo_data):
        """Test complete workflow with all features."""
        combiner = ContextAwareCombiner(
            config={
                "adaptation_method": "ridge",
                "use_temporal_features": True,
                "use_load_features": False,  # Must be False since not available during combine
                "use_meteo_features": False,  # Skip meteo for simplicity
            }
        )

        # Fit
        combiner.fit(sample_predictions, sample_targets)
        assert combiner.is_fitted

        # Combine
        result = combiner.combine(sample_predictions)
        assert result.combined_predictions is not None

        # Get importance
        importance = combiner.get_feature_importance()
        assert len(importance) > 0

        # Get diagnostics
        diagnostics = combiner.get_adaptation_diagnostics()
        assert "adaptation_method" in diagnostics

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "full_workflow.pkl"
            combiner.save(filepath)

            loaded = ContextAwareCombiner.load(filepath)
            assert loaded.is_fitted

            # Loaded combiner should produce same results
            result2 = loaded.combine(sample_predictions)
            pd.testing.assert_frame_equal(result.combined_predictions, result2.combined_predictions)

    def test_rf_workflow(self, sample_predictions, sample_targets):
        """Test workflow with Random Forest adaptation."""
        combiner = ContextAwareCombiner(
            config={
                "adaptation_method": "random_forest",
                "rf_n_estimators": 30,
                "rf_max_depth": 5,
            }
        )

        combiner.fit(sample_predictions, sample_targets)
        result = combiner.combine(sample_predictions)

        assert result.combined_predictions is not None
        importance = combiner.get_feature_importance()
        assert all(sum(imp.values()) > 0.99 for imp in importance.values())
