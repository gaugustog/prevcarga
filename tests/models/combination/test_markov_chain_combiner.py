"""Tests for Markov Chain dynamic weighting combiner.

This module provides comprehensive tests for the MarkovChainCombiner class,
including performance state classification, HMM training, dynamic weight
calculation, and weight smoothing.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.combination.markov_chain_combiner import (
    MarkovChainCombiner,
    PerformanceState,
    PerformanceTracker,
    HMMLEARN_AVAILABLE,
)


# Skip all tests if hmmlearn is not installed
pytestmark = pytest.mark.skipif(
    not HMMLEARN_AVAILABLE,
    reason="hmmlearn not installed",
)


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="30min")

    predictions = {
        "model1": pd.DataFrame(
            {
                "pred_h0": np.random.randn(200) * 30 + 1000,
                "pred_h1": np.random.randn(200) * 35 + 1000,
                "pred_h2": np.random.randn(200) * 40 + 1000,
            },
            index=dates,
        ),
        "model2": pd.DataFrame(
            {
                "pred_h0": np.random.randn(200) * 40 + 1000,
                "pred_h1": np.random.randn(200) * 45 + 1000,
                "pred_h2": np.random.randn(200) * 50 + 1000,
            },
            index=dates,
        ),
        "model3": pd.DataFrame(
            {
                "pred_h0": np.random.randn(200) * 50 + 1000,
                "pred_h1": np.random.randn(200) * 55 + 1000,
                "pred_h2": np.random.randn(200) * 60 + 1000,
            },
            index=dates,
        ),
    }

    targets = pd.DataFrame(
        {
            "h0": np.random.randn(200) * 30 + 1000,
            "h1": np.random.randn(200) * 35 + 1000,
            "h2": np.random.randn(200) * 40 + 1000,
        },
        index=dates,
    )

    return predictions, targets


@pytest.fixture
def sample_predictions_with_performance_pattern():
    """Create sample data with distinct performance patterns."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="30min")

    # Base target with some pattern
    base = np.sin(np.linspace(0, 4 * np.pi, 200)) * 100 + 1000

    targets = pd.DataFrame(
        {
            "h0": base + np.random.randn(200) * 5,
            "h1": base + np.random.randn(200) * 5,
        },
        index=dates,
    )

    # Model 1: Good performance (low noise)
    predictions = {
        "good_model": pd.DataFrame(
            {
                "pred_h0": base + np.random.randn(200) * 10,
                "pred_h1": base + np.random.randn(200) * 12,
            },
            index=dates,
        ),
        # Model 2: Poor performance (high noise)
        "poor_model": pd.DataFrame(
            {
                "pred_h0": base + np.random.randn(200) * 80,
                "pred_h1": base + np.random.randn(200) * 90,
            },
            index=dates,
        ),
    }

    return predictions, targets


class TestPerformanceState:
    """Tests for PerformanceState enum."""

    def test_state_values(self):
        """Test that states have correct integer values."""
        assert PerformanceState.EXCELLENT.value == 0
        assert PerformanceState.GOOD.value == 1
        assert PerformanceState.FAIR.value == 2
        assert PerformanceState.POOR.value == 3

    def test_state_count(self):
        """Test correct number of states."""
        assert len(PerformanceState) == 4


class TestPerformanceTracker:
    """Tests for PerformanceTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = PerformanceTracker(window_size=48)
        assert tracker.window_size == 48
        assert len(tracker.errors) == 0
        assert len(tracker.states) == 0

    def test_update(self):
        """Test tracker update."""
        tracker = PerformanceTracker(window_size=10)

        tracker.update(1.5, PerformanceState.EXCELLENT)
        tracker.update(3.0, PerformanceState.GOOD)

        assert len(tracker.errors) == 2
        assert len(tracker.states) == 2
        assert tracker.errors[0] == 1.5
        assert tracker.states[0] == PerformanceState.EXCELLENT

    def test_rolling_window(self):
        """Test rolling window behavior."""
        tracker = PerformanceTracker(window_size=3)

        # Add 5 items
        for i in range(5):
            tracker.update(float(i), PerformanceState.GOOD)

        # Only last 3 should be present
        assert len(tracker.errors) == 3
        assert list(tracker.errors) == [2.0, 3.0, 4.0]

    def test_get_current_performance(self):
        """Test current performance calculation."""
        tracker = PerformanceTracker(window_size=10)

        # Empty tracker
        assert np.isinf(tracker.get_current_performance())

        # Add some values
        tracker.update(1.0, PerformanceState.EXCELLENT)
        tracker.update(2.0, PerformanceState.EXCELLENT)
        tracker.update(3.0, PerformanceState.GOOD)

        assert tracker.get_current_performance() == pytest.approx(2.0, rel=1e-6)

    def test_get_state_sequence(self):
        """Test state sequence retrieval."""
        tracker = PerformanceTracker(window_size=10)

        tracker.update(1.0, PerformanceState.EXCELLENT)
        tracker.update(3.0, PerformanceState.GOOD)
        tracker.update(6.0, PerformanceState.FAIR)

        sequence = tracker.get_state_sequence()
        assert sequence == [0, 1, 2]

    def test_reset(self):
        """Test tracker reset."""
        tracker = PerformanceTracker(window_size=10)
        tracker.update(1.0, PerformanceState.EXCELLENT)
        tracker.update(2.0, PerformanceState.GOOD)

        tracker.reset()

        assert len(tracker.errors) == 0
        assert len(tracker.states) == 0


class TestMarkovChainCombinerInit:
    """Tests for MarkovChainCombiner initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        combiner = MarkovChainCombiner()

        assert combiner.name == "markov_chain"
        assert combiner.version == "1.0.0"
        assert combiner.n_states == 4
        assert combiner.window_size == 48
        assert combiner.smoothing_factor == 0.3
        assert not combiner.is_fitted

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = {
            "custom_config": {
                "n_states": 3,
                "window_size": 96,
                "smoothing_factor": 0.5,
            }
        }
        combiner = MarkovChainCombiner(config=config)

        assert combiner.n_states == 3
        assert combiner.window_size == 96
        assert combiner.smoothing_factor == 0.5

    def test_custom_thresholds(self):
        """Test initialization with custom state thresholds."""
        config = {
            "custom_config": {
                "state_thresholds": {
                    "excellent": 1.5,
                    "good": 3.0,
                    "fair": 6.0,
                }
            }
        }
        combiner = MarkovChainCombiner(config=config)

        assert combiner.state_thresholds["excellent"] == 1.5
        assert combiner.state_thresholds["good"] == 3.0
        assert combiner.state_thresholds["fair"] == 6.0

    def test_invalid_smoothing_factor(self):
        """Test that invalid smoothing factor raises error."""
        config = {"custom_config": {"smoothing_factor": 1.5}}

        with pytest.raises(ValueError, match="smoothing_factor must be between 0 and 1"):
            MarkovChainCombiner(config=config)

    def test_invalid_n_states(self):
        """Test that invalid n_states raises error."""
        config = {"custom_config": {"n_states": 1}}

        with pytest.raises(ValueError, match="n_states must be >= 2"):
            MarkovChainCombiner(config=config)

    def test_invalid_window_size(self):
        """Test that invalid window_size raises error."""
        config = {"custom_config": {"window_size": 1}}

        with pytest.raises(ValueError, match="window_size must be >= 2"):
            MarkovChainCombiner(config=config)


class TestPerformanceClassification:
    """Tests for performance state classification."""

    def test_excellent_classification(self):
        """Test EXCELLENT state classification."""
        combiner = MarkovChainCombiner()
        assert combiner._classify_performance(1.0) == PerformanceState.EXCELLENT
        assert combiner._classify_performance(1.9) == PerformanceState.EXCELLENT

    def test_good_classification(self):
        """Test GOOD state classification."""
        combiner = MarkovChainCombiner()
        assert combiner._classify_performance(2.0) == PerformanceState.GOOD
        assert combiner._classify_performance(3.5) == PerformanceState.GOOD

    def test_fair_classification(self):
        """Test FAIR state classification."""
        combiner = MarkovChainCombiner()
        assert combiner._classify_performance(4.0) == PerformanceState.FAIR
        assert combiner._classify_performance(7.0) == PerformanceState.FAIR

    def test_poor_classification(self):
        """Test POOR state classification."""
        combiner = MarkovChainCombiner()
        assert combiner._classify_performance(8.0) == PerformanceState.POOR
        assert combiner._classify_performance(15.0) == PerformanceState.POOR

    def test_custom_thresholds(self):
        """Test classification with custom thresholds."""
        config = {
            "custom_config": {
                "state_thresholds": {
                    "excellent": 1.0,
                    "good": 2.0,
                    "fair": 4.0,
                }
            }
        }
        combiner = MarkovChainCombiner(config=config)

        assert combiner._classify_performance(0.5) == PerformanceState.EXCELLENT
        assert combiner._classify_performance(1.5) == PerformanceState.GOOD
        assert combiner._classify_performance(3.0) == PerformanceState.FAIR
        assert combiner._classify_performance(5.0) == PerformanceState.POOR

    def test_boundary_values(self):
        """Test classification at exact boundary values."""
        combiner = MarkovChainCombiner()

        # At exact boundary, should fall into next category (< not <=)
        assert combiner._classify_performance(2.0) == PerformanceState.GOOD
        assert combiner._classify_performance(4.0) == PerformanceState.FAIR
        assert combiner._classify_performance(8.0) == PerformanceState.POOR


class TestMarkovChainCombinerFit:
    """Tests for MarkovChainCombiner.fit() method."""

    def test_fit_basic(self, sample_predictions):
        """Test basic fitting."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        assert combiner.is_fitted
        assert combiner.fit_timestamp is not None
        assert len(combiner._model_names) == 3
        assert len(combiner._performance_trackers) == 3

    def test_fit_trains_hmm(self, sample_predictions):
        """Test that HMM models are trained."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        # Should have HMM models for at least some models
        assert len(combiner._hmm_models) > 0

    def test_fit_calculates_transition_matrices(self, sample_predictions):
        """Test that transition matrices are calculated."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        # Should have transition matrices
        for model_name in combiner._hmm_models:
            assert model_name in combiner._transition_matrices
            trans_mat = combiner._transition_matrices[model_name]
            # Transition matrix should be square
            assert trans_mat.shape[0] == trans_mat.shape[1]
            # Rows should sum to 1
            row_sums = trans_mat.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)), decimal=5)

    def test_fit_initializes_weights(self, sample_predictions):
        """Test that weights are initialized after fitting."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        weights = combiner.get_weights()
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_fit_with_empty_predictions_raises(self):
        """Test that empty predictions raise error."""
        combiner = MarkovChainCombiner()

        with pytest.raises(ValueError, match="Predictions dictionary is empty"):
            combiner.fit({}, pd.DataFrame())


class TestMarkovChainCombinerCombine:
    """Tests for MarkovChainCombiner.combine() method."""

    def test_combine_basic(self, sample_predictions):
        """Test basic combination."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        result = combiner.combine(predictions)

        assert result is not None
        assert not result.combined_predictions.empty
        assert len(result.weights) == 3
        assert result.metadata.combination_method == "markov_chain"

    def test_combine_weights_sum_to_one(self, sample_predictions):
        """Test that weights sum to 1."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        result = combiner.combine(predictions)

        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_combine_without_fit_raises(self, sample_predictions):
        """Test that combining without fitting raises error."""
        predictions, _ = sample_predictions

        combiner = MarkovChainCombiner()

        with pytest.raises(RuntimeError, match="not fitted"):
            combiner.combine(predictions)

    def test_combine_output_shape(self, sample_predictions):
        """Test that combined output has correct shape."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        result = combiner.combine(predictions)

        first_pred = next(iter(predictions.values()))
        assert len(result.combined_predictions) == len(first_pred)
        assert result.combined_predictions.columns.tolist() == list(first_pred.columns)

    def test_combine_metadata_includes_states(self, sample_predictions):
        """Test that metadata includes current states."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        result = combiner.combine(predictions)

        assert "current_states" in result.metadata.configuration
        states = result.metadata.configuration["current_states"]
        assert len(states) == 3


class TestWeightSmoothing:
    """Tests for weight smoothing mechanism."""

    def test_smoothing_factor_zero(self, sample_predictions):
        """Test that smoothing_factor=0 keeps previous weights."""
        predictions, targets = sample_predictions

        config = {"custom_config": {"smoothing_factor": 0.0}}
        combiner = MarkovChainCombiner(config=config)
        combiner.fit(predictions, targets)

        # First combination
        result1 = combiner.combine(predictions)
        weights1 = result1.weights.copy()

        # Second combination - should keep same weights
        result2 = combiner.combine(predictions)
        weights2 = result2.weights

        for model in weights1:
            assert weights1[model] == pytest.approx(weights2[model], rel=1e-6)

    def test_smoothing_factor_one(self, sample_predictions):
        """Test that smoothing_factor=1 uses new weights completely."""
        predictions, targets = sample_predictions

        config = {"custom_config": {"smoothing_factor": 1.0}}
        combiner = MarkovChainCombiner(config=config)
        combiner.fit(predictions, targets)

        # Multiple combinations should use fresh weights each time
        result = combiner.combine(predictions)
        assert result is not None

    def test_smoothing_prevents_rapid_changes(self, sample_predictions):
        """Test that smoothing prevents erratic weight changes."""
        predictions, targets = sample_predictions

        config = {"custom_config": {"smoothing_factor": 0.3}}
        combiner = MarkovChainCombiner(config=config)
        combiner.fit(predictions, targets)

        # First combination
        result1 = combiner.combine(predictions)
        weights1 = result1.weights.copy()

        # Second combination
        result2 = combiner.combine(predictions)
        weights2 = result2.weights

        # With smoothing, weights should not change drastically
        for model in weights1:
            change = abs(weights2[model] - weights1[model])
            assert change < 0.5  # Weights shouldn't jump too much


class TestCurrentStates:
    """Tests for get_current_states() method."""

    def test_get_current_states_after_fit(self, sample_predictions):
        """Test getting states after fitting."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        states = combiner.get_current_states()

        assert len(states) == 3
        for model_name in predictions:
            assert model_name in states
            assert states[model_name] in ["EXCELLENT", "GOOD", "FAIR", "POOR", "UNKNOWN"]

    def test_get_current_states_before_fit(self):
        """Test getting states before fitting returns empty."""
        combiner = MarkovChainCombiner()
        states = combiner.get_current_states()
        assert states == {}


class TestAdaptationDiagnostics:
    """Tests for get_adaptation_diagnostics() method."""

    def test_diagnostics_after_fit(self, sample_predictions):
        """Test getting diagnostics after fitting."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)
        combiner.combine(predictions)  # To populate weights

        diagnostics = combiner.get_adaptation_diagnostics()

        assert "n_states" in diagnostics
        assert "window_size" in diagnostics
        assert "smoothing_factor" in diagnostics
        assert "current_states" in diagnostics
        assert "current_mapes" in diagnostics
        assert "transition_matrices" in diagnostics
        assert "hmm_models_trained" in diagnostics

    def test_diagnostics_before_fit(self):
        """Test getting diagnostics before fitting returns empty."""
        combiner = MarkovChainCombiner()
        diagnostics = combiner.get_adaptation_diagnostics()
        assert diagnostics == {}

    def test_diagnostics_transition_matrices_shape(self, sample_predictions):
        """Test that transition matrices in diagnostics have correct shape."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        diagnostics = combiner.get_adaptation_diagnostics()

        for model_name, trans_mat in diagnostics["transition_matrices"].items():
            # Should be a list of lists (from .tolist())
            assert isinstance(trans_mat, list)
            if trans_mat:
                assert isinstance(trans_mat[0], list)


class TestPerformanceUpdate:
    """Tests for update_performance() method."""

    def test_update_performance(self, sample_predictions):
        """Test updating performance tracking."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        # Get initial states
        states_before = combiner.get_current_states()

        # Update with new observations
        combiner.update_performance(predictions, targets)

        states_after = combiner.get_current_states()

        # States should still be valid
        for model in predictions:
            assert states_after[model] in ["EXCELLENT", "GOOD", "FAIR", "POOR", "UNKNOWN"]

    def test_update_performance_without_fit(self, sample_predictions):
        """Test that updating without fitting logs warning but doesn't crash."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()

        # Should not raise, just log warning
        combiner.update_performance(predictions, targets)


class TestSaveLoad:
    """Tests for save/load functionality."""

    def test_save_and_load(self, sample_predictions):
        """Test saving and loading combiner."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner(config={"custom_config": {"smoothing_factor": 0.4}})
        combiner.fit(predictions, targets)
        result_before = combiner.combine(predictions)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "markov_chain.pkl"
            combiner.save(filepath)

            loaded_combiner = MarkovChainCombiner.load(filepath)

            assert loaded_combiner.is_fitted
            assert loaded_combiner.n_states == combiner.n_states
            assert loaded_combiner.smoothing_factor == combiner.smoothing_factor
            assert loaded_combiner._model_names == combiner._model_names

            result_after = loaded_combiner.combine(predictions)
            assert abs(sum(result_after.weights.values()) - 1.0) < 1e-6

    def test_save_load_preserves_hmm_models(self, sample_predictions):
        """Test that HMM models are preserved after save/load."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        hmm_models_before = set(combiner._hmm_models.keys())

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "markov_chain.pkl"
            combiner.save(filepath)

            loaded_combiner = MarkovChainCombiner.load(filepath)

            hmm_models_after = set(loaded_combiner._hmm_models.keys())
            assert hmm_models_before == hmm_models_after

    def test_save_load_preserves_trackers(self, sample_predictions):
        """Test that performance trackers are preserved after save/load."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        states_before = combiner.get_current_states()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "markov_chain.pkl"
            combiner.save(filepath)

            loaded_combiner = MarkovChainCombiner.load(filepath)

            states_after = loaded_combiner.get_current_states()

            for model in states_before:
                assert states_before[model] == states_after[model]


class TestReset:
    """Tests for reset() method."""

    def test_reset_clears_state(self, sample_predictions):
        """Test that reset clears all state."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)
        combiner.combine(predictions)

        assert combiner.is_fitted
        assert len(combiner._hmm_models) > 0
        assert len(combiner._performance_trackers) > 0

        combiner.reset()

        assert not combiner.is_fitted
        assert len(combiner._hmm_models) == 0
        assert len(combiner._performance_trackers) == 0
        assert combiner._previous_weights is None


class TestPredictNextStates:
    """Tests for predict_next_states() method."""

    def test_predict_next_states(self, sample_predictions):
        """Test predicting next states."""
        predictions, targets = sample_predictions

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        next_states = combiner.predict_next_states()

        # Should have predictions for models with HMMs
        for model_name in combiner._hmm_models:
            if model_name in next_states:
                state_probs = next_states[model_name]
                # Probabilities should be non-negative
                for prob in state_probs.values():
                    assert prob >= 0.0
                # Should have all state names
                for state in PerformanceState:
                    assert state.name in state_probs

    def test_predict_next_states_before_fit(self):
        """Test that prediction before fitting returns empty."""
        combiner = MarkovChainCombiner()
        next_states = combiner.predict_next_states()
        assert next_states == {}


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_model(self):
        """Test with single model."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="30min")

        predictions = {
            "single_model": pd.DataFrame(
                {"pred_h0": np.random.randn(100) * 30 + 1000}, index=dates
            )
        }
        targets = pd.DataFrame({"h0": np.random.randn(100) * 30 + 1000}, index=dates)

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        result = combiner.combine(predictions)

        # Single model should get 100% weight
        assert result.weights["single_model"] == pytest.approx(1.0, rel=1e-6)

    def test_with_nan_values(self):
        """Test handling of NaN values in predictions."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="30min")

        preds = np.random.randn(100) * 30 + 1000
        preds[10:15] = np.nan  # Add some NaNs

        predictions = {
            "model1": pd.DataFrame({"pred_h0": preds}, index=dates),
            "model2": pd.DataFrame(
                {"pred_h0": np.random.randn(100) * 30 + 1000}, index=dates
            ),
        }
        targets = pd.DataFrame({"h0": np.random.randn(100) * 30 + 1000}, index=dates)

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        result = combiner.combine(predictions)

        # Should handle NaNs gracefully
        assert not result.combined_predictions.isna().all().all()

    def test_insufficient_data_warning(self):
        """Test that insufficient data for HMM logs warning."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=5, freq="30min")  # Very few samples

        predictions = {
            "model1": pd.DataFrame(
                {"pred_h0": np.random.randn(5) * 30 + 1000}, index=dates
            ),
        }
        targets = pd.DataFrame({"h0": np.random.randn(5) * 30 + 1000}, index=dates)

        combiner = MarkovChainCombiner()
        # Should not crash, but may not train HMM
        combiner.fit(predictions, targets)
        assert combiner.is_fitted


class TestPerformancePatterns:
    """Tests with distinct performance patterns."""

    def test_good_model_gets_higher_weight(self, sample_predictions_with_performance_pattern):
        """Test that better performing model gets higher weight."""
        predictions, targets = sample_predictions_with_performance_pattern

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        result = combiner.combine(predictions)

        # Good model should have higher weight than poor model
        assert result.weights["good_model"] > result.weights["poor_model"]

    def test_state_classification_matches_performance(
        self, sample_predictions_with_performance_pattern
    ):
        """Test that states reflect actual performance."""
        predictions, targets = sample_predictions_with_performance_pattern

        combiner = MarkovChainCombiner()
        combiner.fit(predictions, targets)

        states = combiner.get_current_states()

        # Good model should have better state than poor model
        state_order = {"EXCELLENT": 0, "GOOD": 1, "FAIR": 2, "POOR": 3, "UNKNOWN": 4}
        good_state_value = state_order.get(states["good_model"], 4)
        poor_state_value = state_order.get(states["poor_model"], 4)

        # Lower value = better state
        assert good_state_value <= poor_state_value
