"""Markov Chain dynamic weighting combiner for model ensembles.

This module provides the MarkovChainCombiner class that uses Hidden Markov Models
(HMM) for dynamic weight adaptation based on model performance states.

The combiner:
- Tracks model performance over rolling windows
- Classifies performance into discrete states (EXCELLENT, GOOD, FAIR, POOR)
- Learns state transition probabilities using HMM
- Adapts combination weights dynamically based on current state
- Applies exponential smoothing to prevent erratic weight changes

Key Features:
- Performance state classification based on MAPE thresholds
- HMM-based transition probability learning via hmmlearn
- Rolling window performance tracking (default: 48 periods)
- Exponential weight smoothing for stability
- Context-aware state transitions (optional)
- Diagnostic tools for weight adaptation analysis

Example:
    ```python
    from src.models.combination import MarkovChainCombiner

    # Training phase - learn state transitions
    train_predictions = {
        "lgbm": train_lgbm_df,
        "rf": train_rf_df,
        "arima": train_arima_df
    }

    combiner = MarkovChainCombiner(config={
        'n_states': 4,
        'window_size': 48,
        'smoothing_factor': 0.3
    })
    combiner.fit(train_predictions, train_targets)

    # Inference phase - dynamic weighting
    result = combiner.combine(test_predictions)
    print(combiner.get_current_states())
    print(combiner.get_adaptation_diagnostics())
    ```
"""

import time
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.combination.base_combiner import BaseCombiner
from src.models.combination.data_structures import (
    CombinationResult,
    CombinerConfig,
)
from src.utils.logger import get_logger

# Optional import for hmmlearn
try:
    from hmmlearn import hmm

    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

# Ruff: Ignore base class interface requirements
# ruff: noqa: ARG002, PLC0415

logger = get_logger(__name__)

# Constants for validation
MIN_STATES = 2
MIN_WINDOW_SIZE = 2


class PerformanceState(Enum):
    """Performance state classifications based on MAPE thresholds.

    States represent qualitative performance levels:
    - EXCELLENT: MAPE < 2% (exceptional accuracy)
    - GOOD: MAPE 2-4% (satisfactory performance)
    - FAIR: MAPE 4-8% (acceptable but needs monitoring)
    - POOR: MAPE > 8% (underperforming, reduce weight)
    """

    EXCELLENT = 0
    GOOD = 1
    FAIR = 2
    POOR = 3


class PerformanceTracker:
    """Track model performance over time using rolling windows.

    Maintains a history of error values and performance states for
    calculating current performance metrics and state sequences.

    Attributes:
        window_size: Size of the rolling window for performance tracking.
        errors: Deque of recent error values (MAPE).
        states: Deque of recent performance states.
    """

    def __init__(self, window_size: int = 48) -> None:
        """Initialize performance tracker.

        Args:
            window_size: Size of rolling window for tracking performance.
                Default is 48 (one day of half-hourly data).
        """
        self.window_size = window_size
        self.errors: deque[float] = deque(maxlen=window_size)
        self.states: deque[PerformanceState] = deque(maxlen=window_size)

    def update(self, error: float, state: PerformanceState) -> None:
        """Update performance tracking with new observation.

        Args:
            error: Error value (MAPE) for this observation.
            state: Classified performance state.
        """
        self.errors.append(error)
        self.states.append(state)

    def get_current_performance(self) -> float:
        """Get current rolling average performance (MAPE).

        Returns:
            Mean of recent errors, or inf if no data.
        """
        if len(self.errors) == 0:
            return float("inf")
        return float(np.mean(list(self.errors)))

    def get_state_sequence(self) -> list[int]:
        """Get sequence of performance state values.

        Returns:
            List of integer state values for HMM training.
        """
        return [state.value for state in self.states]

    def reset(self) -> None:
        """Reset tracking history."""
        self.errors.clear()
        self.states.clear()


class MarkovChainCombiner(BaseCombiner):
    """Dynamic weight adaptation using Hidden Markov Models.

    This combiner implements state-based dynamic weighting where model
    performance is classified into discrete states (EXCELLENT, GOOD, FAIR, POOR),
    and HMMs are used to model state transitions and adapt weights accordingly.

    Performance States:
        - EXCELLENT: MAPE < 2%
        - GOOD: MAPE 2-4%
        - FAIR: MAPE 4-8%
        - POOR: MAPE > 8%

    State Transitions:
        P(state_t | state_t-1) learned from historical performance

    Weight Calculation:
        w_i = f(current_state_i) where f maps states to weights inversely
        proportional to expected error in that state

    Weight Smoothing:
        w_t = alpha * w_new + (1 - alpha) * w_prev

    Benefits:
        - Adapts to changing model performance patterns
        - Leverages temporal patterns in model accuracy
        - Smooth weight transitions prevent erratic behavior
        - Provides interpretable performance state information

    Attributes:
        n_states: Number of performance states (default 4).
        window_size: Rolling window size for performance tracking.
        smoothing_factor: Exponential smoothing factor (0=no change, 1=instant).
        state_thresholds: MAPE thresholds for state classification.
        hmm_n_iter: Maximum HMM training iterations.
        hmm_tol: HMM convergence tolerance.

    Example:
        >>> combiner = MarkovChainCombiner(config={
        ...     'n_states': 4,
        ...     'smoothing_factor': 0.3
        ... })
        >>> combiner.fit(train_predictions, train_targets)
        >>> result = combiner.combine(test_predictions)
        >>> print(combiner.get_current_states())
        {'lgbm': 'EXCELLENT', 'rf': 'GOOD', 'arima': 'FAIR'}
    """

    def __init__(self, config: CombinerConfig | dict[str, Any] | None = None) -> None:
        """Initialize Markov Chain combiner.

        Args:
            config: Configuration as CombinerConfig instance or dictionary.
                Supported custom_config options:
                - n_states: Number of performance states (default 4)
                - window_size: Rolling window size (default 48)
                - smoothing_factor: Weight smoothing factor 0-1 (default 0.3)
                - state_thresholds: Dict with 'excellent', 'good', 'fair' MAPE thresholds
                - hmm_n_iter: Max HMM iterations (default 100)
                - hmm_tol: HMM convergence tolerance (default 1e-4)

        Raises:
            ImportError: If hmmlearn is not installed.
            ValueError: If invalid configuration provided.

        Example:
            >>> combiner = MarkovChainCombiner()  # Default config
            >>> combiner = MarkovChainCombiner(config={
            ...     'n_states': 4,
            ...     'window_size': 96,  # Two days
            ...     'smoothing_factor': 0.5
            ... })
        """
        if not HMMLEARN_AVAILABLE:
            msg = (
                "hmmlearn is not installed. Install with: pip install hmmlearn. "
                "Or use a different combiner type."
            )
            raise ImportError(msg)

        # Initialize base combiner
        super().__init__(config=config)

        # Extract Markov Chain specific configuration from custom_config
        custom = self.config.custom_config

        # Default state thresholds (MAPE percentages)
        default_thresholds = {
            "excellent": 2.0,
            "good": 4.0,
            "fair": 8.0,
        }

        self.n_states = custom.get("n_states", 4)
        self.window_size = custom.get("window_size", 48)
        self.smoothing_factor = custom.get("smoothing_factor", 0.3)
        self.state_thresholds = custom.get("state_thresholds", default_thresholds)
        self.hmm_n_iter = custom.get("hmm_n_iter", 100)
        self.hmm_tol = custom.get("hmm_tol", 1e-4)

        # Validate configuration
        if not 0.0 <= self.smoothing_factor <= 1.0:
            msg = f"smoothing_factor must be between 0 and 1, got {self.smoothing_factor}"
            raise ValueError(msg)

        if self.n_states < MIN_STATES:
            msg = f"n_states must be >= {MIN_STATES}, got {self.n_states}"
            raise ValueError(msg)

        if self.window_size < MIN_WINDOW_SIZE:
            msg = f"window_size must be >= {MIN_WINDOW_SIZE}, got {self.window_size}"
            raise ValueError(msg)

        # Internal state for HMM models and tracking
        self._hmm_models: dict[str, Any] = {}  # {model_name: GaussianHMM}
        self._performance_trackers: dict[str, PerformanceTracker] = {}
        self._transition_matrices: dict[str, np.ndarray] = {}
        self._previous_weights: dict[str, float] | None = None
        self._state_weight_map: dict[PerformanceState, float] = {}

        # Initialize state weight mapping (better state = higher weight multiplier)
        self._initialize_state_weights()

        # Store configuration for persistence
        self.config.custom_config.update(
            {
                "n_states": self.n_states,
                "window_size": self.window_size,
                "smoothing_factor": self.smoothing_factor,
                "state_thresholds": self.state_thresholds,
                "hmm_n_iter": self.hmm_n_iter,
                "hmm_tol": self.hmm_tol,
            }
        )

        logger.debug(
            "Initialized MarkovChainCombiner with n_states=%d, window_size=%d, "
            "smoothing_factor=%.2f",
            self.n_states,
            self.window_size,
            self.smoothing_factor,
        )

    def _initialize_state_weights(self) -> None:
        """Initialize weight multipliers for each performance state.

        Better performance states get higher weight multipliers.
        """
        self._state_weight_map = {
            PerformanceState.EXCELLENT: 4.0,
            PerformanceState.GOOD: 2.0,
            PerformanceState.FAIR: 1.0,
            PerformanceState.POOR: 0.5,
        }

    @property
    def name(self) -> str:
        """Return combiner name.

        Returns:
            Name string identifying this combiner type.
        """
        return "markov_chain"

    @property
    def version(self) -> str:
        """Return combiner version.

        Returns:
            Version string following semantic versioning.
        """
        return "1.0.0"

    def _classify_performance(self, mape: float) -> PerformanceState:
        """Classify MAPE into performance state.

        Args:
            mape: Mean Absolute Percentage Error value.

        Returns:
            Classified performance state.

        Example:
            >>> combiner._classify_performance(1.5)
            PerformanceState.EXCELLENT
            >>> combiner._classify_performance(3.0)
            PerformanceState.GOOD
        """
        if mape < self.state_thresholds["excellent"]:
            return PerformanceState.EXCELLENT
        if mape < self.state_thresholds["good"]:
            return PerformanceState.GOOD
        if mape < self.state_thresholds["fair"]:
            return PerformanceState.FAIR
        return PerformanceState.POOR

    def _calculate_mape(
        self,
        predictions: pd.DataFrame,
        targets: pd.DataFrame,
        pred_cols: list[str],
        idx: int,
    ) -> float | None:
        """Calculate MAPE for a single observation across horizons.

        Args:
            predictions: Prediction DataFrame for one model.
            targets: Target DataFrame.
            pred_cols: List of prediction column names.
            idx: Row index to calculate MAPE for.

        Returns:
            Average MAPE across horizons, or None if no valid calculations.
        """
        errors = []
        for col in pred_cols:
            horizon = col.split("_h")[1]
            target_col = f"h{horizon}"

            if target_col in targets.columns:
                pred_val = predictions[col].iloc[idx]
                true_val = targets[target_col].iloc[idx]

                if not pd.isna(pred_val) and not pd.isna(true_val) and true_val != 0:
                    mape = np.abs((true_val - pred_val) / true_val) * 100
                    errors.append(mape)

        return float(np.mean(errors)) if errors else None

    def fit(
        self,
        train_predictions: dict[str, pd.DataFrame],
        train_targets: pd.DataFrame,
        validation_predictions: dict[str, pd.DataFrame] | None = None,
        validation_targets: pd.DataFrame | None = None,
    ) -> None:
        """Fit Markov Chain combiner by learning performance states and transitions.

        This method:
        1. Calculates error metrics for each model at each timestamp
        2. Classifies errors into performance states
        3. Builds state sequences for each model
        4. Trains HMM to model state transitions
        5. Stores transition matrices for dynamic weighting

        Args:
            train_predictions: Dictionary mapping model names to predictions.
                Each DataFrame should have columns like 'pred_h0', 'pred_h1', etc.
            train_targets: True target values for training period.
                Should have columns like 'h0', 'h1', etc.
            validation_predictions: Optional validation predictions (unused).
            validation_targets: Optional validation targets (unused).

        Raises:
            ValueError: If prediction format is invalid.

        Example:
            >>> combiner = MarkovChainCombiner()
            >>> combiner.fit(train_predictions, train_targets)
            >>> print(combiner.is_fitted)
            True
        """
        logger.info(
            "Fitting Markov Chain combiner with %d states, window_size=%d",
            self.n_states,
            self.window_size,
        )

        # Validate predictions
        self._validate_predictions(train_predictions)

        # Store model names
        self._model_names = sorted(train_predictions.keys())

        # Get prediction columns
        first_pred = next(iter(train_predictions.values()))
        pred_cols = sorted([col for col in first_pred.columns if col.startswith("pred_h")])

        if not pred_cols:
            msg = "No prediction columns found (expected pred_h0, pred_h1, etc.)"
            raise ValueError(msg)

        n_samples = len(first_pred)

        logger.info(
            "Training HMM on %d samples from %d models",
            n_samples,
            len(self._model_names),
        )

        # Initialize performance trackers
        for model_name in self._model_names:
            self._performance_trackers[model_name] = PerformanceTracker(
                window_size=self.window_size
            )

        # Build performance state sequences for each model
        performance_sequences: dict[str, list[int]] = {model: [] for model in self._model_names}

        for idx in range(n_samples):
            for model_name in self._model_names:
                mape = self._calculate_mape(
                    train_predictions[model_name],
                    train_targets,
                    pred_cols,
                    idx,
                )

                if mape is not None:
                    state = self._classify_performance(mape)
                    performance_sequences[model_name].append(state.value)
                    self._performance_trackers[model_name].update(mape, state)

        # Train HMM for each model
        for model_name in self._model_names:
            sequence = performance_sequences[model_name]

            if len(sequence) < self.n_states * 2:
                logger.warning(
                    "Insufficient data for HMM training for %s: %d samples (need at least %d)",
                    model_name,
                    len(sequence),
                    self.n_states * 2,
                )
                continue

            logger.info(
                "Training HMM for %s (%d state observations)", model_name, len(sequence)
            )

            # Prepare sequence for HMM (needs to be 2D)
            sequence_array = np.array(sequence).reshape(-1, 1).astype(float)

            # Create and train HMM
            model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=self.hmm_n_iter,
                tol=self.hmm_tol,
                random_state=42,
            )

            try:
                model.fit(sequence_array)
                self._hmm_models[model_name] = model
                self._transition_matrices[model_name] = model.transmat_.copy()

                logger.info(
                    "  %s HMM converged: %s",
                    model_name,
                    model.monitor_.converged,
                )
            except Exception as e:
                logger.warning(
                    "HMM training failed for %s: %s. Using fallback weights.",
                    model_name,
                    str(e),
                )

        # Calculate initial weights from current performance
        self._update_global_weights()

        # Mark as fitted
        self._fitted = True
        from datetime import UTC

        self._fit_timestamp = datetime.now(tz=UTC)

        logger.info(
            "Markov Chain combiner fitted successfully (%d HMM models trained)",
            len(self._hmm_models),
        )

    def _update_global_weights(self) -> None:
        """Update global weights based on current performance states."""
        if not self._performance_trackers:
            return

        weights = self._calculate_dynamic_weights_from_trackers()
        self._global_weights = weights

    def _calculate_dynamic_weights_from_trackers(self) -> dict[str, float]:
        """Calculate dynamic weights based on current performance states.

        Uses inverse relationship between error and weight, with state-based
        multipliers for additional differentiation.

        Returns:
            Dictionary of normalized weights for each model.
        """
        raw_weights = {}

        for model_name in self._model_names:
            if model_name in self._performance_trackers:
                tracker = self._performance_trackers[model_name]
                perf = tracker.get_current_performance()

                if np.isinf(perf):
                    # No data yet, use neutral weight
                    raw_weights[model_name] = 1.0
                else:
                    # Classify current state
                    state = self._classify_performance(perf)
                    state_multiplier = self._state_weight_map.get(state, 1.0)

                    # Weight inversely proportional to error, multiplied by state factor
                    raw_weights[model_name] = state_multiplier / (perf + 1.0)
            else:
                raw_weights[model_name] = 1.0

        # Normalize weights to sum to 1
        total = sum(raw_weights.values())
        if total > 0:
            weights = {m: w / total for m, w in raw_weights.items()}
        else:
            # Fallback to equal weights
            n = len(self._model_names)
            equal_weight = 1.0 / n
            weights = dict.fromkeys(self._model_names, equal_weight)

        return weights

    def _smooth_weights(
        self,
        new_weights: dict[str, float],
        prev_weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply exponential smoothing to weights.

        Prevents erratic weight changes by blending new weights with previous.

        Args:
            new_weights: Newly calculated weights.
            prev_weights: Previous weights.

        Returns:
            Smoothed and normalized weights.
        """
        smoothed = {}

        for model_name, w_new in new_weights.items():
            w_prev = prev_weights.get(model_name, w_new)

            # Exponential smoothing: w_t = alpha * w_new + (1 - alpha) * w_prev
            smoothed[model_name] = (
                self.smoothing_factor * w_new + (1 - self.smoothing_factor) * w_prev
            )

        # Renormalize to sum to 1
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {m: w / total for m, w in smoothed.items()}

        return smoothed

    def combine(
        self,
        predictions: dict[str, pd.DataFrame],
        metadata: dict[str, Any] | None = None,
    ) -> CombinationResult:
        """Combine predictions using dynamic Markov Chain weights.

        Uses current performance states to calculate dynamic weights,
        applies smoothing, and combines predictions.

        Args:
            predictions: Dictionary mapping model names to prediction DataFrames.
                Each DataFrame should have columns like 'pred_h0', 'pred_h1', etc.
            metadata: Optional metadata about predictions.

        Returns:
            CombinationResult with dynamically weighted predictions.

        Raises:
            RuntimeError: If combiner not fitted.
            ValueError: If prediction format is invalid.

        Example:
            >>> result = combiner.combine(test_predictions)
            >>> print(result.weights)
            {'lgbm': 0.45, 'rf': 0.30, 'arima': 0.25}
        """
        start_time = time.time()

        # Check if fitted
        self._check_is_fitted()

        logger.info(
            "Combining %d models using Markov Chain dynamic weights",
            len(predictions),
        )

        # Validate predictions
        self._validate_predictions(predictions)

        # Handle missing predictions
        cleaned_predictions, excluded_models = self._handle_missing_predictions(predictions)

        if excluded_models:
            logger.warning(
                "Excluded %d models due to missing data: %s",
                len(excluded_models),
                excluded_models,
            )

        # Get prediction columns
        first_pred = next(iter(cleaned_predictions.values()))
        pred_cols = sorted([col for col in first_pred.columns if col.startswith("pred_h")])

        # Calculate dynamic weights based on current performance
        dynamic_weights = self._calculate_dynamic_weights_from_trackers()

        # Filter weights for available models
        available_models = [m for m in self._model_names if m in cleaned_predictions]
        dynamic_weights = {m: dynamic_weights.get(m, 0.0) for m in available_models}

        # Renormalize if some models were excluded
        total = sum(dynamic_weights.values())
        if total > 0:
            dynamic_weights = {m: w / total for m, w in dynamic_weights.items()}
        else:
            # Fallback to equal weights
            n = len(available_models)
            equal_weight = 1.0 / n
            dynamic_weights = dict.fromkeys(available_models, equal_weight)

        # Apply weight smoothing if we have previous weights
        if self._previous_weights is not None:
            dynamic_weights = self._smooth_weights(dynamic_weights, self._previous_weights)

        self._previous_weights = dynamic_weights.copy()

        # Combine predictions
        combined_df = pd.DataFrame(index=first_pred.index)

        for col in pred_cols:
            weighted_sum = np.zeros(len(first_pred))

            for model_name, weight in dynamic_weights.items():
                if model_name in cleaned_predictions:
                    model_values = cleaned_predictions[model_name][col].to_numpy()
                    # Handle NaN values
                    model_values = np.nan_to_num(model_values, nan=0.0)
                    weighted_sum += weight * model_values

            combined_df[col] = weighted_sum

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create metadata
        metadata_obj = self._create_metadata(
            models_used=list(cleaned_predictions.keys()),
            models_excluded=excluded_models,
            processing_time=processing_time,
        )

        # Add Markov Chain specific metadata
        metadata_obj.configuration["n_states"] = self.n_states
        metadata_obj.configuration["window_size"] = self.window_size
        metadata_obj.configuration["smoothing_factor"] = self.smoothing_factor
        metadata_obj.configuration["current_states"] = self.get_current_states()

        # Add performance metrics
        metadata_obj.performance_metrics["weights_entropy"] = float(
            -sum(w * np.log(w + 1e-10) for w in dynamic_weights.values())
        )

        # Create result
        result = CombinationResult(
            combined_predictions=combined_df,
            weights=dynamic_weights,
            metadata=metadata_obj,
            model_contributions=None,
            confidence_intervals=None,
        )

        logger.info(
            "Combined %d models using %s (processing time: %.3fs)",
            len(cleaned_predictions),
            self.name,
            processing_time,
        )

        return result

    def update_performance(
        self,
        predictions: dict[str, pd.DataFrame],
        targets: pd.DataFrame,
    ) -> None:
        """Update performance tracking with new observations.

        Call this method during inference to update the performance
        trackers with actual observed errors.

        Args:
            predictions: Model predictions for recent period.
            targets: Actual target values for the same period.

        Example:
            >>> # After getting actual values
            >>> combiner.update_performance(recent_predictions, actual_values)
            >>> print(combiner.get_current_states())
        """
        if not self._fitted:
            logger.warning("Combiner not fitted. Cannot update performance.")
            return

        first_pred = next(iter(predictions.values()))
        pred_cols = sorted([col for col in first_pred.columns if col.startswith("pred_h")])

        for idx in range(len(first_pred)):
            for model_name in self._model_names:
                if model_name not in predictions:
                    continue

                mape = self._calculate_mape(predictions[model_name], targets, pred_cols, idx)

                if mape is not None and model_name in self._performance_trackers:
                    state = self._classify_performance(mape)
                    self._performance_trackers[model_name].update(mape, state)

        logger.debug("Updated performance tracking with %d observations", len(first_pred))

    def get_current_states(self) -> dict[str, str]:
        """Get current performance states for all models.

        Returns:
            Dictionary mapping model names to state names.

        Example:
            >>> combiner.get_current_states()
            {'lgbm': 'EXCELLENT', 'rf': 'GOOD', 'arima': 'FAIR'}
        """
        states = {}

        for model_name, tracker in self._performance_trackers.items():
            perf = tracker.get_current_performance()
            if np.isinf(perf):
                states[model_name] = "UNKNOWN"
            else:
                state = self._classify_performance(perf)
                states[model_name] = state.name

        return states

    def get_adaptation_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information about weight adaptation.

        Returns comprehensive information about the current state of
        the Markov Chain combiner for debugging and analysis.

        Returns:
            Dictionary with diagnostic information including:
            - n_states: Number of performance states
            - window_size: Rolling window size
            - smoothing_factor: Weight smoothing factor
            - current_states: Current state for each model
            - current_weights: Current combination weights
            - current_mapes: Current rolling MAPE for each model
            - transition_matrices: Learned transition matrices
            - hmm_models_trained: List of models with trained HMMs

        Example:
            >>> diagnostics = combiner.get_adaptation_diagnostics()
            >>> print(diagnostics['current_states'])
            >>> print(diagnostics['transition_matrices']['lgbm'])
        """
        if not self._fitted:
            return {}

        diagnostics: dict[str, Any] = {
            "n_states": self.n_states,
            "window_size": self.window_size,
            "smoothing_factor": self.smoothing_factor,
            "state_thresholds": self.state_thresholds.copy(),
            "current_states": self.get_current_states(),
            "current_weights": self._previous_weights.copy() if self._previous_weights else {},
            "current_mapes": {},
            "transition_matrices": {},
            "hmm_models_trained": list(self._hmm_models.keys()),
        }

        # Add current MAPE values
        for model_name, tracker in self._performance_trackers.items():
            perf = tracker.get_current_performance()
            diagnostics["current_mapes"][model_name] = perf if not np.isinf(perf) else None

        # Add transition matrices (as lists for JSON serialization)
        for model_name, trans_mat in self._transition_matrices.items():
            diagnostics["transition_matrices"][model_name] = trans_mat.tolist()

        return diagnostics

    def predict_next_states(self) -> dict[str, dict[str, float]]:
        """Predict probability distribution of next states for each model.

        Uses the trained HMMs to predict the most likely next state
        given current performance patterns.

        Returns:
            Dictionary mapping model names to state probability distributions.

        Example:
            >>> next_states = combiner.predict_next_states()
            >>> print(next_states['lgbm'])
            {'EXCELLENT': 0.3, 'GOOD': 0.4, 'FAIR': 0.2, 'POOR': 0.1}
        """
        if not self._fitted:
            return {}

        predictions = {}

        for model_name in self._model_names:
            if model_name not in self._hmm_models:
                continue

            hmm_model = self._hmm_models[model_name]
            tracker = self._performance_trackers.get(model_name)

            if tracker is None or len(tracker.states) == 0:
                continue

            # Get current state sequence
            state_seq = np.array(tracker.get_state_sequence()).reshape(-1, 1).astype(float)

            # Get posterior probabilities for last observation
            try:
                posteriors = hmm_model.predict_proba(state_seq)
                last_posterior = posteriors[-1]

                # Map to state names
                state_probs = {}
                for state in PerformanceState:
                    if state.value < len(last_posterior):
                        state_probs[state.name] = float(last_posterior[state.value])
                    else:
                        state_probs[state.name] = 0.0

                predictions[model_name] = state_probs
            except Exception as e:
                logger.warning(
                    "Failed to predict next state for %s: %s",
                    model_name,
                    str(e),
                )

        return predictions

    def save(self, filepath: str | Path) -> None:
        """Save combiner state to file.

        Serializes the combiner's HMM models, weights, configuration,
        and performance trackers for later loading.

        Args:
            filepath: Path to save file (typically .pkl or .joblib).

        Example:
            >>> combiner.save("combiners/markov_chain_v1.pkl")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Serialize performance trackers
        tracker_states = {}
        for model_name, tracker in self._performance_trackers.items():
            tracker_states[model_name] = {
                "errors": list(tracker.errors),
                "states": [s.value for s in tracker.states],
                "window_size": tracker.window_size,
            }

        state = {
            "name": self.name,
            "version": self.version,
            "config": self.config.to_dict(),
            "weights": self._weights.copy(),
            "global_weights": self._global_weights.copy(),
            "fitted": self._fitted,
            "fit_timestamp": self._fit_timestamp.isoformat() if self._fit_timestamp else None,
            "performance_history": self._performance_history.copy(),
            "model_names": self._model_names.copy(),
            # Markov Chain specific state
            "hmm_models": self._hmm_models,  # hmmlearn models are picklable
            "transition_matrices": {k: v.tolist() for k, v in self._transition_matrices.items()},
            "tracker_states": tracker_states,
            "previous_weights": self._previous_weights,
            "n_states": self.n_states,
            "window_size": self.window_size,
            "smoothing_factor": self.smoothing_factor,
            "state_thresholds": self.state_thresholds,
            "hmm_n_iter": self.hmm_n_iter,
            "hmm_tol": self.hmm_tol,
        }

        joblib.dump(state, filepath)
        logger.info("Saved %s combiner to %s", self.name, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "MarkovChainCombiner":
        """Load combiner from file.

        Deserializes a previously saved combiner state, including HMM models
        and performance trackers.

        Args:
            filepath: Path to load file.

        Returns:
            Loaded MarkovChainCombiner instance.

        Example:
            >>> combiner = MarkovChainCombiner.load("combiners/markov_chain_v1.pkl")
            >>> result = combiner.combine(test_predictions)
        """
        filepath = Path(filepath)
        state = joblib.load(filepath)

        # Create instance with saved config
        instance = cls(config=state["config"])

        # Restore base combiner state
        instance._weights = state["weights"]
        instance._global_weights = state["global_weights"]
        instance._fitted = state["fitted"]
        instance._fit_timestamp = (
            datetime.fromisoformat(state["fit_timestamp"]) if state["fit_timestamp"] else None
        )
        instance._performance_history = state["performance_history"]
        instance._model_names = state["model_names"]

        # Restore Markov Chain specific state
        instance._hmm_models = state.get("hmm_models", {})
        instance._transition_matrices = {
            k: np.array(v) for k, v in state.get("transition_matrices", {}).items()
        }
        instance._previous_weights = state.get("previous_weights")

        # Restore configuration
        instance.n_states = state.get("n_states", 4)
        instance.window_size = state.get("window_size", 48)
        instance.smoothing_factor = state.get("smoothing_factor", 0.3)
        instance.state_thresholds = state.get(
            "state_thresholds",
            {"excellent": 2.0, "good": 4.0, "fair": 8.0},
        )
        instance.hmm_n_iter = state.get("hmm_n_iter", 100)
        instance.hmm_tol = state.get("hmm_tol", 1e-4)

        # Restore performance trackers
        tracker_states = state.get("tracker_states", {})
        for model_name, tracker_data in tracker_states.items():
            tracker = PerformanceTracker(window_size=tracker_data["window_size"])
            for error, state_val in zip(
                tracker_data["errors"], tracker_data["states"], strict=True
            ):
                tracker.errors.append(error)
                tracker.states.append(PerformanceState(state_val))
            instance._performance_trackers[model_name] = tracker

        logger.info("Loaded %s combiner from %s", instance.name, filepath)

        return instance

    def reset(self) -> None:
        """Reset combiner to unfitted state.

        Clears all learned weights, HMM models, and performance trackers.
        """
        # Call parent reset
        super().reset()

        # Reset Markov Chain specific state
        self._hmm_models = {}
        self._performance_trackers = {}
        self._transition_matrices = {}
        self._previous_weights = None

        logger.info("Reset %s combiner (including HMM models) to unfitted state", self.name)
