"""Simple and weighted averaging combiner for model ensembles.

This module provides the SimpleAveragingCombiner class that combines predictions
from multiple forecasting models using simple averaging (equal weights) or
weighted averaging (custom weights).

The combiner supports:
- Simple averaging: All models have equal weights (1/n)
- Weighted averaging: Custom weights specified per model
- Automatic weight normalization
- Graceful handling of missing values (NaN)
- Weight validation and persistence
- Optional weight optimization based on historical performance

Example:
    ```python
    from src.models.combination import SimpleAveragingCombiner, CombinerConfig

    # Simple averaging (equal weights)
    combiner = SimpleAveragingCombiner()
    combiner.fit(train_predictions, train_targets)
    result = combiner.combine(test_predictions)

    # Weighted averaging with custom weights
    config = CombinerConfig(require_fit=False)
    combiner = SimpleAveragingCombiner(config=config, weighted=True)
    weights = {"lgbm": 0.5, "rf": 0.3, "xgb": 0.2}
    result = combiner.combine(test_predictions, weights=weights)
    ```
"""

import time
from datetime import datetime
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

logger = get_logger(__name__)


class SimpleAveragingCombiner(BaseCombiner):
    """Combiner that uses simple or weighted averaging of model predictions.

    This combiner implements two averaging strategies:

    1. Simple Averaging: All models receive equal weights (1/n where n is number of models).
       No fitting is required - weights are always uniform.

    2. Weighted Averaging: Custom weights are provided for each model. Weights must
       sum to 1.0 and can be specified at combination time or learned during fitting.

    The combiner handles missing values (NaN) gracefully by:
    - Computing average only over valid (non-NaN) predictions
    - Adjusting weights proportionally for valid values
    - Excluding models with excessive missing data (based on config.missing_threshold)

    Attributes:
        config: CombinerConfig instance with configuration options.
        weighted: If True, uses weighted averaging; if False, uses simple averaging.

    Example:
        >>> # Simple averaging
        >>> combiner = SimpleAveragingCombiner(weighted=False)
        >>> combiner.fit(train_preds, train_targets)
        >>> result = combiner.combine(test_preds)
        >>> print(result.weights)  # {'lgbm': 0.333, 'rf': 0.333, 'xgb': 0.333}

        >>> # Weighted averaging with custom weights
        >>> combiner = SimpleAveragingCombiner(weighted=True)
        >>> weights = {"lgbm": 0.5, "rf": 0.3, "xgb": 0.2}
        >>> result = combiner.combine(test_preds, weights=weights)
        >>> print(result.weights)  # {'lgbm': 0.5, 'rf': 0.3, 'xgb': 0.2}
    """

    def __init__(
        self,
        config: CombinerConfig | dict[str, Any] | None = None,
        weighted: bool = False,
    ) -> None:
        """Initialize the averaging combiner.

        Args:
            config: Configuration as CombinerConfig instance or dictionary.
                If None, default configuration is used. For simple averaging,
                consider setting require_fit=False since no training is needed.
            weighted: If True, use weighted averaging; if False, use simple
                averaging with equal weights. Default is False (simple averaging).

        Example:
            >>> # Simple averaging, no fit required
            >>> config = CombinerConfig(require_fit=False)
            >>> combiner = SimpleAveragingCombiner(config=config, weighted=False)

            >>> # Weighted averaging with fit required
            >>> config = CombinerConfig(require_fit=True)
            >>> combiner = SimpleAveragingCombiner(config=config, weighted=True)
        """
        super().__init__(config=config)
        self.weighted = weighted

        # Store weighted flag in config for persistence
        self.config.custom_config["weighted"] = weighted

        logger.debug(
            "Initialized SimpleAveragingCombiner (weighted=%s) with config: %s",
            self.weighted,
            self.config.to_dict(),
        )

    @property
    def name(self) -> str:
        """Return combiner name.

        Returns:
            Name string identifying this combiner type.
        """
        return "simple_averaging" if not self.weighted else "weighted_averaging"

    @property
    def version(self) -> str:
        """Return combiner version.

        Returns:
            Version string following semantic versioning.
        """
        return "1.0.0"

    def fit(
        self,
        train_predictions: dict[str, pd.DataFrame],
        _train_targets: pd.DataFrame,
        _validation_predictions: dict[str, pd.DataFrame] | None = None,
        _validation_targets: pd.DataFrame | None = None,
    ) -> None:
        """Fit the averaging combiner.

        For simple averaging (weighted=False):
            - Sets equal weights for all models (1/n)
            - No actual optimization is performed

        For weighted averaging (weighted=True):
            - Can optionally learn optimal weights based on historical performance
            - Currently sets equal weights as baseline (future: implement optimization)

        Args:
            train_predictions: Dictionary mapping model names to predictions
                on the training set. Each DataFrame should have columns like
                'pred_h0', 'pred_h1', etc.
            train_targets: True target values for training period.
                Should have columns like 'h0', 'h1', etc.
            validation_predictions: Optional validation predictions for
                hyperparameter tuning (currently unused).
            validation_targets: Optional validation targets (currently unused).

        Raises:
            ValueError: If prediction format is invalid.

        Example:
            >>> combiner = SimpleAveragingCombiner(weighted=False)
            >>> combiner.fit(train_preds, train_targets)
            >>> print(combiner.get_weights())
            {'lgbm': 0.333, 'rf': 0.333, 'xgb': 0.333}
        """
        self._validate_predictions(train_predictions)

        self._model_names = list(train_predictions.keys())

        # For simple averaging, always use equal weights
        if not self.weighted:
            self._global_weights = self._initialize_weights(self._model_names, equal=True)
            logger.info(
                "Fitted simple averaging combiner with equal weights for %d models",
                len(self._model_names),
            )
        else:
            # For weighted averaging, initialize with equal weights as baseline
            # Future enhancement: implement weight optimization based on train_targets
            self._global_weights = self._initialize_weights(self._model_names, equal=True)
            logger.info(
                "Fitted weighted averaging combiner with baseline equal weights for %d models "
                "(optimization not yet implemented)",
                len(self._model_names),
            )

        self._fitted = True
        self._fit_timestamp = datetime.now()  # noqa: DTZ005

        # Validate the weights
        if self.config.validate_weights:
            self._validate_weights(self._global_weights)

    def combine(  # noqa: PLR0915
        self,
        predictions: dict[str, pd.DataFrame],
        _metadata: dict[str, Any] | None = None,
        weights: dict[str, float] | None = None,
    ) -> CombinationResult:
        """Combine predictions from multiple models using averaging.

        For simple averaging:
            - Uses equal weights (1/n) for all models
            - Ignores custom weights if provided

        For weighted averaging:
            - Uses custom weights if provided, otherwise uses fitted weights
            - Validates that weights sum to 1.0

        Missing values (NaN) are handled by:
            - Excluding NaN values from the average computation
            - Adjusting weights proportionally for valid values only

        Args:
            predictions: Dictionary mapping model names to prediction DataFrames.
                Each DataFrame should have columns like 'pred_h0', 'pred_h1', etc.
                and a DatetimeIndex.
            metadata: Optional metadata about predictions (currently unused).
            weights: Optional custom weights for weighted averaging.
                Must sum to 1.0. Keys must match prediction model names.
                Only used if weighted=True.

        Returns:
            CombinationResult with:
                - combined_predictions: DataFrame with averaged predictions
                - weights: Dictionary of weights used for each model
                - metadata: Information about the combination process
                - model_contributions: Individual contributions (if config.store_contributions=True)
                - confidence_intervals: Combined intervals (if config.store_intervals=True)

        Raises:
            ValueError: If predictions format is invalid or weights are invalid.
            RuntimeError: If combiner not fitted (when config.require_fit=True).

        Example:
            >>> # Simple averaging
            >>> result = combiner.combine(predictions)
            >>> print(result.weights)  # Equal weights

            >>> # Weighted averaging with custom weights
            >>> weights = {"lgbm": 0.5, "rf": 0.3, "xgb": 0.2}
            >>> result = combiner.combine(predictions, weights=weights)
            >>> print(result.weights)  # Custom weights
        """
        start_time = time.time()

        # Check if fitting is required
        self._check_is_fitted()

        # Validate predictions format
        self._validate_predictions(predictions)

        # Handle models with excessive missing data
        cleaned_predictions, excluded_models = self._handle_missing_predictions(predictions)

        # Determine weights to use
        if not self.weighted:
            # Simple averaging: always use equal weights
            n_models = len(cleaned_predictions)
            final_weights = dict.fromkeys(cleaned_predictions.keys(), 1.0 / n_models)
            logger.debug("Using simple averaging with equal weights for %d models", n_models)
        # Weighted averaging: use custom weights or fitted weights
        elif weights is not None:
            # Use custom weights provided at combination time
            # Filter to only include models present in cleaned_predictions
            final_weights = {name: weights[name] for name in cleaned_predictions if name in weights}

            # Check if any models are missing from weights
            missing_models = set(cleaned_predictions) - set(final_weights)
            if missing_models:
                msg = (
                    f"Custom weights missing for models: {missing_models}. "
                    f"Weights must be provided for all models."
                )
                raise ValueError(msg)

            # Normalize weights to ensure they sum to 1
            final_weights = self._normalize_weights(final_weights)

            # Validate weights if configured
            if self.config.validate_weights:
                self._validate_weights(final_weights)

            logger.debug("Using custom weights for %d models", len(final_weights))
        else:
            # Use fitted weights
            final_weights = {
                name: self._global_weights.get(name, 0.0) for name in cleaned_predictions
            }

            # Normalize in case some models were excluded
            final_weights = self._normalize_weights(final_weights)

            logger.debug(
                "Using fitted weights for %d models",
                len(final_weights),
            )

        # Get prediction columns
        first_pred = next(iter(cleaned_predictions.values()))
        pred_cols = sorted([col for col in first_pred.columns if col.startswith("pred_h")])

        # Perform weighted averaging with NaN handling
        combined_df = pd.DataFrame(index=first_pred.index)
        model_contributions = {} if self.config.store_contributions else None

        for col in pred_cols:
            # Stack predictions from all models
            # Shape: (n_samples, n_models)
            pred_stack = np.stack(
                [cleaned_predictions[name][col].to_numpy() for name in final_weights],
                axis=1,
            )

            # Create mask for valid (non-NaN) predictions
            valid_mask = ~np.isnan(pred_stack)

            # Compute weighted average handling NaN
            # For each sample, renormalize weights to only include valid models
            weighted_sum = np.zeros(len(first_pred))
            weight_sum = np.zeros(len(first_pred))

            for i, (_name, weight) in enumerate(final_weights.items()):
                pred_values = pred_stack[:, i]
                valid = valid_mask[:, i]

                # Add weighted contribution where valid
                weighted_sum += np.where(valid, pred_values * weight, 0.0)
                weight_sum += np.where(valid, weight, 0.0)

            # Compute average (avoid division by zero)
            with np.errstate(divide="ignore", invalid="ignore"):
                combined_values = np.where(
                    weight_sum > 0,
                    weighted_sum / weight_sum,
                    np.nan,
                )

            combined_df[col] = combined_values

            # Store individual contributions if requested
            if self.config.store_contributions:
                for name, weight in final_weights.items():
                    if name not in model_contributions:
                        model_contributions[name] = pd.DataFrame(index=first_pred.index)
                    # Store weighted contribution
                    model_contributions[name][col] = (
                        cleaned_predictions[name][col].to_numpy() * weight
                    )

        # Combine prediction intervals if requested
        confidence_intervals = None
        if self.config.store_intervals:
            confidence_intervals = self._combine_intervals(cleaned_predictions, final_weights)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create metadata
        metadata_obj = self._create_metadata(
            models_used=list(cleaned_predictions.keys()),
            models_excluded=excluded_models,
            processing_time=processing_time,
            combination_method=self.name,
            weighted=self.weighted,
            weights_used=final_weights.copy(),
        )

        # Add custom metadata fields
        metadata_obj.configuration["weighted"] = self.weighted
        metadata_obj.performance_metrics["weights_entropy"] = float(
            -sum(w * np.log(w + 1e-10) for w in final_weights.values())
        )

        # Create result
        result = CombinationResult(
            combined_predictions=combined_df,
            weights=final_weights,
            metadata=metadata_obj,
            model_contributions=model_contributions,
            confidence_intervals=confidence_intervals,
        )

        logger.info(
            "Combined %d models using %s (processing time: %.3fs)",
            len(final_weights),
            self.name,
            processing_time,
        )

        return result

    @classmethod
    def load(cls, filepath: str | Path) -> "SimpleAveragingCombiner":
        """Load combiner from file.

        Deserializes a previously saved combiner state, including the weighted flag.

        Args:
            filepath: Path to load file.

        Returns:
            Loaded SimpleAveragingCombiner instance.

        Example:
            >>> combiner = SimpleAveragingCombiner.load("combiners/simple_avg.pkl")
        """
        filepath = Path(filepath)
        state = joblib.load(filepath)

        # Extract weighted flag from config
        weighted = state["config"].get("custom_config", {}).get("weighted", False)

        # Create instance with saved config and weighted flag
        instance = cls(config=state["config"], weighted=weighted)

        # Restore state
        instance._weights = state["weights"]
        instance._global_weights = state["global_weights"]
        instance._fitted = state["fitted"]
        instance._fit_timestamp = (
            datetime.fromisoformat(state["fit_timestamp"]) if state["fit_timestamp"] else None
        )
        instance._performance_history = state["performance_history"]
        instance._model_names = state["model_names"]

        logger.info("Loaded %s combiner from %s", instance.name, filepath)

        return instance
