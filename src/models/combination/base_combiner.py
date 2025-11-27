"""Base interface for model combination strategies.

This module provides the BaseCombiner abstract class that defines the standard
contract for all model combination strategies in the PrevCarga forecasting system.

The framework supports:
- Multiple combination methods (averaging, weighted voting, selection, etc.)
- 26 time series (17 areas + 4 subsystems + 4 losses + 1 national)
- 9 forecast horizons (D+0 to D+8)
- Both point forecasts and prediction intervals
- Missing prediction handling
- Weight management and persistence

Example:
    ```python
    from src.models.combination import BaseCombiner, CombinationResult

    class WeightedAverageCombiner(BaseCombiner):
        @property
        def name(self) -> str:
            return "weighted_average"

        @property
        def version(self) -> str:
            return "1.0.0"

        def combine(self, predictions, metadata=None):
            # Implement weighted combination
            ...

        def fit(self, train_predictions, train_targets, **kwargs):
            # Learn optimal weights from training data
            ...

    combiner = WeightedAverageCombiner()
    combiner.fit(train_preds, train_targets)
    result = combiner.combine(test_preds)
    ```
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.combination.data_structures import (
    CombinationMetadata,
    CombinationResult,
    CombinerConfig,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Constants for PrevCarga forecasting system
N_TIME_SERIES = 26  # 17 areas + 4 subsystems + 4 losses + 1 national
N_HORIZONS = 9  # D+0 to D+8


class BaseCombiner(ABC):
    """Abstract base class for model combination strategies.

    Defines standard interface for combining predictions from multiple
    forecasting models. Subclasses implement specific combination logic
    such as simple averaging, weighted voting, model selection, stacking, etc.

    Key Features:
        - Standard combine() and fit() interface
        - Input validation and missing value handling
        - Metadata tracking and structured logging
        - Weight management with per-time-series and per-horizon support
        - Prediction interval combination
        - Model persistence (save/load)

    Attributes:
        config: CombinerConfig instance with configuration options.

    Example:
        >>> class SimpleCombiner(BaseCombiner):
        ...     @property
        ...     def name(self) -> str:
        ...         return "simple_average"
        ...
        ...     @property
        ...     def version(self) -> str:
        ...         return "1.0.0"
        ...
        ...     def combine(self, predictions, metadata=None):
        ...         # Implementation
        ...         pass
        ...
        ...     def fit(self, train_predictions, train_targets, **kwargs):
        ...         # Implementation
        ...         pass
        >>>
        >>> combiner = SimpleCombiner()
        >>> combiner.fit(train_preds, train_targets)
        >>> result = combiner.combine(test_preds)
    """

    def __init__(self, config: CombinerConfig | dict[str, Any] | None = None) -> None:
        """Initialize combiner.

        Args:
            config: Configuration as CombinerConfig instance or dictionary.
                If None, default configuration is used.
        """
        if config is None:
            self.config = CombinerConfig()
        elif isinstance(config, dict):
            self.config = CombinerConfig.from_dict(config)
        else:
            self.config = config

        # Internal state
        self._weights: dict[str, dict[str, float]] = {}  # {time_series: {model: weight}}
        self._global_weights: dict[str, float] = {}  # {model: weight} for all time series
        self._fitted = False
        self._fit_timestamp: datetime | None = None
        self._performance_history: list[dict[str, Any]] = []
        self._model_names: list[str] = []

        logger.debug(
            "Initialized %s combiner with config: %s",
            self.__class__.__name__,
            self.config.to_dict(),
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Return combiner name.

        Returns:
            Unique name identifying this combiner type.
        """
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Return combiner version.

        Returns:
            Version string (e.g., "1.0.0").
        """
        ...

    @property
    def is_fitted(self) -> bool:
        """Check if combiner has been fitted.

        Returns:
            True if fit() has been called successfully.
        """
        return self._fitted

    @property
    def fit_timestamp(self) -> datetime | None:
        """Get timestamp of last fit operation.

        Returns:
            Datetime of last fit, or None if not fitted.
        """
        return self._fit_timestamp

    @property
    def model_names(self) -> list[str]:
        """Get list of model names the combiner was fitted on.

        Returns:
            List of model names.
        """
        return self._model_names.copy()

    @abstractmethod
    def combine(
        self,
        predictions: dict[str, pd.DataFrame],
        metadata: dict[str, Any] | None = None,
    ) -> CombinationResult:
        """Combine predictions from multiple models.

        Args:
            predictions: Dictionary mapping model names to prediction DataFrames.
                Each DataFrame should have columns like 'pred_h0', 'pred_h1', etc.
                for different forecast horizons, and an index representing time
                (typically DatetimeIndex for semi-hourly data).
            metadata: Optional metadata about predictions (e.g., data quality info).

        Returns:
            CombinationResult with combined predictions, weights, and metadata.

        Raises:
            ValueError: If predictions format is invalid.
            RuntimeError: If combiner not fitted (when required by config).

        Example:
            >>> predictions = {
            ...     "lgbm": lgbm_predictions_df,
            ...     "rf": rf_predictions_df,
            ...     "xgb": xgb_predictions_df
            ... }
            >>> result = combiner.combine(predictions)
            >>> combined_df = result.combined_predictions
        """
        ...

    @abstractmethod
    def fit(
        self,
        train_predictions: dict[str, pd.DataFrame],
        train_targets: pd.DataFrame,
        validation_predictions: dict[str, pd.DataFrame] | None = None,
        validation_targets: pd.DataFrame | None = None,
    ) -> None:
        """Fit/train the combination strategy.

        Learn optimal combination weights or parameters from training data.
        Different combiners implement different learning strategies.

        Args:
            train_predictions: Dictionary mapping model names to predictions
                on the training set.
            train_targets: True target values for training period.
            validation_predictions: Optional validation predictions for
                hyperparameter tuning.
            validation_targets: Optional validation targets.

        Raises:
            ValueError: If input format is invalid.

        Example:
            >>> combiner.fit(
            ...     train_predictions={"lgbm": train_lgbm, "rf": train_rf},
            ...     train_targets=train_y,
            ...     validation_predictions={"lgbm": val_lgbm, "rf": val_rf},
            ...     validation_targets=val_y
            ... )
        """
        ...

    def get_weights(
        self,
        time_series: str | None = None,
        horizon: int | None = None,
    ) -> dict[str, float]:
        """Get combination weights.

        Retrieves the weights used for combining model predictions.
        Can optionally filter by time series or horizon.

        Args:
            time_series: Optional specific time series name.
                If None, returns global/average weights.
            horizon: Optional specific horizon (0-8).
                Currently not implemented (returns same weights for all horizons).

        Returns:
            Dictionary mapping model names to their weights.

        Raises:
            RuntimeError: If combiner not fitted.

        Example:
            >>> weights = combiner.get_weights()
            >>> print(weights)
            {'lgbm': 0.45, 'rf': 0.30, 'xgb': 0.25}

            >>> weights_seco = combiner.get_weights(time_series='SECO')
            >>> print(weights_seco)
            {'lgbm': 0.50, 'rf': 0.25, 'xgb': 0.25}
        """
        if not self.is_fitted:
            msg = "Combiner not fitted. Call fit() first."
            raise RuntimeError(msg)

        # If specific time series requested
        if time_series is not None:
            if time_series in self._weights:
                return self._weights[time_series].copy()
            # Fall back to global weights
            return self._global_weights.copy()

        # Return global weights
        if self._global_weights:
            return self._global_weights.copy()

        # Calculate average weights across all time series
        if not self._weights:
            return {}

        all_models = set()
        for ts_weights in self._weights.values():
            all_models.update(ts_weights.keys())

        avg_weights = {}
        for model in all_models:
            weights = [ts_weights.get(model, 0.0) for ts_weights in self._weights.values()]
            avg_weights[model] = float(np.mean(weights))

        # Normalize to sum to 1
        total = sum(avg_weights.values())
        if total > 0:
            avg_weights = {k: v / total for k, v in avg_weights.items()}

        return avg_weights

    def _check_is_fitted(self) -> None:
        """Check if combiner is fitted and raise if not.

        Raises:
            RuntimeError: If combiner not fitted and config.require_fit is True.
        """
        if self.config.require_fit and not self._fitted:
            msg = f"{self.name} combiner not fitted. Call fit() first."
            raise RuntimeError(msg)

    def _validate_predictions(self, predictions: dict[str, pd.DataFrame]) -> None:
        """Validate prediction format and consistency.

        Checks that all model predictions have consistent format, columns,
        and index alignment.

        Args:
            predictions: Dictionary of model predictions.

        Raises:
            ValueError: If predictions are invalid.
        """
        if not predictions:
            msg = "Predictions dictionary is empty"
            raise ValueError(msg)

        # Get first model's predictions as reference
        first_model = next(iter(predictions.keys()))
        first_pred = predictions[first_model]

        if first_pred.empty:
            msg = f"Predictions for {first_model!r} are empty"
            raise ValueError(msg)

        # Get expected prediction columns (pred_h0, pred_h1, etc.)
        expected_cols = sorted([col for col in first_pred.columns if col.startswith("pred_h")])

        if not expected_cols:
            msg = "No prediction columns found (expected pred_h0, pred_h1, etc.)"
            raise ValueError(msg)

        # Validate consistency across models
        for model_name, pred_df in predictions.items():
            if pred_df.empty:
                msg = f"Predictions for {model_name!r} are empty"
                raise ValueError(msg)

            # Check columns match
            model_cols = sorted([col for col in pred_df.columns if col.startswith("pred_h")])
            if model_cols != expected_cols:
                msg = (
                    f"Model {model_name!r} has inconsistent columns: "
                    f"expected {expected_cols}, got {model_cols}"
                )
                raise ValueError(msg)

            # Check index alignment
            if not pred_df.index.equals(first_pred.index):
                msg = f"Model {model_name!r} has misaligned index with reference model {first_model!r}"
                raise ValueError(msg)

        logger.debug(
            "Validated predictions from %d models with %d horizons",
            len(predictions),
            len(expected_cols),
        )

    def _handle_missing_predictions(
        self,
        predictions: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """Handle missing predictions from some models.

        Detects models with excessive missing values and excludes them
        from the combination.

        Args:
            predictions: Dictionary of model predictions.

        Returns:
            Tuple of (cleaned_predictions, excluded_models).

        Raises:
            ValueError: If all models are excluded due to missing data.
        """
        cleaned = {}
        excluded = []

        for model_name, pred_df in predictions.items():
            # Check for excessive missing values
            pred_cols = [col for col in pred_df.columns if col.startswith("pred_h")]
            missing_pct = float(pred_df[pred_cols].isna().mean().mean())

            if missing_pct > self.config.missing_threshold:
                excluded.append(model_name)
                logger.warning(
                    "Excluding %s: %.1f%% missing predictions (threshold: %.1f%%)",
                    model_name,
                    missing_pct * 100,
                    self.config.missing_threshold * 100,
                )
            else:
                cleaned[model_name] = pred_df

        if not cleaned:
            msg = (
                f"All models excluded due to missing predictions "
                f"(threshold: {self.config.missing_threshold:.1%}). "
                f"Excluded: {excluded}"
            )
            raise ValueError(msg)

        return cleaned, excluded

    def _create_metadata(
        self,
        models_used: list[str],
        models_excluded: list[str],
        processing_time: float,
        **kwargs: Any,
    ) -> CombinationMetadata:
        """Create combination metadata.

        Args:
            models_used: List of models used in combination.
            models_excluded: List of models excluded.
            processing_time: Processing time in seconds.
            **kwargs: Additional metadata fields (performance_metrics, warnings, etc.).

        Returns:
            CombinationMetadata instance.
        """
        metadata = CombinationMetadata(
            timestamp=datetime.now(),
            combination_method=self.name,
            models_used=models_used.copy(),
            models_excluded=models_excluded.copy(),
            processing_time_seconds=processing_time,
            configuration=self.config.to_dict(),
        )

        # Add any additional fields
        if "performance_metrics" in kwargs:
            metadata.performance_metrics = kwargs["performance_metrics"]
        if "warnings" in kwargs:
            for warning in kwargs["warnings"]:
                metadata.add_warning(warning)

        return metadata

    def _initialize_weights(
        self,
        model_names: list[str],
        equal: bool = True,
    ) -> dict[str, float]:
        """Initialize weights for models.

        Args:
            model_names: List of model names.
            equal: If True, initialize with equal weights.

        Returns:
            Dictionary mapping model names to weights.
        """
        n_models = len(model_names)
        if n_models == 0:
            return {}

        if equal:
            weight = 1.0 / n_models
            return {name: weight for name in model_names}

        # Initialize with small random perturbations around equal weights
        base_weight = 1.0 / n_models
        weights = {}
        for name in model_names:
            weights[name] = base_weight + np.random.uniform(-0.01, 0.01)

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _validate_weights(self, weights: dict[str, float]) -> None:
        """Validate that weights are valid.

        Args:
            weights: Dictionary of model weights.

        Raises:
            ValueError: If weights are invalid.
        """
        if not weights:
            msg = "Weights dictionary is empty"
            raise ValueError(msg)

        # Check all weights are non-negative
        if any(w < 0 for w in weights.values()):
            neg_models = [k for k, v in weights.items() if v < 0]
            msg = f"Weights must be non-negative. Negative weights found for: {neg_models}"
            raise ValueError(msg)

        # Check weights sum to 1
        weight_sum = sum(weights.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            msg = f"Weights must sum to 1, got {weight_sum:.6f}"
            raise ValueError(msg)

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights to sum to 1.

        Args:
            weights: Dictionary of model weights.

        Returns:
            Normalized weights.
        """
        total = sum(weights.values())
        if total == 0:
            # Equal weights if all are zero
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        return {k: v / total for k, v in weights.items()}

    def _track_performance(
        self,
        result: CombinationResult,
        targets: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Track performance of the combination.

        Calculates and stores performance metrics for the combination.

        Args:
            result: Combination result.
            targets: Optional true targets for metric calculation.

        Returns:
            Dictionary of performance metrics.
        """
        metrics: dict[str, float] = {}

        # Basic metrics
        metrics["n_models"] = float(len(result.weights))
        metrics["weight_entropy"] = float(
            -sum(w * np.log(w + 1e-10) for w in result.weights.values())
        )
        metrics["max_weight"] = float(max(result.weights.values()))
        metrics["min_weight"] = float(min(result.weights.values()))

        # If targets provided, calculate accuracy metrics
        if targets is not None:
            pred_cols = [col for col in result.combined_predictions.columns if col.startswith("pred_h")]
            for col in pred_cols:
                if col.replace("pred_", "") in targets.columns:
                    target_col = col.replace("pred_", "")
                    pred = result.combined_predictions[col]
                    actual = targets[target_col]

                    # RMSE
                    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
                    metrics[f"rmse_{col}"] = rmse

                    # MAE
                    mae = float(np.mean(np.abs(pred - actual)))
                    metrics[f"mae_{col}"] = mae

        # Store in history
        self._performance_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
            }
        )

        return metrics

    def _combine_intervals(
        self,
        predictions: dict[str, pd.DataFrame],
        weights: dict[str, float],
    ) -> pd.DataFrame | None:
        """Combine prediction intervals from models.

        Calculates combined confidence intervals using weighted combination
        of individual model intervals.

        Args:
            predictions: Dictionary of model predictions.
            weights: Dictionary of model weights.

        Returns:
            DataFrame with combined intervals (lower and upper bounds),
            or None if interval columns not available.
        """
        # Check if interval columns exist
        first_pred = next(iter(predictions.values()))
        lower_cols = [col for col in first_pred.columns if col.endswith("_lower")]
        upper_cols = [col for col in first_pred.columns if col.endswith("_upper")]

        if not lower_cols or not upper_cols:
            return None

        combined_intervals = pd.DataFrame(index=first_pred.index)

        # Combine lower bounds (minimum across weighted models)
        for col in lower_cols:
            values = []
            for model_name, pred_df in predictions.items():
                if col in pred_df.columns and model_name in weights:
                    values.append(pred_df[col].values * weights[model_name])
            if values:
                combined_intervals[col] = np.sum(values, axis=0)

        # Combine upper bounds
        for col in upper_cols:
            values = []
            for model_name, pred_df in predictions.items():
                if col in pred_df.columns and model_name in weights:
                    values.append(pred_df[col].values * weights[model_name])
            if values:
                combined_intervals[col] = np.sum(values, axis=0)

        return combined_intervals if not combined_intervals.empty else None

    def save(self, filepath: str | Path) -> None:
        """Save combiner state to file.

        Serializes the combiner's weights, configuration, and metadata
        for later loading.

        Args:
            filepath: Path to save file (typically .pkl or .joblib).

        Example:
            >>> combiner.save("combiners/weighted_combiner_v1.pkl")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

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
        }

        joblib.dump(state, filepath)
        logger.info("Saved %s combiner to %s", self.name, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "BaseCombiner":
        """Load combiner from file.

        Deserializes a previously saved combiner state.

        Args:
            filepath: Path to load file.

        Returns:
            Loaded combiner instance.

        Example:
            >>> combiner = WeightedCombiner.load("combiners/weighted_combiner_v1.pkl")
        """
        filepath = Path(filepath)
        state = joblib.load(filepath)

        # Create instance with saved config
        instance = cls(config=state["config"])

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

    def get_performance_history(self) -> list[dict[str, Any]]:
        """Get historical performance metrics.

        Returns:
            List of performance metric dictionaries from previous combinations.
        """
        return self._performance_history.copy()

    def reset(self) -> None:
        """Reset combiner to unfitted state.

        Clears all learned weights and performance history.
        """
        self._weights = {}
        self._global_weights = {}
        self._fitted = False
        self._fit_timestamp = None
        self._performance_history = []
        self._model_names = []
        logger.info("Reset %s combiner to unfitted state", self.name)

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation of the combiner.
        """
        status = "fitted" if self._fitted else "not fitted"
        n_models = len(self._model_names) if self._model_names else 0
        return f"{self.__class__.__name__}(name={self.name!r}, version={self.version!r}, {status}, n_models={n_models})"
