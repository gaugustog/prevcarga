"""Best model selector for ensemble forecasting.

This module provides the BestModelSelector class that selects the single
best-performing model from an ensemble based on validation performance.
Unlike combiners that blend forecasts, this selects one model's forecast
entirely.

Key Features:
- Multiple selection criteria (min_mape, min_mae, min_rmse, max_r2)
- Per-horizon selection (different models for each forecast horizon)
- Fallback mechanism when selected model unavailable
- Full compatibility with BaseCombiner interface

Example:
    ```python
    from src.models.combination import BestModelSelector, CombinerConfig
    import numpy as np
    import pandas as pd

    # Historical data for selection
    train_predictions = {
        "lgbm": pd.DataFrame({"pred_h0": [100, 200, 300]}),
        "rf": pd.DataFrame({"pred_h0": [110, 190, 310]}),
        "arima": pd.DataFrame({"pred_h0": [105, 195, 305]})
    }
    train_targets = pd.DataFrame({"h0": [98, 202, 295]})

    # Configure selector
    config = CombinerConfig(require_fit=True)
    selector = BestModelSelector(
        selection_criterion="min_mape",
        config=config
    )

    # Select best model
    selector.fit(train_predictions, train_targets)
    print(f"Best model: {selector.get_best_model()}")  # 'lgbm'

    # Apply to test data
    test_predictions = {
        "lgbm": pd.DataFrame({"pred_h0": [100, 200, 300]}),
        "rf": pd.DataFrame({"pred_h0": [110, 190, 310]}),
        "arima": pd.DataFrame({"pred_h0": [105, 195, 305]})
    }
    result = selector.combine(test_predictions)
    # result.combined_predictions uses only lgbm forecast
    # result.metadata['selected_model'] = 'lgbm'
    ```
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

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

# Type aliases
SelectionCriterion = Literal["min_mape", "min_mae", "min_rmse", "max_r2", "min_combined"]
FallbackStrategy = Literal["second_best", "simple_average", "error"]


class BestModelSelector(BaseCombiner):
    """Selector that chooses the single best-performing model from an ensemble.

    This selector identifies the best model based on validation performance
    and returns only that model's forecast. Unlike combiners that blend
    forecasts, this is a pure selection strategy.

    Key Features:
        - Multiple selection criteria (MAPE, MAE, RMSE, R²)
        - Optional per-horizon selection (different model for each horizon)
        - Fallback mechanism when best model unavailable
        - Full metadata tracking

    Attributes:
        config: CombinerConfig instance with configuration options.
        selection_criterion: Metric to use for selection.
        per_horizon: Whether to select different models per horizon.
        fallback_strategy: What to do if best model unavailable.

    Example:
        >>> # Single best model across all horizons
        >>> selector = BestModelSelector(selection_criterion="min_mape")
        >>> selector.fit(train_predictions, train_targets)
        >>> result = selector.combine(test_predictions)

        >>> # Different best model per horizon
        >>> selector = BestModelSelector(
        ...     selection_criterion="min_mae",
        ...     per_horizon=True
        ... )
        >>> selector.fit(train_predictions, train_targets)
        >>> print(selector.get_best_models())  # Dict of horizon -> model
    """

    def __init__(
        self,
        config: CombinerConfig | dict[str, Any] | None = None,
        selection_criterion: SelectionCriterion = "min_mape",
        per_horizon: bool = False,
        fallback_strategy: FallbackStrategy = "error",
        combined_weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize the best model selector.

        Args:
            config: Configuration as CombinerConfig instance or dictionary.
                If None, default configuration with require_fit=True is used.
            selection_criterion: Metric to use for model selection:
                - "min_mape": Select model with lowest MAPE
                - "min_mae": Select model with lowest MAE
                - "min_rmse": Select model with lowest RMSE
                - "max_r2": Select model with highest R²
                - "min_combined": Weighted combination of metrics
            per_horizon: If True, select different best model for each horizon.
                If False, use same best model for all horizons.
            fallback_strategy: What to do if best model unavailable:
                - "error": Raise an error
                - "second_best": Use second-best model
                - "simple_average": Fall back to simple averaging
            combined_weights: Weights for "min_combined" criterion.
                Dict mapping metric names to weights (must sum to 1).
                Default: {"mape": 0.4, "mae": 0.3, "rmse": 0.3}

        Example:
            >>> # Standard MAPE-based selection
            >>> selector = BestModelSelector(selection_criterion="min_mape")

            >>> # Per-horizon selection with MAE
            >>> selector = BestModelSelector(
            ...     selection_criterion="min_mae",
            ...     per_horizon=True
            ... )

            >>> # Combined metric selection
            >>> selector = BestModelSelector(
            ...     selection_criterion="min_combined",
            ...     combined_weights={"mape": 0.5, "mae": 0.25, "rmse": 0.25}
            ... )
        """
        # Default to requiring fit for selectors
        if config is None:
            config = CombinerConfig(require_fit=True)
        elif isinstance(config, dict):
            config_dict = config.copy()
            if "require_fit" not in config_dict:
                config_dict["require_fit"] = True
            config = CombinerConfig.from_dict(config_dict)

        super().__init__(config=config)

        self.selection_criterion = selection_criterion
        self.per_horizon = per_horizon
        self.fallback_strategy = fallback_strategy

        # Weights for combined metric
        if combined_weights is None:
            self.combined_weights = {"mape": 0.4, "mae": 0.3, "rmse": 0.3}
        else:
            self.combined_weights = combined_weights
            # Validate weights sum to 1
            weight_sum = sum(combined_weights.values())
            if not np.isclose(weight_sum, 1.0, atol=1e-6):
                msg = f"combined_weights must sum to 1, got {weight_sum:.6f}"
                raise ValueError(msg)

        # Store selection results
        self._best_model: str | None = None  # Single best model (when per_horizon=False)
        self._best_models_per_horizon: dict[str, str] = (
            {}
        )  # horizon -> model (when per_horizon=True)
        self._model_scores: dict[str, dict[str, float]] = {}  # model -> {metric: score}
        self._ranked_models: list[str] = []  # Models ranked by performance

        # Store config options for persistence
        self.config.custom_config["selection_criterion"] = selection_criterion
        self.config.custom_config["per_horizon"] = per_horizon
        self.config.custom_config["fallback_strategy"] = fallback_strategy
        self.config.custom_config["combined_weights"] = self.combined_weights.copy()

        logger.debug(
            "Initialized BestModelSelector (criterion=%s, per_horizon=%s) with config: %s",
            self.selection_criterion,
            self.per_horizon,
            self.config.to_dict(),
        )

    @property
    def name(self) -> str:
        """Return selector name.

        Returns:
            Name string identifying this selector type.
        """
        return "best_model_selector"

    @property
    def version(self) -> str:
        """Return selector version.

        Returns:
            Version string following semantic versioning.
        """
        return "1.0.0"

    def fit(
        self,
        train_predictions: dict[str, pd.DataFrame],
        train_targets: pd.DataFrame,
        validation_predictions: dict[str, pd.DataFrame] | None = None,
        validation_targets: pd.DataFrame | None = None,
    ) -> None:
        """Select best model(s) based on validation performance.

        Computes performance metrics for each model and selects the best
        according to the configured criterion.

        Args:
            train_predictions: Dictionary mapping model names to predictions
                on the training set. Each DataFrame should have columns like
                'pred_h0', 'pred_h1', etc.
            train_targets: True target values for training period.
                Should have columns like 'h0', 'h1', etc.
            validation_predictions: Optional validation predictions.
                If provided, used instead of train_predictions for selection.
            validation_targets: Optional validation targets.
                If provided, used instead of train_targets for selection.

        Raises:
            ValueError: If prediction format is invalid.

        Example:
            >>> selector.fit(train_predictions, train_targets)
            >>> print(selector.get_best_model())  # 'lgbm'

            >>> # Use validation set for selection
            >>> selector.fit(
            ...     train_predictions, train_targets,
            ...     validation_predictions, validation_targets
            ... )
        """
        self._validate_predictions(train_predictions)
        self._model_names = list(train_predictions.keys())

        # Use validation set if provided, otherwise use training set
        predictions = (
            validation_predictions if validation_predictions is not None else train_predictions
        )
        targets = validation_targets if validation_targets is not None else train_targets

        if self.per_horizon:
            # Select best model for each horizon separately
            self._select_per_horizon(predictions, targets)
            logger.info(
                "Selected best models per horizon using %s: %s",
                self.selection_criterion,
                self._best_models_per_horizon,
            )
        else:
            # Select single best model across all horizons
            self._select_global_best(predictions, targets)
            logger.info(
                "Selected best model using %s: %s (score: %.4f)",
                self.selection_criterion,
                self._best_model,
                self._model_scores.get(self._best_model or "", {}).get("combined", 0.0),
            )

        self._fitted = True
        self._fit_timestamp = datetime.now()  # noqa: DTZ005

    def combine(
        self,
        predictions: dict[str, pd.DataFrame],
        _metadata: dict[str, Any] | None = None,
        override_model: str | None = None,
    ) -> CombinationResult:
        """Return forecast from best model.

        Args:
            predictions: Dictionary mapping model names to prediction DataFrames.
                Each DataFrame should have columns like 'pred_h0', 'pred_h1', etc.
            metadata: Optional metadata about predictions (currently unused).
            override_model: Optional model name to use instead of selected best.
                Useful for testing different model selections.

        Returns:
            CombinationResult with:
                - combined_predictions: DataFrame with selected model's forecast
                - weights: Single weight of 1.0 for selected model
                - metadata: Information including which model was selected

        Raises:
            ValueError: If predictions format is invalid.
            RuntimeError: If selector not fitted (when config.require_fit=True).
            KeyError: If best model not in predictions (when fallback_strategy="error").

        Example:
            >>> result = selector.combine(predictions)
            >>> print(result.metadata.configuration["selected_model"])  # 'lgbm'
            >>> print(result.weights)  # {'lgbm': 1.0}

            >>> # Override selection for testing
            >>> result = selector.combine(predictions, override_model="rf")
        """
        start_time = time.time()

        # Check if fitting is required
        self._check_is_fitted()

        # Validate predictions format
        self._validate_predictions(predictions)

        # Handle models with excessive missing data
        cleaned_predictions, excluded_models = self._handle_missing_predictions(predictions)

        # Determine which model(s) to use
        if self.per_horizon:
            # Use per-horizon selection
            combined_df, selected_models = self._combine_per_horizon(
                cleaned_predictions, override_model
            )
            # Create weights dict with all selected models
            unique_models = set(selected_models.values())
            final_weights = {model: 1.0 / len(unique_models) for model in unique_models}
        else:
            # Use single best model
            if override_model is not None:
                selected_model = override_model
            else:
                selected_model = self._best_model or ""

            # Handle case where best model not available
            if selected_model not in cleaned_predictions:
                selected_model = self._handle_missing_model(
                    selected_model, cleaned_predictions, excluded_models
                )

            # Extract selected model's predictions
            combined_df = cleaned_predictions[selected_model].copy()
            final_weights = {selected_model: 1.0}
            selected_models = dict.fromkeys(combined_df.columns, selected_model)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create metadata
        metadata_obj = self._create_metadata(
            models_used=list(final_weights),
            models_excluded=excluded_models,
            processing_time=processing_time,
        )

        # Add selection-specific metadata
        if self.per_horizon:
            metadata_obj.configuration["selected_models"] = selected_models
        else:
            metadata_obj.configuration["selected_model"] = next(iter(final_weights))
        metadata_obj.configuration["selection_criterion"] = self.selection_criterion
        metadata_obj.configuration["per_horizon"] = self.per_horizon
        metadata_obj.configuration["model_scores"] = self._model_scores.copy()

        # Create result
        result = CombinationResult(
            combined_predictions=combined_df,
            weights=final_weights,
            metadata=metadata_obj,
            model_contributions=None,  # Not applicable for selection
            confidence_intervals=None,
        )

        logger.info(
            "Selected model forecast (processing time: %.3fs)",
            processing_time,
        )

        return result

    def get_best_model(self) -> str:
        """Get the selected best model.

        Returns:
            Name of the best model (when per_horizon=False).

        Raises:
            RuntimeError: If selector not fitted or per_horizon=True.

        Example:
            >>> selector.fit(train_predictions, train_targets)
            >>> print(selector.get_best_model())  # 'lgbm'
        """
        if not self.is_fitted:
            msg = "Selector not fitted. Call fit() first."
            raise RuntimeError(msg)

        if self.per_horizon:
            msg = "get_best_model() not available for per_horizon=True. Use get_best_models()."
            raise RuntimeError(msg)

        if self._best_model is None:
            msg = "No best model selected"
            raise RuntimeError(msg)

        return self._best_model

    def get_best_models(self) -> dict[str, str]:
        """Get the selected best models per horizon.

        Returns:
            Dictionary mapping horizon names (e.g., 'pred_h0') to model names.

        Raises:
            RuntimeError: If selector not fitted or per_horizon=False.

        Example:
            >>> selector.fit(train_predictions, train_targets)
            >>> print(selector.get_best_models())
            {'pred_h0': 'lgbm', 'pred_h1': 'rf', 'pred_h2': 'lgbm'}
        """
        if not self.is_fitted:
            msg = "Selector not fitted. Call fit() first."
            raise RuntimeError(msg)

        if not self.per_horizon:
            msg = "get_best_models() only available for per_horizon=True. Use get_best_model()."
            raise RuntimeError(msg)

        return self._best_models_per_horizon.copy()

    def get_model_scores(self) -> dict[str, dict[str, float]]:
        """Get performance scores for all models.

        Returns:
            Dictionary mapping model names to their metric scores.
            Each model's dict contains keys like 'mape', 'mae', 'rmse', 'r2'.

        Raises:
            RuntimeError: If selector not fitted.

        Example:
            >>> selector.fit(train_predictions, train_targets)
            >>> scores = selector.get_model_scores()
            >>> print(scores['lgbm']['mape'])  # 2.5
        """
        if not self.is_fitted:
            msg = "Selector not fitted. Call fit() first."
            raise RuntimeError(msg)

        return {model: metrics.copy() for model, metrics in self._model_scores.items()}

    def _select_global_best(  # noqa: PLR0915, PLR0912
        self,
        predictions: dict[str, pd.DataFrame],
        targets: pd.DataFrame,
    ) -> None:
        """Select single best model across all horizons.

        Args:
            predictions: Model predictions.
            targets: True target values.
        """
        # Compute metrics for each model across all horizons
        self._model_scores = {}

        for model_name, pred_df in predictions.items():
            # Get prediction columns
            pred_cols = sorted([col for col in pred_df.columns if col.startswith("pred_h")])

            # Collect all errors across horizons
            all_errors = []
            all_actuals = []
            all_preds = []

            for pred_col in pred_cols:
                target_col = pred_col.replace("pred_", "")
                if target_col not in targets.columns:
                    continue

                actual = targets[target_col].to_numpy()
                forecast = pred_df[pred_col].to_numpy()

                # Remove NaN values
                valid_mask = ~(np.isnan(actual) | np.isnan(forecast))
                actual_valid = actual[valid_mask]
                forecast_valid = forecast[valid_mask]

                if len(actual_valid) == 0:
                    continue

                all_actuals.extend(actual_valid)
                all_preds.extend(forecast_valid)
                all_errors.extend(np.abs(actual_valid - forecast_valid))

            # Compute aggregate metrics
            actual_arr = np.array(all_actuals)
            pred_arr = np.array(all_preds)
            errors = np.array(all_errors)

            metrics = {}
            metrics["mae"] = float(np.mean(errors))
            metrics["rmse"] = float(np.sqrt(np.mean((actual_arr - pred_arr) ** 2)))

            # MAPE with epsilon for numerical stability
            epsilon = 1e-8
            mape_values = np.abs((actual_arr - pred_arr) / (np.abs(actual_arr) + epsilon)) * 100
            metrics["mape"] = float(np.mean(mape_values))

            # R²
            ss_res = np.sum((actual_arr - pred_arr) ** 2)
            ss_tot = np.sum((actual_arr - np.mean(actual_arr)) ** 2)
            metrics["r2"] = float(1.0 - ss_res / (ss_tot + epsilon))

            # Combined score (for ranking)
            if self.selection_criterion == "min_combined":
                combined_score = 0.0
                if "mape" in self.combined_weights:
                    combined_score += metrics["mape"] * self.combined_weights["mape"]
                if "mae" in self.combined_weights:
                    combined_score += metrics["mae"] * self.combined_weights["mae"]
                if "rmse" in self.combined_weights:
                    combined_score += metrics["rmse"] * self.combined_weights["rmse"]
                metrics["combined"] = combined_score

            self._model_scores[model_name] = metrics

        # Select best model based on criterion
        if self.selection_criterion == "min_mape":
            self._best_model = min(
                self._model_scores.items(),
                key=lambda x: x[1]["mape"],
            )[0]
            self._ranked_models = sorted(
                self._model_scores.keys(),
                key=lambda x: self._model_scores[x]["mape"],
            )
        elif self.selection_criterion == "min_mae":
            self._best_model = min(
                self._model_scores.items(),
                key=lambda x: x[1]["mae"],
            )[0]
            self._ranked_models = sorted(
                self._model_scores.keys(),
                key=lambda x: self._model_scores[x]["mae"],
            )
        elif self.selection_criterion == "min_rmse":
            self._best_model = min(
                self._model_scores.items(),
                key=lambda x: x[1]["rmse"],
            )[0]
            self._ranked_models = sorted(
                self._model_scores.keys(),
                key=lambda x: self._model_scores[x]["rmse"],
            )
        elif self.selection_criterion == "max_r2":
            self._best_model = max(
                self._model_scores.items(),
                key=lambda x: x[1]["r2"],
            )[0]
            self._ranked_models = sorted(
                self._model_scores.keys(),
                key=lambda x: self._model_scores[x]["r2"],
                reverse=True,
            )
        elif self.selection_criterion == "min_combined":
            self._best_model = min(
                self._model_scores.items(),
                key=lambda x: x[1]["combined"],
            )[0]
            self._ranked_models = sorted(
                self._model_scores.keys(),
                key=lambda x: self._model_scores[x]["combined"],
            )
        else:
            msg = f"Unknown selection criterion: {self.selection_criterion}"
            raise ValueError(msg)

        # Set equal weights for all models (not used, but required by base class)
        self._global_weights = {name: 1.0 / len(predictions) for name in predictions}

    def _select_per_horizon(  # noqa: PLR0915, PLR0912
        self,
        predictions: dict[str, pd.DataFrame],
        targets: pd.DataFrame,
    ) -> None:
        """Select best model for each horizon separately.

        Args:
            predictions: Model predictions.
            targets: True target values.
        """
        # Get all prediction columns
        first_pred = next(iter(predictions.values()))
        pred_cols = sorted([col for col in first_pred.columns if col.startswith("pred_h")])

        self._best_models_per_horizon = {}
        self._model_scores = {}

        for pred_col in pred_cols:
            target_col = pred_col.replace("pred_", "")
            if target_col not in targets.columns:
                continue

            # Compute metrics for each model for this horizon
            horizon_scores = {}

            for model_name, pred_df in predictions.items():
                actual = targets[target_col].to_numpy()
                forecast = pred_df[pred_col].to_numpy()

                # Remove NaN values
                valid_mask = ~(np.isnan(actual) | np.isnan(forecast))
                actual_valid = actual[valid_mask]
                forecast_valid = forecast[valid_mask]

                if len(actual_valid) == 0:
                    continue

                errors = np.abs(actual_valid - forecast_valid)
                metrics = {}
                metrics["mae"] = float(np.mean(errors))
                metrics["rmse"] = float(np.sqrt(np.mean((actual_valid - forecast_valid) ** 2)))

                # MAPE
                epsilon = 1e-8
                mape_values = (
                    np.abs((actual_valid - forecast_valid) / (np.abs(actual_valid) + epsilon)) * 100
                )
                metrics["mape"] = float(np.mean(mape_values))

                # R²
                ss_res = np.sum((actual_valid - forecast_valid) ** 2)
                ss_tot = np.sum((actual_valid - np.mean(actual_valid)) ** 2)
                metrics["r2"] = float(1.0 - ss_res / (ss_tot + epsilon))

                horizon_scores[model_name] = metrics

                # Accumulate scores for get_model_scores()
                if model_name not in self._model_scores:
                    self._model_scores[model_name] = {
                        "mape": [],
                        "mae": [],
                        "rmse": [],
                        "r2": [],
                    }
                self._model_scores[model_name]["mape"].append(metrics["mape"])
                self._model_scores[model_name]["mae"].append(metrics["mae"])
                self._model_scores[model_name]["rmse"].append(metrics["rmse"])
                self._model_scores[model_name]["r2"].append(metrics["r2"])

            # Select best model for this horizon
            if self.selection_criterion == "min_mape":
                best_model = min(horizon_scores.items(), key=lambda x: x[1]["mape"])[0]
            elif self.selection_criterion == "min_mae":
                best_model = min(horizon_scores.items(), key=lambda x: x[1]["mae"])[0]
            elif self.selection_criterion == "min_rmse":
                best_model = min(horizon_scores.items(), key=lambda x: x[1]["rmse"])[0]
            elif self.selection_criterion == "max_r2":
                best_model = max(horizon_scores.items(), key=lambda x: x[1]["r2"])[0]
            elif self.selection_criterion == "min_combined":
                # Compute combined score for this horizon
                combined_scores = {}
                for model_name, metrics in horizon_scores.items():
                    score = 0.0
                    if "mape" in self.combined_weights:
                        score += metrics["mape"] * self.combined_weights["mape"]
                    if "mae" in self.combined_weights:
                        score += metrics["mae"] * self.combined_weights["mae"]
                    if "rmse" in self.combined_weights:
                        score += metrics["rmse"] * self.combined_weights["rmse"]
                    combined_scores[model_name] = score
                best_model = min(combined_scores.items(), key=lambda x: x[1])[0]
            else:
                msg = f"Unknown selection criterion: {self.selection_criterion}"
                raise ValueError(msg)

            self._best_models_per_horizon[pred_col] = best_model

        # Average scores across horizons for get_model_scores()
        for model_name in self._model_scores:
            self._model_scores[model_name] = {
                "mape": float(np.mean(self._model_scores[model_name]["mape"])),
                "mae": float(np.mean(self._model_scores[model_name]["mae"])),
                "rmse": float(np.mean(self._model_scores[model_name]["rmse"])),
                "r2": float(np.mean(self._model_scores[model_name]["r2"])),
            }

        # Set equal weights for all models (not used, but required by base class)
        self._global_weights = {name: 1.0 / len(predictions) for name in predictions}

    def _combine_per_horizon(
        self,
        predictions: dict[str, pd.DataFrame],
        override_model: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """Combine predictions using per-horizon selection.

        Args:
            predictions: Model predictions.
            override_model: Optional override to use same model for all horizons.

        Returns:
            Tuple of (combined_df, selected_models_dict).
        """
        first_pred = next(iter(predictions.values()))
        combined_df = pd.DataFrame(index=first_pred.index)
        selected_models = {}

        for pred_col in self._best_models_per_horizon:
            if override_model is not None:
                selected_model = override_model
            else:
                selected_model = self._best_models_per_horizon[pred_col]

            # Handle missing model
            if selected_model not in predictions:
                selected_model = self._handle_missing_model(selected_model, predictions, [])

            combined_df[pred_col] = predictions[selected_model][pred_col]
            selected_models[pred_col] = selected_model

        return combined_df, selected_models

    def _handle_missing_model(
        self,
        missing_model: str,
        available_predictions: dict[str, pd.DataFrame],
        excluded_models: list[str],
    ) -> str:
        """Handle case where selected best model is not available.

        Args:
            missing_model: Name of the missing model.
            available_predictions: Available model predictions.
            excluded_models: Models that were excluded.

        Returns:
            Name of fallback model to use.

        Raises:
            KeyError: If fallback_strategy="error".
        """
        if self.fallback_strategy == "error":
            msg = (
                f"Best model {missing_model!r} not available in predictions. "
                f"Available: {list(available_predictions.keys())}, "
                f"Excluded: {excluded_models}"
            )
            raise KeyError(msg)

        if self.fallback_strategy == "second_best":
            # Use second-best model from ranking
            min_models_for_fallback = 2
            if len(self._ranked_models) < min_models_for_fallback:
                msg = (
                    f"Cannot use second-best model (only {len(self._ranked_models)} models). "
                    f"Missing model: {missing_model!r}"
                )
                raise KeyError(msg)

            for ranked_model in self._ranked_models[1:]:
                if ranked_model in available_predictions:
                    logger.warning(
                        "Best model %s not available, using second-best: %s",
                        missing_model,
                        ranked_model,
                    )
                    return ranked_model

            msg = f"No fallback model available. Missing: {missing_model!r}"
            raise KeyError(msg)

        if self.fallback_strategy == "simple_average":
            # Use first available model (simple_average would require combining)
            # For simplicity, just use first available
            fallback_model = next(iter(available_predictions.keys()))
            logger.warning(
                "Best model %s not available, using fallback: %s",
                missing_model,
                fallback_model,
            )
            return fallback_model

        msg = f"Unknown fallback strategy: {self.fallback_strategy}"
        raise ValueError(msg)

    @classmethod
    def load(cls, filepath: str | Path) -> "BestModelSelector":
        """Load selector from file.

        Deserializes a previously saved selector state.

        Args:
            filepath: Path to load file.

        Returns:
            Loaded BestModelSelector instance.

        Example:
            >>> selector = BestModelSelector.load("selectors/best_model.pkl")
        """
        filepath = Path(filepath)
        state = joblib.load(filepath)

        # Extract custom config
        custom_config = state["config"].get("custom_config", {})
        selection_criterion = custom_config.get("selection_criterion", "min_mape")
        per_horizon = custom_config.get("per_horizon", False)
        fallback_strategy = custom_config.get("fallback_strategy", "error")
        combined_weights = custom_config.get("combined_weights")

        # Create instance
        instance = cls(
            config=state["config"],
            selection_criterion=selection_criterion,
            per_horizon=per_horizon,
            fallback_strategy=fallback_strategy,
            combined_weights=combined_weights,
        )

        # Restore state
        instance._weights = state["weights"]
        instance._global_weights = state["global_weights"]
        instance._fitted = state["fitted"]
        instance._fit_timestamp = (
            datetime.fromisoformat(state["fit_timestamp"]) if state["fit_timestamp"] else None
        )
        instance._performance_history = state["performance_history"]
        instance._model_names = state["model_names"]

        # Restore selection-specific state
        instance._best_model = state.get("best_model")
        instance._best_models_per_horizon = state.get("best_models_per_horizon", {})
        instance._model_scores = state.get("model_scores", {})
        instance._ranked_models = state.get("ranked_models", [])

        logger.info("Loaded %s selector from %s", instance.name, filepath)

        return instance

    def save(self, filepath: str | Path) -> None:
        """Save selector state to file.

        Serializes the selector's selection, configuration, and metadata.

        Args:
            filepath: Path to save file (typically .pkl or .joblib).

        Example:
            >>> selector.save("selectors/best_model_v1.pkl")
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
            "best_model": self._best_model,
            "best_models_per_horizon": self._best_models_per_horizon.copy(),
            "model_scores": {
                model: metrics.copy() for model, metrics in self._model_scores.items()
            },
            "ranked_models": self._ranked_models.copy(),
        }

        joblib.dump(state, filepath)
        logger.info("Saved %s selector to %s", self.name, filepath)

    def reset(self) -> None:
        """Reset selector to unfitted state.

        Clears all learned weights, performance history, and selection results.
        """
        super().reset()
        self._best_model = None
        self._best_models_per_horizon = {}
        self._model_scores = {}
        self._ranked_models = []
        logger.info("Reset %s selector to unfitted state", self.name)
