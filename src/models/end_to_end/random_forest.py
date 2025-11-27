"""Random Forest model implementation for multi-horizon load forecasting.

This module provides the RandomForestModel class for electric load forecasting
across D+0 through D+8 horizons using Random Forest regression. It supports:
- Multi-horizon forecasting (D+0 to D+8)
- Horizon-safe feature selection to prevent temporal data leakage
- Out-of-bag error estimation
- Prediction confidence intervals from tree ensemble
- Parallel training across horizons

Example:
    ```python
    from src.models.end_to_end.random_forest import RandomForestModel
    from src.models.config.rf_config import RandomForestConfig
    import pandas as pd

    # Create and train model
    model = RandomForestModel()
    config = RandomForestConfig(n_estimators=200, max_features_per_horizon=50)
    model.fit(X_train, y_train, config.model_dump())

    # Make predictions with confidence intervals
    predictions = model.predict(X_test, horizons=[0, 1, 2])
    # Returns: h0, h0_std, h0_lower, h0_upper, h1, h1_std, ...

    # Get feature importance per horizon
    importance = model.get_feature_importance()

    # Save/load
    model.save("models/rf_model.joblib")
    loaded_model = RandomForestModel.load("models/rf_model.joblib")
    ```
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

from src.models.base import BaseModel, register_model_class
from src.models.config.rf_config import RandomForestConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@register_model_class("random_forest")
class RandomForestModel(BaseModel):
    """Random Forest model for multi-horizon electric load forecasting.

    This model uses Random Forest regression with horizon-specific feature selection
    to prevent temporal data leakage. It trains separate models for each forecast
    horizon (D+0 through D+8) and provides prediction confidence intervals based on
    the ensemble of decision trees.

    Key features:
    - Horizon-safe feature filtering (prevents using future information)
    - Per-horizon feature selection using importance thresholds
    - Out-of-bag (OOB) error estimation for model validation
    - 95% confidence intervals from tree predictions variance
    - Parallel training across horizons

    Attributes:
        models: Dictionary mapping horizon -> RandomForestRegressor instance.
        selected_features: Dictionary mapping horizon -> list of feature names used.
        oob_scores: Dictionary mapping horizon -> out-of-bag R² score.
        _config: RandomForestConfig instance with training configuration.
    """

    def __init__(self) -> None:
        """Initialize RandomForestModel with empty state."""
        super().__init__()
        self.models: dict[int, RandomForestRegressor] = {}
        self.selected_features: dict[int, list[str]] = {}
        self.oob_scores: dict[int, float] = {}
        self._config: RandomForestConfig | None = None
        logger.debug("Initialized RandomForestModel")

    @property
    def name(self) -> str:
        """Return model name identifier.

        Returns:
            Model name "random_forest".
        """
        return "random_forest"

    @property
    def version(self) -> str:
        """Return model version.

        Returns:
            Semantic version string.
        """
        return "1.0.0"

    @property
    def supported_horizons(self) -> list[int]:
        """Return supported forecast horizons.

        Returns:
            List containing [0, 1, 2, 3, 4, 5, 6, 7, 8] for D+0 through D+8.
        """
        return [0, 1, 2, 3, 4, 5, 6, 7, 8]

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series | pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        """Train Random Forest models for all horizons.

        This method:
        1. Validates input data
        2. Parses configuration into RandomForestConfig
        3. For each horizon:
           - Filters horizon-safe features
           - Selects top features using importance
           - Trains Random Forest with OOB scoring
        4. Stores training metadata and OOB scores

        Args:
            X: Training features DataFrame with shape (n_samples, n_features).
            y: Training target as Series (single column) or DataFrame.
               If DataFrame, uses first column.
            config: Model configuration dictionary. Should match RandomForestConfig schema.

        Raises:
            ValueError: If input data is invalid or incompatible.
            RuntimeError: If training fails.

        Example:
            ```python
            from src.models.config.rf_config import RandomForestConfig

            config = RandomForestConfig(n_estimators=200, max_features_per_horizon=50)
            model.fit(X_train, y_train, config.model_dump())
            ```
        """
        logger.info(
            "Starting RandomForestModel training with %d samples, %d features",
            len(X),
            len(X.columns),
        )

        # Validate inputs
        if X.empty:
            msg = "Training features X cannot be empty"
            raise ValueError(msg)

        if isinstance(y, pd.DataFrame):
            if y.shape[1] > 1:
                logger.warning("Target y has %d columns, using first column only", y.shape[1])
            y = y.iloc[:, 0]

        if len(X) != len(y):
            msg = f"X and y must have same length: {len(X)} != {len(y)}"
            raise ValueError(msg)

        # Parse configuration
        try:
            self._config = RandomForestConfig(**config)
        except Exception as e:
            msg = f"Invalid configuration: {e}"
            raise ValueError(msg) from e

        # Store original feature names
        self._feature_names = list(X.columns)

        # Train model for each horizon
        for horizon in self.supported_horizons:
            try:
                logger.info("Training model for horizon D+%d", horizon)
                self._train_horizon_model(X, y, horizon)
            except Exception as e:
                msg = f"Training failed for horizon {horizon}: {e}"
                raise RuntimeError(msg) from e

        # Mark as fitted
        self._is_fitted = True

        # Store training metadata
        self._training_metadata = {
            "trained_at": datetime.now(UTC),
            "training_samples": len(X),
            "training_features": len(X.columns),
            "oob_scores": self.oob_scores,
            "features_per_horizon": {h: len(feats) for h, feats in self.selected_features.items()},
            "training_config": self._config.model_dump(),
        }

        logger.info(
            "RandomForestModel training completed. OOB scores: %s",
            {f"h{h}": f"{score:.4f}" for h, score in self.oob_scores.items()},
        )

    def predict(
        self,
        X: pd.DataFrame,  # noqa: N803
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Generate predictions with confidence intervals for specified horizons.

        This method generates predictions using the ensemble of trees and calculates
        95% confidence intervals from the standard deviation across trees.

        Args:
            X: Features DataFrame with shape (n_samples, n_features).
            horizons: List of horizons to predict [0-8]. If None, predicts all
                     supported horizons. Must be subset of [0, 1, 2, 3, 4, 5, 6, 7, 8].

        Returns:
            DataFrame with predictions and confidence intervals.
            For each horizon h, includes columns:
            - h{h}: Point prediction
            - h{h}_std: Standard deviation across trees
            - h{h}_lower: Lower bound of 95% CI (pred - 1.96 * std)
            - h{h}_upper: Upper bound of 95% CI (pred + 1.96 * std)

        Raises:
            ValueError: If model not fitted or horizons are invalid.
            RuntimeError: If prediction fails.

        Example:
            ```python
            # Predict all horizons
            preds = model.predict(X_test)

            # Predict specific horizons
            preds = model.predict(X_test, horizons=[0, 1, 2])
            # Returns: h0, h0_std, h0_lower, h0_upper, h1, h1_std, h1_lower, h1_upper, ...
            ```
        """
        self._check_is_fitted()

        # Validate horizons
        if horizons is None:
            horizons = self.supported_horizons
        else:
            invalid = set(horizons) - set(self.supported_horizons)
            if invalid:
                msg = (
                    f"Invalid horizons {sorted(invalid)}. "
                    f"Supported horizons: {self.supported_horizons}"
                )
                raise ValueError(msg)

        logger.info("Generating predictions for %d samples, horizons %s", len(X), horizons)

        # Create predictions dataframe
        predictions = pd.DataFrame(index=X.index)

        for horizon in sorted(horizons):
            # Get horizon-specific features
            horizon_features = self.selected_features[horizon]
            X_horizon = X[horizon_features]  # noqa: N806

            try:
                # Get predictions from all trees
                model = self.models[horizon]
                tree_predictions = np.array(
                    [tree.predict(X_horizon) for tree in model.estimators_]
                )

                # Calculate mean and standard deviation across trees
                pred_mean = tree_predictions.mean(axis=0)
                pred_std = tree_predictions.std(axis=0)

                # 95% confidence interval (1.96 * std for normal distribution)
                pred_lower = pred_mean - 1.96 * pred_std
                pred_upper = pred_mean + 1.96 * pred_std

                # Add to predictions dataframe
                predictions[f"h{horizon}"] = pred_mean
                predictions[f"h{horizon}_std"] = pred_std
                predictions[f"h{horizon}_lower"] = pred_lower
                predictions[f"h{horizon}_upper"] = pred_upper

            except Exception as e:
                msg = f"Prediction failed for horizon {horizon}: {e}"
                raise RuntimeError(msg) from e

        logger.info("Predictions generated successfully for horizons %s", horizons)
        return predictions

    def get_feature_importance(self) -> dict[str, float]:
        """Return aggregated feature importance across all horizons.

        This method aggregates feature importance from all horizon models by
        averaging the importance values for each feature that appears in any
        horizon model.

        Returns:
            Dictionary mapping feature names to aggregated importance scores.
            Scores are normalized to sum to 1.0.

        Raises:
            ValueError: If model is not fitted.

        Example:
            ```python
            importance = model.get_feature_importance()
            # Returns: {"lag_48": 0.25, "hour": 0.15, "temperature": 0.10, ...}

            # Get per-horizon importance
            per_horizon = model.get_feature_importance_per_horizon()
            # Returns: {0: {"lag_48": 0.3, ...}, 1: {"lag_96": 0.4, ...}, ...}
            ```
        """
        self._check_is_fitted()

        # Aggregate importance across all horizons
        aggregated_importance: dict[str, list[float]] = {}

        for horizon, model in self.models.items():
            horizon_features = self.selected_features[horizon]
            importance_values = model.feature_importances_

            for feature, importance in zip(horizon_features, importance_values, strict=False):
                if feature not in aggregated_importance:
                    aggregated_importance[feature] = []
                aggregated_importance[feature].append(importance)

        # Average importance values for each feature
        averaged_importance = {
            feature: np.mean(values) for feature, values in aggregated_importance.items()
        }

        # Normalize to sum to 1.0
        total = sum(averaged_importance.values())
        if total > 0:
            averaged_importance = {k: v / total for k, v in averaged_importance.items()}

        logger.debug("Extracted aggregated feature importance for %d features", len(averaged_importance))
        return averaged_importance

    def get_feature_importance_per_horizon(self) -> dict[int, dict[str, float]]:
        """Return feature importance for each horizon separately.

        Returns:
            Dictionary mapping horizon -> {feature: importance}.
            Importance scores are normalized per horizon.

        Raises:
            ValueError: If model is not fitted.
        """
        self._check_is_fitted()

        per_horizon_importance = {}

        for horizon, model in self.models.items():
            horizon_features = self.selected_features[horizon]
            importance_values = model.feature_importances_

            # Create importance dict for this horizon
            horizon_importance = dict(zip(horizon_features, importance_values, strict=False))

            # Normalize to sum to 1.0
            total = sum(horizon_importance.values())
            if total > 0:
                horizon_importance = {k: v / total for k, v in horizon_importance.items()}

            per_horizon_importance[horizon] = horizon_importance

        logger.debug("Extracted per-horizon feature importance for %d horizons", len(per_horizon_importance))
        return per_horizon_importance

    def _train_horizon_model(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        horizon: int,
    ) -> None:
        """Train Random Forest model for a specific horizon.

        Args:
            X: Training features DataFrame.
            y: Training target Series.
            horizon: Forecast horizon (0-8).
        """
        # Get horizon-safe features
        safe_features = self._get_horizon_safe_features(list(X.columns), horizon)

        if not safe_features:
            msg = f"No horizon-safe features available for horizon {horizon}"
            raise ValueError(msg)

        logger.debug(
            "Horizon %d: %d/%d features are horizon-safe",
            horizon,
            len(safe_features),
            len(X.columns),
        )

        # Filter to safe features
        X_safe = X[safe_features]  # noqa: N806

        # Select top features for this horizon
        selected_features = self._select_features_for_horizon(X_safe, y, horizon)
        X_selected = X_safe[selected_features]  # noqa: N806

        logger.debug(
            "Horizon %d: Selected %d features after importance filtering",
            horizon,
            len(selected_features),
        )

        # Create target for this horizon
        y_horizon = self._create_horizon_target(y, horizon)

        # Ensure X and y are aligned
        if len(X_selected) != len(y_horizon):
            min_len = min(len(X_selected), len(y_horizon))
            X_selected = X_selected.iloc[:min_len]  # noqa: N806
            y_horizon = y_horizon.iloc[:min_len]

        # Train Random Forest model
        sklearn_params = self._config.to_sklearn_params()
        model = RandomForestRegressor(**sklearn_params)

        logger.debug("Training RandomForest for horizon %d with %d samples", horizon, len(X_selected))
        model.fit(X_selected, y_horizon)

        # Store model and metadata
        self.models[horizon] = model
        self.selected_features[horizon] = selected_features

        # Store OOB score if available
        if self._config.oob_score and self._config.bootstrap:
            self.oob_scores[horizon] = model.oob_score_
            logger.debug("Horizon %d OOB score: %.4f", horizon, model.oob_score_)
        else:
            self.oob_scores[horizon] = np.nan

    def _get_horizon_safe_features(self, all_features: list[str], horizon: int) -> list[str]:
        """Filter features to prevent temporal data leakage for a given horizon.

        This method implements horizon-safe feature selection by:
        1. Rejecting features with "future", "forward", or "actual" keywords
        2. For lag features, ensuring lag >= horizon * periods_per_day
        3. Logging rejected features for debugging

        Args:
            all_features: List of all available feature names.
            horizon: Forecast horizon (0-8).

        Returns:
            List of horizon-safe feature names.
        """
        safe_features = []
        rejected_features = []

        # Calculate minimum required lag for this horizon
        min_lag = horizon * self._config.periods_per_day

        # Keywords that indicate future information
        future_keywords = ["future", "forward", "actual"]

        for feature in all_features:
            feature_lower = feature.lower()

            # Reject features with future keywords
            if any(keyword in feature_lower for keyword in future_keywords):
                rejected_features.append((feature, "contains future keyword"))
                continue

            # Check lag features
            # Patterns: "lag_N" or "lag_Nh" where N is the lag period
            lag_match = re.search(r"lag[_-]?(\d+)h?", feature_lower)
            if lag_match:
                lag_value = int(lag_match.group(1))
                if lag_value < min_lag:
                    rejected_features.append((feature, f"lag {lag_value} < required {min_lag}"))
                    continue

            # Feature is safe
            safe_features.append(feature)

        # Log rejected features
        if rejected_features:
            max_log_features = 5
            logger.debug(
                "Horizon %d: Rejected %d features due to temporal leakage:",
                horizon,
                len(rejected_features),
            )
            for feature, reason in rejected_features[:max_log_features]:
                logger.debug("  - %s: %s", feature, reason)
            if len(rejected_features) > max_log_features:
                logger.debug("  ... and %d more", len(rejected_features) - max_log_features)

        return safe_features

    def _select_features_for_horizon(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
        horizon: int,
    ) -> list[str]:
        """Select top features for a specific horizon using importance.

        Args:
            X: Features DataFrame (already filtered to horizon-safe features).
            y: Target Series.
            horizon: Forecast horizon (0-8).

        Returns:
            List of selected feature names.
        """
        # If fewer features than max, use all
        if len(X.columns) <= self._config.max_features_per_horizon:
            return list(X.columns)

        # Create target for this horizon
        y_horizon = self._create_horizon_target(y, horizon)

        # Ensure alignment
        if len(X) != len(y_horizon):
            min_len = min(len(X), len(y_horizon))
            X = X.iloc[:min_len]  # noqa: N806
            y_horizon = y_horizon.iloc[:min_len]

        # Train temporary Random Forest for feature selection
        sklearn_params = self._config.to_sklearn_params()
        # Use fewer trees for feature selection (faster)
        temp_model = RandomForestRegressor(
            n_estimators=min(50, sklearn_params["n_estimators"]),
            max_depth=sklearn_params["max_depth"],
            random_state=sklearn_params["random_state"],
            n_jobs=sklearn_params["n_jobs"],
        )

        logger.debug("Training temporary model for feature selection (horizon %d)", horizon)
        temp_model.fit(X, y_horizon)

        # Use SelectFromModel with importance threshold
        selector = SelectFromModel(
            temp_model,
            threshold=self._config.importance_threshold,
            max_features=self._config.max_features_per_horizon,
            prefit=True,
        )

        # Get selected feature mask
        feature_mask = selector.get_support()
        selected_features = [feat for feat, selected in zip(X.columns, feature_mask, strict=False) if selected]

        # Ensure at least some features are selected
        if not selected_features:
            logger.warning(
                "No features selected with threshold %.3f, using top %d by importance",
                self._config.importance_threshold,
                min(10, len(X.columns)),
            )
            # Fall back to top N by importance
            importance = temp_model.feature_importances_
            top_indices = np.argsort(importance)[-min(10, len(X.columns)):]
            selected_features = [X.columns[i] for i in top_indices]

        return selected_features

    def _create_horizon_target(self, y: pd.Series, horizon: int) -> pd.Series:
        """Create target variable shifted by horizon periods.

        Args:
            y: Original target Series.
            horizon: Forecast horizon (0-8).

        Returns:
            Shifted target Series for the specified horizon.
        """
        if horizon == 0:
            # D+0: no shift
            return y.copy()

        # Shift target forward by horizon * periods_per_day
        shift_periods = horizon * self._config.periods_per_day
        y_shifted = y.shift(-shift_periods)

        # Drop NaN values created by shift
        y_shifted = y_shifted.dropna()

        logger.debug(
            "Created horizon %d target: shifted by %d periods, %d -> %d samples",
            horizon,
            shift_periods,
            len(y),
            len(y_shifted),
        )

        return y_shifted

    def save(self, path: str | Path) -> None:
        """Save model to disk using joblib.

        This method saves:
        - All trained Random Forest models (one per horizon)
        - Selected features per horizon
        - OOB scores
        - Configuration and metadata

        Args:
            path: File path to save the model.

        Raises:
            ValueError: If model is not fitted.
            OSError: If file cannot be written.
        """
        self._check_is_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data to save
        save_data = {
            "models": self.models,
            "selected_features": self.selected_features,
            "oob_scores": self.oob_scores,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
            "training_metadata": self._training_metadata,
            "config": self._config.model_dump() if self._config else {},
        }

        try:
            joblib.dump(save_data, path)
            logger.info("Saved RandomForestModel to %s", path)
        except Exception as e:
            msg = f"Failed to save model to {path}: {e}"
            raise OSError(msg) from e

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestModel":
        """Load model from disk using joblib.

        Args:
            path: File path to load the model from.

        Returns:
            Loaded RandomForestModel instance.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            ValueError: If loaded object is invalid.
            OSError: If file cannot be read.
        """
        path = Path(path)

        if not path.exists():
            msg = f"Model file not found: {path}"
            raise FileNotFoundError(msg)

        try:
            save_data = joblib.load(path)
        except Exception as e:
            msg = f"Failed to load model from {path}: {e}"
            raise OSError(msg) from e

        # Create model instance and restore state
        model = cls()
        model.models = save_data["models"]
        model.selected_features = save_data["selected_features"]
        model.oob_scores = save_data["oob_scores"]
        model._feature_names = save_data["feature_names"]
        model._is_fitted = save_data["is_fitted"]
        model._training_metadata = save_data["training_metadata"]

        # Restore config
        if save_data["config"]:
            model._config = RandomForestConfig(**save_data["config"])

        logger.info("Loaded RandomForestModel from %s", path)
        return model
