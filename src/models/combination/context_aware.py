"""Context-aware weight adaptation combiner.

This module provides a combiner that dynamically adjusts combination weights
based on external conditions and forecasting context. Uses temporal features
(hour, day of week, month), load characteristics (level, volatility), and
optional meteorological features to adapt weights.

The module supports multiple adaptation strategies:
- Ridge: Linear adaptation with regularization
- Random Forest: Non-linear pattern learning

Key Features:
- Context feature extraction (temporal, load, meteorological)
- Multiple adaptation models (Ridge, Random Forest)
- Weight constraint enforcement (sum=1, bounds)
- Feature importance analysis for interpretability
- Integration with BaseCombiner interface

Example:
    ```python
    from src.models.combination import ContextAwareCombiner

    # Create combiner with Ridge adaptation
    combiner = ContextAwareCombiner(config={
        'adaptation_method': 'ridge',
        'use_temporal_features': True,
        'use_load_features': True
    })

    # Fit on training data
    combiner.fit(train_predictions, train_targets)

    # Combine test predictions
    result = combiner.combine(test_predictions)

    # Get feature importance
    importance = combiner.get_feature_importance()
    ```
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.combination.base_combiner import BaseCombiner
from src.models.combination.data_structures import CombinationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContextAwareConfig:
    """Configuration for context-aware combiner.

    Attributes:
        adaptation_method: Method for weight adaptation ('ridge', 'random_forest').
        ridge_alpha: Regularization strength for Ridge regression.
        rf_n_estimators: Number of trees for Random Forest.
        rf_max_depth: Maximum depth for Random Forest trees.
        use_temporal_features: Whether to use temporal features.
        use_load_features: Whether to use load characteristic features (requires
            load data to be passed during both fit and combine).
        use_meteo_features: Whether to use meteorological features (requires
            meteo data to be passed during both fit and combine).
        min_weight: Minimum allowed weight for any model.
        max_weight: Maximum allowed weight for any model.
        weight_smoothing: Smoothing factor for weight changes (0-1).
        optimization_max_iter: Maximum iterations for weight optimization.
    """

    adaptation_method: str = "ridge"
    ridge_alpha: float = 1.0
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    use_temporal_features: bool = True
    use_load_features: bool = False  # Default False since not available during combine
    use_meteo_features: bool = False
    min_weight: float = 0.01
    max_weight: float = 0.95
    weight_smoothing: float = 0.0
    optimization_max_iter: int = 100


@dataclass
class ContextAnalysis:
    """Analysis of context-weight relationships.

    Attributes:
        correlations: Correlation between context features and weights.
        top_features: Top influential features per model.
        stability_scores: Stability of context-weight relationships.
        relationship_summary: Summary of relationships found.
    """

    correlations: dict[str, dict[str, float]] = field(default_factory=dict)
    top_features: dict[str, list[str]] = field(default_factory=dict)
    stability_scores: dict[str, float] = field(default_factory=dict)
    relationship_summary: str = ""


class ContextFeatureExtractor:
    """Extract context features for weight adaptation.

    Extracts temporal, load characteristic, and meteorological features
    from timestamps and data for use in context-aware weight adaptation.

    Attributes:
        config: Configuration dictionary.
        scaler: StandardScaler for feature normalization.
        feature_names: List of extracted feature names.
        is_fitted: Whether the scaler has been fitted.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize feature extractor.

        Args:
            config: Configuration dictionary with feature flags.
        """
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_fitted = False

    def extract(
        self,
        timestamps: pd.DatetimeIndex,
        load_data: pd.Series | None = None,
        meteo_data: pd.DataFrame | None = None,
        fit_scaler: bool = False,
    ) -> pd.DataFrame:
        """Extract context features from timestamps and data.

        Args:
            timestamps: Datetime index for feature extraction.
            load_data: Optional load values for load features.
            meteo_data: Optional meteorological data.
            fit_scaler: Whether to fit the scaler on this data.

        Returns:
            DataFrame with extracted context features.
        """
        features = pd.DataFrame(index=timestamps)

        # Temporal features
        if self.config.get("use_temporal_features", True):
            features = self._extract_temporal_features(features, timestamps)

        # Load characteristics
        if self.config.get("use_load_features", True) and load_data is not None:
            features = self._extract_load_features(features, load_data)

        # Meteorological features
        if self.config.get("use_meteo_features", False) and meteo_data is not None:
            features = self._extract_meteo_features(features, meteo_data)

        # Fill any remaining NaN
        features = features.fillna(0)

        self.feature_names = list(features.columns)

        # Optionally fit and transform with scaler
        if fit_scaler:
            self.is_fitted = True
            return pd.DataFrame(
                self.scaler.fit_transform(features),
                index=features.index,
                columns=features.columns,
            )
        if self.is_fitted:
            return pd.DataFrame(
                self.scaler.transform(features),
                index=features.index,
                columns=features.columns,
            )

        return features

    def _extract_temporal_features(
        self,
        features: pd.DataFrame,
        timestamps: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Extract temporal features from timestamps.

        Args:
            features: Feature DataFrame to add to.
            timestamps: Datetime index.

        Returns:
            DataFrame with temporal features added.
        """
        # Raw temporal features
        features["hour"] = timestamps.hour
        features["day_of_week"] = timestamps.dayofweek
        features["month"] = timestamps.month
        weekend_start_day = 5  # Saturday
        features["is_weekend"] = (timestamps.dayofweek >= weekend_start_day).astype(int)

        # Cyclic encoding for hour
        features["hour_sin"] = np.sin(2 * np.pi * timestamps.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * timestamps.hour / 24)

        # Cyclic encoding for day of week
        features["dow_sin"] = np.sin(2 * np.pi * timestamps.dayofweek / 7)
        features["dow_cos"] = np.cos(2 * np.pi * timestamps.dayofweek / 7)

        # Cyclic encoding for month
        features["month_sin"] = np.sin(2 * np.pi * (timestamps.month - 1) / 12)
        features["month_cos"] = np.cos(2 * np.pi * (timestamps.month - 1) / 12)

        # Quarter
        features["quarter"] = timestamps.quarter

        return features

    def _extract_load_features(
        self,
        features: pd.DataFrame,
        load_data: pd.Series,
    ) -> pd.DataFrame:
        """Extract load characteristic features.

        Args:
            features: Feature DataFrame to add to.
            load_data: Load time series.

        Returns:
            DataFrame with load features added.
        """
        # Align load_data with features index
        load_aligned = load_data.reindex(features.index)

        # Load level (relative to mean)
        load_mean = load_aligned.mean()
        if load_mean > 0:
            features["load_level"] = load_aligned / load_mean
        else:
            features["load_level"] = 0.0

        # Load volatility (rolling std)
        features["load_volatility"] = load_aligned.rolling(window=48, min_periods=1).std().fillna(0)

        # Load trend (recent direction)
        features["load_trend"] = load_aligned.diff(periods=1).fillna(0)

        # Load regime (categorized)
        p25 = load_aligned.quantile(0.25)
        p75 = load_aligned.quantile(0.75)
        features["load_regime_low"] = (load_aligned < p25).astype(int)
        features["load_regime_high"] = (load_aligned > p75).astype(int)

        return features

    def _extract_meteo_features(
        self,
        features: pd.DataFrame,
        meteo_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract meteorological features.

        Args:
            features: Feature DataFrame to add to.
            meteo_data: Meteorological data DataFrame.

        Returns:
            DataFrame with meteorological features added.
        """
        if "temperature" in meteo_data.columns:
            temp = meteo_data["temperature"].reindex(features.index)
            features["temperature"] = temp.fillna(temp.mean())

            # Temperature deviation from mean
            temp_mean = temp.mean()
            features["temp_deviation"] = (temp - temp_mean).fillna(0)

            # Extreme temperature flags
            features["temp_extreme_high"] = (temp > temp.quantile(0.95)).astype(int)
            features["temp_extreme_low"] = (temp < temp.quantile(0.05)).astype(int)

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names.

        Returns:
            List of feature names.
        """
        return self.feature_names.copy()


class ContextAwareCombiner(BaseCombiner):
    """Context-aware weight adaptation combiner.

    Adjusts combination weights based on external conditions and forecasting
    context. Uses temporal features, load characteristics, and optional
    meteorological features to adapt weights dynamically.

    Adaptation Function:
        w(context) = adaptation_model(context_features)

    Context Features:
        - Temporal: hour, day_of_week, month, is_weekend (with cyclic encoding)
        - Load: load_level, volatility, trend, regime
        - Meteorological: temperature, deviations (optional)

    Adaptation Models:
        - Ridge: Linear adaptation with L2 regularization
        - Random Forest: Non-linear pattern learning

    Weight Constraints:
        - sum(w_i) = 1
        - w_i in [min_weight, max_weight]

    Attributes:
        adaptation_method: The adaptation method ('ridge' or 'random_forest').
        feature_extractor: ContextFeatureExtractor instance.
        adaptation_models: Trained adaptation models per model.
        base_weights: Average optimal weights per model.

    Example:
        >>> combiner = ContextAwareCombiner(config={'adaptation_method': 'ridge'})
        >>> combiner.fit(train_predictions, train_targets)
        >>> result = combiner.combine(test_predictions)
        >>> print(combiner.get_feature_importance())
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize context-aware combiner.

        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)

        # Parse config - use custom_config for context-aware specific settings
        config_dict = {}
        if hasattr(self.config, "to_dict"):
            base_config = self.config.to_dict()
            config_dict = base_config.get("custom_config", {})

        # Also check the original config dict if provided
        if isinstance(config, dict):
            # Merge original config for context-aware specific params
            for key in [
                "adaptation_method",
                "ridge_alpha",
                "rf_n_estimators",
                "rf_max_depth",
                "use_temporal_features",
                "use_load_features",
                "use_meteo_features",
                "min_weight",
                "max_weight",
                "weight_smoothing",
                "optimization_max_iter",
            ]:
                if key in config:
                    config_dict[key] = config[key]

        config_obj = ContextAwareConfig(**{k: v for k, v in config_dict.items() if hasattr(ContextAwareConfig, k)})

        self.adaptation_method = config_obj.adaptation_method
        self.ridge_alpha = config_obj.ridge_alpha
        self.rf_n_estimators = config_obj.rf_n_estimators
        self.rf_max_depth = config_obj.rf_max_depth
        self.min_weight = config_obj.min_weight
        self.max_weight = config_obj.max_weight
        self.weight_smoothing = config_obj.weight_smoothing
        self.optimization_max_iter = config_obj.optimization_max_iter

        # Store feature config for consistent feature extraction
        self._feature_config = {
            "use_temporal_features": config_obj.use_temporal_features,
            "use_load_features": config_obj.use_load_features,
            "use_meteo_features": config_obj.use_meteo_features,
        }
        self._has_load_features = False  # Track if load features were used during fit

        # Initialize components
        self.feature_extractor = ContextFeatureExtractor(self._feature_config)
        self.adaptation_models: dict[str, Any] = {}
        self.base_weights: dict[str, float] = {}
        self._previous_weights: dict[str, float] | None = None

        logger.debug(
            "Initialized ContextAwareCombiner with method=%s",
            self.adaptation_method,
        )

    @property
    def name(self) -> str:
        """Return combiner name.

        Returns:
            Combiner name.
        """
        return "context_aware"

    @property
    def version(self) -> str:
        """Return combiner version.

        Returns:
            Version string.
        """
        return "1.0.0"

    def fit(
        self,
        train_predictions: dict[str, pd.DataFrame],
        train_targets: pd.DataFrame,
        validation_predictions: dict[str, pd.DataFrame] | None = None,  # noqa: ARG002
        validation_targets: pd.DataFrame | None = None,  # noqa: ARG002
    ) -> None:
        """Fit context-aware adaptation model.

        Learns the relationship between context features and optimal weights
        by finding optimal weights for each training context and training
        an adaptation model to predict weights from context.

        Args:
            train_predictions: Base model predictions on training set.
            train_targets: True target values for training period.
            validation_predictions: Optional validation predictions (not used).
            validation_targets: Optional validation targets (not used).
        """
        logger.info("Fitting context-aware combiner with method=%s", self.adaptation_method)

        # Validate inputs
        self._validate_predictions(train_predictions)

        model_names = list(train_predictions.keys())
        self._model_names = model_names
        first_pred = train_predictions[model_names[0]]

        # Get load data if available and configured
        load_data = None
        if self._feature_config.get("use_load_features", True):
            load_data = self._get_load_data(train_targets)
            self._has_load_features = load_data is not None
        else:
            self._has_load_features = False

        # Extract context features from timestamps
        context_features = self.feature_extractor.extract(
            timestamps=first_pred.index,
            load_data=load_data,
            fit_scaler=True,
        )

        logger.info("Extracted %d context features", len(context_features.columns))

        # Calculate optimal weights for each context
        optimal_weights = self._calculate_optimal_weights_by_context(
            train_predictions,
            train_targets,
            context_features,
        )

        # Train adaptation model for each model's weight
        for model_name in model_names:
            logger.debug("Training adaptation model for %s", model_name)

            # Target: optimal weight for this model
            y_target = optimal_weights[model_name].to_numpy()
            x_features = context_features.to_numpy()

            # Create and train adaptation model
            adaptation_model = self._create_adaptation_model()
            adaptation_model.fit(x_features, y_target)

            self.adaptation_models[model_name] = adaptation_model

        # Store base weights (mean optimal weights)
        self.base_weights = {model: float(optimal_weights[model].mean()) for model in model_names}

        # Set global weights for base class
        self._global_weights = self.base_weights.copy()

        # Mark as fitted
        self._fitted = True
        self._fit_timestamp = datetime.now(tz=UTC)

        logger.info("Context-aware combiner trained successfully")

    def combine(
        self,
        predictions: dict[str, pd.DataFrame],
        metadata: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> CombinationResult:
        """Combine predictions with context-aware weights.

        Extracts context features from timestamps and applies trained
        adaptation models to predict context-specific weights.

        Args:
            predictions: Base model predictions.
            metadata: Optional metadata (not used).

        Returns:
            CombinationResult with combined predictions and adapted weights.

        Raises:
            RuntimeError: If combiner has not been fitted.
        """
        if not self.is_fitted:
            msg = "Context-aware combiner must be fitted before combining"
            raise RuntimeError(msg)

        start_time = time.time()

        logger.info("Combining %d models with context-aware weights", len(predictions))

        # Validate predictions
        self._validate_predictions(predictions)

        first_pred = next(iter(predictions.values()))
        pred_cols = [col for col in first_pred.columns if col.startswith("pred_h")]

        # Extract context features for current data
        # Use same feature configuration as during fit
        # Note: During combine, we may not have load_data, so the feature extractor
        # needs to handle this - we temporarily disable load features during combine
        # if they weren't actually used during training or are unavailable
        temp_config = self._feature_config.copy()
        if not self._has_load_features:
            temp_config["use_load_features"] = False
        self.feature_extractor.config = temp_config

        context_features = self.feature_extractor.extract(timestamps=first_pred.index)

        # Predict adapted weights for each timestamp
        adapted_weights_by_timestamp = self._predict_adapted_weights(context_features, list(predictions.keys()))

        # Combine predictions with adapted weights
        combined_df = pd.DataFrame(index=first_pred.index)

        for col in pred_cols:
            combined_values = np.zeros(len(first_pred))

            for i in range(len(first_pred)):
                timestamp_weights = adapted_weights_by_timestamp[i]

                weighted_sum = sum(
                    timestamp_weights.get(model, 0.0) * predictions[model][col].iloc[i] for model in predictions
                )

                combined_values[i] = weighted_sum

            combined_df[col] = combined_values

        # Calculate average weights for metadata
        avg_weights = {}
        for model in predictions:
            model_weights = [w.get(model, 0.0) for w in adapted_weights_by_timestamp]
            avg_weights[model] = float(np.mean(model_weights))

        # Create metadata
        processing_time = time.time() - start_time
        metadata_obj = self._create_metadata(
            models_used=list(predictions.keys()),
            models_excluded=[],
            processing_time=processing_time,
        )
        metadata_obj.configuration["adaptation_method"] = self.adaptation_method

        return CombinationResult(
            combined_predictions=combined_df,
            weights=avg_weights,
            metadata=metadata_obj,
        )

    def _create_adaptation_model(self) -> Any:
        """Create adaptation model based on configuration.

        Returns:
            Scikit-learn model instance.

        Raises:
            ValueError: If unknown adaptation method specified.
        """
        if self.adaptation_method == "ridge":
            return Ridge(alpha=self.ridge_alpha)

        if self.adaptation_method == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                random_state=42,
                n_jobs=-1,
            )

        msg = f"Unknown adaptation method: {self.adaptation_method}"
        raise ValueError(msg)

    def _calculate_optimal_weights_by_context(
        self,
        predictions: dict[str, pd.DataFrame],
        targets: pd.DataFrame,
        context_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate optimal weights for each timestamp.

        Uses optimization to find the best weights for each context
        by minimizing the prediction error.

        Args:
            predictions: Model predictions.
            targets: True targets.
            context_features: Context features.

        Returns:
            DataFrame with optimal weights per timestamp.
        """
        model_names = list(predictions.keys())
        n_models = len(model_names)

        pred_cols = [col for col in next(iter(predictions.values())).columns if col.startswith("pred_h")]

        optimal_weights = pd.DataFrame(index=context_features.index)

        # Initialize with equal weights
        for model in model_names:
            optimal_weights[model] = 1.0 / n_models

        # For each timestamp, find optimal weights
        for idx in range(len(context_features)):
            # Get predictions and targets for this timestamp
            preds_array = []
            targets_array = []

            for col in pred_cols:
                horizon = int(col.split("_h")[1])
                target_col = f"target_h{horizon}"

                if target_col in targets.columns:
                    pred_vals = [predictions[m][col].iloc[idx] for m in model_names]
                    target_val = targets[target_col].iloc[idx]

                    if not np.isnan(target_val) and not any(np.isnan(pred_vals)):
                        preds_array.append(pred_vals)
                        targets_array.append(target_val)

            if len(preds_array) == 0:
                # No valid targets, use equal weights
                continue

            preds_np = np.array(preds_array)  # (n_horizons, n_models)
            targets_np = np.array(targets_array)  # (n_horizons,)

            # Optimize weights for this timestamp using a closure to bind variables
            def _make_objective(p_arr: np.ndarray, t_arr: np.ndarray):
                def objective(w: np.ndarray) -> float:
                    combined = p_arr @ w
                    return float(np.mean(np.abs(t_arr - combined)))

                return objective

            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
            bounds = [(0.0, 1.0) for _ in range(n_models)]

            result = minimize(
                _make_objective(preds_np, targets_np),
                x0=np.ones(n_models) / n_models,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": self.optimization_max_iter},
            )

            # Store optimal weights
            for j, model in enumerate(model_names):
                optimal_weights.loc[context_features.index[idx], model] = result.x[j]

        return optimal_weights

    def _predict_adapted_weights(
        self,
        context_features: pd.DataFrame,
        model_names: list[str],
    ) -> list[dict[str, float]]:
        """Predict adapted weights for given contexts.

        Args:
            context_features: Context features.
            model_names: List of model names to get weights for.

        Returns:
            List of weight dictionaries per timestamp.
        """
        x_features = context_features.to_numpy()
        adapted_weights = []

        for i in range(len(x_features)):
            timestamp_weights = {}

            for model_name in model_names:
                if model_name in self.adaptation_models:
                    predicted_weight = self.adaptation_models[model_name].predict(x_features[i : i + 1])[0]
                    timestamp_weights[model_name] = predicted_weight
                else:
                    # Model not in trained adaptation models, use base weight
                    timestamp_weights[model_name] = self.base_weights.get(model_name, 1.0 / len(model_names))

            # Enforce constraints
            timestamp_weights = self._enforce_weight_constraints(timestamp_weights)

            # Apply smoothing if configured
            if self.weight_smoothing > 0 and self._previous_weights is not None:
                timestamp_weights = self._apply_weight_smoothing(timestamp_weights)

            self._previous_weights = timestamp_weights.copy()
            adapted_weights.append(timestamp_weights)

        return adapted_weights

    def _enforce_weight_constraints(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Enforce weight constraints (sum=1, bounds).

        Args:
            weights: Raw predicted weights.

        Returns:
            Constrained weights that sum to 1.
        """
        # Clip to bounds
        constrained = {model: np.clip(w, self.min_weight, self.max_weight) for model, w in weights.items()}

        # Normalize to sum to 1
        total = sum(constrained.values())
        if total > 0:
            constrained = {model: w / total for model, w in constrained.items()}
        else:
            # Fallback to equal weights
            n = len(constrained)
            constrained = dict.fromkeys(constrained.keys(), 1.0 / n)

        return constrained

    def _apply_weight_smoothing(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply exponential smoothing to weights.

        Args:
            weights: Current weights.

        Returns:
            Smoothed weights.
        """
        if self._previous_weights is None:
            return weights

        alpha = 1.0 - self.weight_smoothing
        smoothed = {}

        for model, weight in weights.items():
            prev_weight = self._previous_weights.get(model, weight)
            smoothed[model] = alpha * weight + self.weight_smoothing * prev_weight

        # Renormalize after smoothing
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {m: w / total for m, w in smoothed.items()}

        return smoothed

    def _get_load_data(self, targets: pd.DataFrame) -> pd.Series | None:
        """Extract load data from targets for feature extraction.

        Args:
            targets: Target DataFrame.

        Returns:
            Load series if available, None otherwise.
        """
        # Try to get first target column as load proxy
        target_cols = [col for col in targets.columns if col.startswith("target_h")]
        if target_cols:
            return targets[target_cols[0]]
        return None

    def get_feature_importance(self) -> dict[str, dict[str, float]]:
        """Get feature importance for weight adaptation.

        Returns:
            Dictionary mapping model names to feature importance dictionaries.
        """
        if not self.is_fitted:
            return {}

        importances: dict[str, dict[str, float]] = {}

        for model_name, adaptation_model in self.adaptation_models.items():
            if hasattr(adaptation_model, "coef_"):
                # Ridge: use coefficients (absolute value for importance)
                importances[model_name] = {
                    feature: float(abs(coef))
                    for feature, coef in zip(
                        self.feature_extractor.feature_names,
                        adaptation_model.coef_,
                        strict=False,
                    )
                }
            elif hasattr(adaptation_model, "feature_importances_"):
                # Random Forest: use feature_importances_
                importances[model_name] = {
                    feature: float(imp)
                    for feature, imp in zip(
                        self.feature_extractor.feature_names,
                        adaptation_model.feature_importances_,
                        strict=False,
                    )
                }

        return importances

    def get_top_features(self, n: int = 5) -> dict[str, list[tuple[str, float]]]:
        """Get top N most important features per model.

        Args:
            n: Number of top features to return.

        Returns:
            Dictionary mapping model names to list of (feature, importance) tuples.
        """
        importances = self.get_feature_importance()
        top_features: dict[str, list[tuple[str, float]]] = {}

        for model_name, feat_imp in importances.items():
            sorted_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
            top_features[model_name] = sorted_features[:n]

        return top_features

    def analyze_context_relationships(
        self,
        context_features: pd.DataFrame,
        optimal_weights: pd.DataFrame,
    ) -> ContextAnalysis:
        """Analyze relationships between context and weights.

        Args:
            context_features: Context features DataFrame.
            optimal_weights: Optimal weights DataFrame.

        Returns:
            ContextAnalysis with correlation and relationship information.
        """
        analysis = ContextAnalysis()

        # Calculate correlations
        for model in optimal_weights.columns:
            model_corrs = {}
            for feature in context_features.columns:
                corr = np.corrcoef(context_features[feature].values, optimal_weights[model].values)[0, 1]
                if not np.isnan(corr):
                    model_corrs[feature] = float(corr)
            analysis.correlations[model] = model_corrs

            # Get top features by absolute correlation
            sorted_features = sorted(model_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
            analysis.top_features[model] = [f for f, _ in sorted_features[:5]]

        # Calculate stability (std of weights)
        for model in optimal_weights.columns:
            analysis.stability_scores[model] = float(optimal_weights[model].std())

        # Generate summary
        summary_lines = ["Context-Weight Relationship Analysis:"]
        for model in optimal_weights.columns:
            top = analysis.top_features.get(model, [])[:3]
            summary_lines.append(f"  {model}: Top features: {top}")

        analysis.relationship_summary = "\n".join(summary_lines)

        return analysis

    def get_adaptation_diagnostics(self) -> dict[str, Any]:
        """Get diagnostics about the adaptation models.

        Returns:
            Dictionary with diagnostic information.
        """
        if not self.is_fitted:
            return {}

        diagnostics = {
            "adaptation_method": self.adaptation_method,
            "n_models": len(self.adaptation_models),
            "n_features": len(self.feature_extractor.feature_names),
            "feature_names": self.feature_extractor.feature_names,
            "base_weights": self.base_weights,
        }

        # Add method-specific diagnostics
        if self.adaptation_method == "ridge":
            diagnostics["ridge_alpha"] = self.ridge_alpha
        elif self.adaptation_method == "random_forest":
            diagnostics["rf_n_estimators"] = self.rf_n_estimators
            diagnostics["rf_max_depth"] = self.rf_max_depth

        return diagnostics

    def save(self, filepath: str | Path) -> None:
        """Save combiner state to file.

        Args:
            filepath: Path to save file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "name": self.name,
            "version": self.version,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
            "adaptation_method": self.adaptation_method,
            "ridge_alpha": self.ridge_alpha,
            "rf_n_estimators": self.rf_n_estimators,
            "rf_max_depth": self.rf_max_depth,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "weight_smoothing": self.weight_smoothing,
            "adaptation_models": self.adaptation_models,
            "base_weights": self.base_weights,
            "feature_extractor_config": self.feature_extractor.config,
            "feature_extractor_names": self.feature_extractor.feature_names,
            "feature_extractor_scaler": self.feature_extractor.scaler,
            "feature_extractor_fitted": self.feature_extractor.is_fitted,
            "fitted": self._fitted,
            "fit_timestamp": self._fit_timestamp.isoformat() if self._fit_timestamp else None,
            "model_names": self._model_names,
            "global_weights": self._global_weights,
            "feature_config": self._feature_config,
            "has_load_features": self._has_load_features,
        }

        joblib.dump(state, filepath)
        logger.info("Saved context-aware combiner to %s", filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> ContextAwareCombiner:
        """Load combiner from file.

        Args:
            filepath: Path to load from.

        Returns:
            Loaded combiner instance.
        """
        filepath = Path(filepath)
        state = joblib.load(filepath)

        # Create instance with saved config
        instance = cls(config=state.get("config", {}))

        # Restore state
        instance.adaptation_method = state["adaptation_method"]
        instance.ridge_alpha = state["ridge_alpha"]
        instance.rf_n_estimators = state["rf_n_estimators"]
        instance.rf_max_depth = state["rf_max_depth"]
        instance.min_weight = state["min_weight"]
        instance.max_weight = state["max_weight"]
        instance.weight_smoothing = state["weight_smoothing"]
        instance.adaptation_models = state["adaptation_models"]
        instance.base_weights = state["base_weights"]

        # Restore feature extractor
        instance.feature_extractor.config = state["feature_extractor_config"]
        instance.feature_extractor.feature_names = state["feature_extractor_names"]
        instance.feature_extractor.scaler = state["feature_extractor_scaler"]
        instance.feature_extractor.is_fitted = state["feature_extractor_fitted"]

        instance._fitted = state["fitted"]
        instance._fit_timestamp = (
            datetime.fromisoformat(state["fit_timestamp"]) if state["fit_timestamp"] else None
        )
        instance._model_names = state["model_names"]
        instance._global_weights = state["global_weights"]
        instance._feature_config = state.get("feature_config", {})
        instance._has_load_features = state.get("has_load_features", False)

        logger.info("Loaded context-aware combiner from %s", filepath)

        return instance
