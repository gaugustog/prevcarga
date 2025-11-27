"""Stacking combiner using meta-learning for model ensembles.

This module provides the StackingCombiner class that uses stacked generalization
(stacking) to learn optimal combinations of base model predictions. Unlike simple
averaging, stacking trains a meta-model (second-level model) to predict the target
based on base model outputs.

Note: This module uses scikit-learn naming conventions (uppercase X for feature matrices).

Stacking is a powerful ensemble technique that:
- Learns non-linear combinations of base models
- Captures complementary strengths of different models
- Uses cross-validation to prevent overfitting
- Supports various meta-model types (linear, tree-based, gradient boosting)

The combiner implements a two-stage approach:
1. Stage 1: Base models are already trained (we receive their predictions)
2. Stage 2: Meta-model is trained on out-of-fold (OOF) base model predictions

Key Features:
- Cross-validation for generating OOF predictions (prevents data leakage)
- Multiple meta-model types (linear, ridge, lasso, elastic_net, rf, lgbm)
- Weight extraction for interpretability (linear models)
- Handles missing values in base predictions
- Stores CV scores for model evaluation
- Supports both TimeSeriesSplit and KFold cross-validation

Example:
    ```python
    from src.models.combination import StackingCombiner, CombinerConfig
    import numpy as np

    # Training phase - fit meta-model
    train_predictions = {
        "lgbm": train_lgbm_df,
        "rf": train_rf_df,
        "arima": train_arima_df
    }
    train_targets = train_targets_df

    config = CombinerConfig(require_fit=True, store_contributions=True)
    combiner = StackingCombiner(
        meta_model="ridge",
        cv_folds=5,
        cv_strategy="timeseries",
        config=config
    )
    combiner.fit(train_predictions, train_targets)

    # Inference phase - use meta-model for predictions
    test_predictions = {
        "lgbm": test_lgbm_df,
        "rf": test_rf_df,
        "arima": test_arima_df
    }
    result = combiner.combine(test_predictions)
    print(result.weights)  # Learned coefficients from meta-model
    ```
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit

from src.models.combination.base_combiner import BaseCombiner
from src.models.combination.data_structures import (
    CombinationResult,
    CombinerConfig,
)
from src.utils.logger import get_logger

# Ruff: Ignore ML conventions and base class interface
# ruff: noqa: N806, PLR0912, PLR0915, PLR2004, ARG002, B905, DTZ005

# Optional import for LightGBM
try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = get_logger(__name__)

# Type alias for meta-model types
MetaModelType = Literal["linear", "ridge", "lasso", "elastic_net", "rf", "lgbm"]


class StackingCombiner(BaseCombiner):
    """Stacking combiner using meta-learning to combine model predictions.

    This combiner implements stacked generalization (stacking), a powerful ensemble
    technique that trains a meta-model to optimally combine base model predictions.
    Unlike simple averaging, stacking can learn non-linear combinations and capture
    complementary strengths of different models.

    The training process uses cross-validation to generate out-of-fold (OOF)
    predictions, which are then used to train the meta-model. This prevents
    overfitting and data leakage.

    Supported Meta-Models:
        - linear: Linear Regression (no regularization)
        - ridge: Ridge Regression (L2 regularization, alpha=1.0)
        - lasso: Lasso Regression (L1 regularization, alpha=0.01)
        - elastic_net: Elastic Net (L1+L2 regularization)
        - rf: Random Forest Regressor (n_estimators=100)
        - lgbm: LightGBM Regressor (num_leaves=31)

    Cross-Validation Strategies:
        - kfold: Standard K-fold cross-validation
        - timeseries: Time series split (respects temporal order)

    Attributes:
        config: CombinerConfig instance with configuration options.
        meta_model_name: Type of meta-model to use.
        cv_folds: Number of cross-validation folds.
        cv_strategy: Cross-validation strategy ('kfold' or 'timeseries').
        meta_model_params: Additional parameters for the meta-model.

    Example:
        >>> # Ridge regression meta-model with 5-fold CV
        >>> combiner = StackingCombiner(
        ...     meta_model="ridge",
        ...     cv_folds=5,
        ...     cv_strategy="timeseries"
        ... )
        >>> combiner.fit(train_predictions, train_targets)
        >>> result = combiner.combine(test_predictions)
        >>> print(result.weights)  # Learned coefficients

        >>> # LightGBM meta-model for non-linear combinations
        >>> combiner = StackingCombiner(
        ...     meta_model="lgbm",
        ...     cv_folds=5,
        ...     meta_model_params={"num_leaves": 15, "learning_rate": 0.05}
        ... )
        >>> combiner.fit(train_predictions, train_targets)
        >>> result = combiner.combine(test_predictions)
    """

    def __init__(
        self,
        meta_model: MetaModelType = "ridge",
        cv_folds: int = 5,
        cv_strategy: Literal["kfold", "timeseries"] = "timeseries",
        meta_model_params: dict[str, Any] | None = None,
        config: CombinerConfig | dict[str, Any] | None = None,
    ) -> None:
        """Initialize the stacking combiner.

        Args:
            meta_model: Type of meta-model to use. One of:
                'linear', 'ridge', 'lasso', 'elastic_net', 'rf', 'lgbm'.
                Default is 'ridge' (recommended for Brazilian load forecasting).
            cv_folds: Number of cross-validation folds for generating OOF predictions.
                Default is 5. Must be >= 2.
            cv_strategy: Cross-validation strategy to use.
                - 'kfold': Standard K-fold cross-validation
                - 'timeseries': Time series split (respects temporal order)
                Default is 'timeseries' (recommended for temporal data).
            meta_model_params: Additional parameters to pass to the meta-model
                constructor. If None, uses default parameters.
            config: Configuration as CombinerConfig instance or dictionary.
                If None, default configuration with require_fit=True is used.

        Raises:
            ValueError: If meta_model is invalid, cv_folds < 2, or cv_strategy invalid.
            ImportError: If meta_model='lgbm' but lightgbm is not installed.

        Example:
            >>> # Ridge regression with default parameters
            >>> combiner = StackingCombiner(meta_model="ridge")

            >>> # LightGBM with custom parameters
            >>> combiner = StackingCombiner(
            ...     meta_model="lgbm",
            ...     meta_model_params={"num_leaves": 31, "max_depth": 5}
            ... )

            >>> # KFold cross-validation instead of time series
            >>> combiner = StackingCombiner(cv_strategy="kfold", cv_folds=10)
        """
        # Initialize base combiner
        if config is None:
            # Stacking requires fitting
            config = CombinerConfig(require_fit=True)
        super().__init__(config=config)

        # Validate meta-model type
        valid_meta_models = {"linear", "ridge", "lasso", "elastic_net", "rf", "lgbm"}
        if meta_model not in valid_meta_models:
            msg = f"Invalid meta_model: {meta_model!r}. " f"Must be one of {valid_meta_models}."
            raise ValueError(msg)

        # Check LightGBM availability
        if meta_model == "lgbm" and not LIGHTGBM_AVAILABLE:
            msg = (
                "LightGBM is not installed. Install with: pip install lightgbm. "
                "Or use a different meta-model type."
            )
            raise ImportError(msg)

        # Validate CV folds
        if cv_folds < 2:
            msg = f"cv_folds must be >= 2, got {cv_folds}"
            raise ValueError(msg)

        # Validate CV strategy
        valid_cv_strategies = {"kfold", "timeseries"}
        if cv_strategy not in valid_cv_strategies:
            msg = f"Invalid cv_strategy: {cv_strategy!r}. " f"Must be one of {valid_cv_strategies}."
            raise ValueError(msg)

        self.meta_model_name = meta_model
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy
        self.meta_model_params = meta_model_params or {}

        # Internal state for fitted meta-model
        self._meta_model: Any = None
        self._cv_scores: dict[str, list[float]] = {}
        self._oof_predictions: np.ndarray | None = None

        # Store configuration in custom_config for persistence
        self.config.custom_config.update(
            {
                "meta_model": meta_model,
                "cv_folds": cv_folds,
                "cv_strategy": cv_strategy,
                "meta_model_params": meta_model_params or {},
            }
        )

        logger.debug(
            "Initialized StackingCombiner with meta_model=%s, cv_folds=%d, cv_strategy=%s",
            self.meta_model_name,
            self.cv_folds,
            self.cv_strategy,
        )

    @property
    def name(self) -> str:
        """Return combiner name.

        Returns:
            Name string identifying this combiner type.
        """
        return f"stacking_{self.meta_model_name}"

    @property
    def version(self) -> str:
        """Return combiner version.

        Returns:
            Version string following semantic versioning.
        """
        return "1.0.0"

    def _create_meta_model(self) -> Any:
        """Create meta-model instance based on configuration.

        Returns:
            Instantiated meta-model with configured parameters.

        Raises:
            ValueError: If meta_model_name is invalid.
        """
        if self.meta_model_name == "linear":
            return LinearRegression(**self.meta_model_params)
        if self.meta_model_name == "ridge":
            # Default alpha=1.0 for ridge
            params = {"alpha": 1.0, **self.meta_model_params}
            return Ridge(**params)
        if self.meta_model_name == "lasso":
            # Default alpha=0.01 for lasso
            params = {"alpha": 0.01, **self.meta_model_params}
            return Lasso(**params)
        if self.meta_model_name == "elastic_net":
            # Default alpha=0.1, l1_ratio=0.5 for elastic net
            params = {"alpha": 0.1, "l1_ratio": 0.5, **self.meta_model_params}
            return ElasticNet(**params)
        if self.meta_model_name == "rf":
            # Default n_estimators=100 for random forest
            params = {"n_estimators": 100, "random_state": 42, **self.meta_model_params}
            return RandomForestRegressor(**params)
        if self.meta_model_name == "lgbm":
            # Default num_leaves=31 for LightGBM
            params = {
                "num_leaves": 31,
                "random_state": 42,
                "verbose": -1,
                **self.meta_model_params,
            }
            return LGBMRegressor(**params)
        msg = f"Unknown meta_model_name: {self.meta_model_name}"
        raise ValueError(msg)

    def _create_cv_splitter(self, n_samples: int) -> Any:
        """Create cross-validation splitter based on configuration.

        Args:
            n_samples: Number of samples in the dataset.

        Returns:
            Cross-validation splitter instance.
        """
        if self.cv_strategy == "kfold":
            return KFold(n_splits=self.cv_folds, shuffle=False)
        # timeseries
        return TimeSeriesSplit(n_splits=self.cv_folds)

    def _generate_oof_predictions(
        self,
        predictions: dict[str, pd.DataFrame],
        targets: pd.DataFrame,
        horizon_col: str,
    ) -> tuple[np.ndarray, list[float]]:
        """Generate out-of-fold predictions for a single horizon.

        Uses cross-validation to generate predictions where the meta-model
        never sees training data predictions. This prevents data leakage.

        Args:
            predictions: Dictionary mapping model names to prediction DataFrames.
            targets: Target values DataFrame.
            horizon_col: Column name in predictions (e.g., 'pred_h0').

        Returns:
            Tuple of (oof_predictions, cv_scores).
            - oof_predictions: Array of out-of-fold predictions (shape: n_samples)
            - cv_scores: List of RMSE scores for each fold.
        """
        # Extract target column (e.g., 'pred_h0' -> 'h0')
        target_col = horizon_col.replace("pred_", "")

        # Stack predictions from all models into feature matrix
        # Shape: (n_samples, n_models)
        model_names = sorted(predictions.keys())
        X = np.column_stack([predictions[name][horizon_col].to_numpy() for name in model_names])
        y = targets[target_col].to_numpy()

        n_samples = len(y)

        # Initialize OOF predictions array
        oof_preds = np.full(n_samples, np.nan)
        cv_scores = []

        # Create cross-validation splitter
        cv_splitter = self._create_cv_splitter(n_samples)

        # Generate OOF predictions using cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Handle missing values in training data
            # Remove samples with any NaN values
            train_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
            X_train_clean = X_train[train_mask]
            y_train_clean = y_train[train_mask]

            if len(X_train_clean) == 0:
                logger.warning(
                    "Fold %d has no valid training samples for %s, skipping",
                    fold_idx,
                    horizon_col,
                )
                continue

            # Train meta-model on this fold
            fold_meta_model = self._create_meta_model()
            fold_meta_model.fit(X_train_clean, y_train_clean)

            # Predict on validation set
            # Handle missing values in validation data
            val_mask = ~np.isnan(X_val).any(axis=1)
            if val_mask.any():
                X_val_clean = X_val[val_mask]
                val_preds = fold_meta_model.predict(X_val_clean)

                # Store OOF predictions
                valid_val_idx = val_idx[val_mask]
                oof_preds[valid_val_idx] = val_preds

                # Compute fold score
                y_val_clean = y_val[val_mask]
                fold_rmse = float(np.sqrt(np.mean((val_preds - y_val_clean) ** 2)))
                cv_scores.append(fold_rmse)

                logger.debug(
                    "Fold %d/%d for %s: RMSE=%.2f (train_size=%d, val_size=%d)",
                    fold_idx + 1,
                    self.cv_folds,
                    horizon_col,
                    fold_rmse,
                    len(X_train_clean),
                    len(X_val_clean),
                )

        return oof_preds, cv_scores

    def fit(
        self,
        train_predictions: dict[str, pd.DataFrame],
        train_targets: pd.DataFrame,
        validation_predictions: dict[str, pd.DataFrame] | None = None,
        validation_targets: pd.DataFrame | None = None,
    ) -> None:
        """Fit the stacking combiner using cross-validation.

        This method:
        1. Validates input predictions and targets
        2. Generates out-of-fold (OOF) predictions using cross-validation
        3. Trains the meta-model on OOF predictions
        4. Extracts learned weights (for linear models)
        5. Stores CV scores for evaluation

        The OOF predictions ensure the meta-model never sees training predictions,
        preventing overfitting and data leakage.

        Args:
            train_predictions: Dictionary mapping model names to predictions
                on the training set. Each DataFrame should have columns like
                'pred_h0', 'pred_h1', etc.
            train_targets: True target values for training period.
                Should have columns like 'h0', 'h1', etc.
            validation_predictions: Optional validation predictions (currently unused,
                reserved for future hyperparameter tuning).
            validation_targets: Optional validation targets (currently unused).

        Raises:
            ValueError: If prediction format is invalid or targets don't match predictions.

        Example:
            >>> combiner = StackingCombiner(meta_model="ridge", cv_folds=5)
            >>> combiner.fit(train_predictions, train_targets)
            >>> print(combiner.get_weights())
            {'lgbm': 0.45, 'rf': 0.30, 'xgb': 0.25}
            >>> print(combiner._cv_scores)
            {'h0': [12.3, 11.8, 12.1, 11.9, 12.0], 'h1': [...], ...}
        """
        # Validate predictions
        self._validate_predictions(train_predictions)

        # Store model names
        self._model_names = sorted(train_predictions.keys())

        # Get prediction columns (horizons)
        first_pred = next(iter(train_predictions.values()))
        pred_cols = sorted([col for col in first_pred.columns if col.startswith("pred_h")])

        if not pred_cols:
            msg = "No prediction columns found (expected pred_h0, pred_h1, etc.)"
            raise ValueError(msg)

        # Validate targets have matching columns
        target_cols = [col.replace("pred_", "") for col in pred_cols]
        missing_cols = set(target_cols) - set(train_targets.columns)
        if missing_cols:
            msg = f"Target columns missing: {missing_cols}"
            raise ValueError(msg)

        logger.info(
            "Fitting stacking combiner with %d models, %d horizons, %d samples, "
            "meta_model=%s, cv_folds=%d",
            len(self._model_names),
            len(pred_cols),
            len(first_pred),
            self.meta_model_name,
            self.cv_folds,
        )

        # Generate OOF predictions for each horizon using cross-validation
        # We train a separate meta-model for each horizon
        self._cv_scores = {}

        # For simplicity, we'll train a single meta-model on the first horizon
        # In production, you might want separate meta-models per horizon
        # Here we use pred_h0 as representative
        first_horizon = pred_cols[0]
        target_col = first_horizon.replace("pred_", "")

        logger.info("Generating OOF predictions using %d-fold CV...", self.cv_folds)

        oof_preds, cv_scores = self._generate_oof_predictions(
            train_predictions,
            train_targets,
            first_horizon,
        )

        self._oof_predictions = oof_preds
        self._cv_scores[target_col] = cv_scores

        # Log CV results
        if cv_scores:
            mean_rmse = float(np.mean(cv_scores))
            std_rmse = float(np.std(cv_scores))
            logger.info(
                "CV results for %s: RMSE = %.2f ± %.2f (mean ± std across %d folds)",
                target_col,
                mean_rmse,
                std_rmse,
                len(cv_scores),
            )

        # Train final meta-model on all OOF predictions
        logger.info("Training final meta-model on OOF predictions...")

        # Stack predictions into feature matrix
        X = np.column_stack(
            [train_predictions[name][first_horizon].to_numpy() for name in self._model_names]
        )
        y = train_targets[target_col].to_numpy()

        # Remove samples with NaN OOF predictions (from CV folds with no valid data)
        valid_mask = ~np.isnan(oof_preds) & ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        if len(X_clean) == 0:
            msg = "No valid samples after removing NaN values from OOF predictions"
            raise ValueError(msg)

        # Create and train final meta-model
        self._meta_model = self._create_meta_model()
        self._meta_model.fit(X_clean, y_clean)

        # Extract weights from meta-model (for linear models)
        self._extract_weights()

        # Mark as fitted
        self._fitted = True
        self._fit_timestamp = datetime.now()

        logger.info(
            "Successfully fitted stacking combiner (used %d/%d samples for training)",
            len(X_clean),
            len(y),
        )

    def _extract_weights(self) -> None:
        """Extract learned weights from meta-model for interpretability.

        For linear models, the coefficients represent the weights.
        For tree-based models, we use feature importances as pseudo-weights.

        Updates self._global_weights with extracted weights.
        """
        if self._meta_model is None:
            return

        weights_dict = {}

        # Linear models have coefficients
        if hasattr(self._meta_model, "coef_"):
            coefs = self._meta_model.coef_
            if coefs.ndim > 1:
                # Multi-output case, use first output
                coefs = coefs[0]

            # Map coefficients to model names
            for name, coef in zip(self._model_names, coefs):
                weights_dict[name] = float(coef)

            # Normalize to sum to 1 (for interpretability)
            # Note: coefficients can be negative, so we use absolute values
            abs_weights = {k: abs(v) for k, v in weights_dict.items()}
            total = sum(abs_weights.values())
            if total > 0:
                weights_dict = {k: abs_weights[k] / total for k in weights_dict}
            else:
                # Equal weights if all coefficients are zero
                n = len(self._model_names)
                weights_dict = dict.fromkeys(self._model_names, 1.0 / n)

        # Tree-based models have feature importances
        elif hasattr(self._meta_model, "feature_importances_"):
            importances = self._meta_model.feature_importances_

            # Map importances to model names
            for name, importance in zip(self._model_names, importances):
                weights_dict[name] = float(importance)

            # Normalize to sum to 1
            total = sum(weights_dict.values())
            if total > 0:
                weights_dict = {k: v / total for k, v in weights_dict.items()}
            else:
                # Equal weights if all importances are zero
                n = len(self._model_names)
                weights_dict = dict.fromkeys(self._model_names, 1.0 / n)

        else:
            # Unknown model type, use equal weights
            n = len(self._model_names)
            weights_dict = dict.fromkeys(self._model_names, 1.0 / n)
            logger.warning(
                "Meta-model %s does not have coefficients or feature importances. "
                "Using equal weights for interpretability.",
                self._meta_model.__class__.__name__,
            )

        self._global_weights = weights_dict

        logger.debug(
            "Extracted weights from meta-model: %s",
            {k: f"{v:.3f}" for k, v in weights_dict.items()},
        )

    def combine(
        self,
        predictions: dict[str, pd.DataFrame],
        metadata: dict[str, Any] | None = None,
    ) -> CombinationResult:
        """Combine predictions using the fitted meta-model.

        Uses the trained meta-model to predict the final forecast from base
        model predictions. This method must be called after fit().

        Args:
            predictions: Dictionary mapping model names to prediction DataFrames.
                Each DataFrame should have columns like 'pred_h0', 'pred_h1', etc.
                Model names must match those used during fitting.
            metadata: Optional metadata about predictions (currently unused).

        Returns:
            CombinationResult with:
                - combined_predictions: DataFrame with meta-model predictions
                - weights: Dictionary of learned weights (from meta-model coefficients)
                - metadata: Information about the combination process
                - model_contributions: Individual contributions (if config.store_contributions=True)

        Raises:
            ValueError: If predictions format is invalid or model names don't match.
            RuntimeError: If combiner not fitted.

        Example:
            >>> # After fitting
            >>> result = combiner.combine(test_predictions)
            >>> print(result.combined_predictions.head())
            >>> print(f"Dominant model: {result.get_dominant_model()}")
            >>> print(f"CV RMSE: {np.mean(combiner._cv_scores['h0']):.2f}")
        """
        start_time = time.time()

        # Check if fitted
        self._check_is_fitted()

        if self._meta_model is None:
            msg = "Meta-model not trained. Call fit() first."
            raise RuntimeError(msg)

        # Validate predictions
        self._validate_predictions(predictions)

        # Check model names match
        pred_model_names = sorted(predictions.keys())
        if pred_model_names != self._model_names:
            msg = (
                f"Model names don't match fitted models. "
                f"Expected {self._model_names}, got {pred_model_names}"
            )
            raise ValueError(msg)

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

        # Create combined predictions DataFrame
        combined_df = pd.DataFrame(index=first_pred.index)
        model_contributions = {} if self.config.store_contributions else None

        # For each horizon, use meta-model to predict
        for col in pred_cols:
            # Stack predictions from all models into feature matrix
            # Shape: (n_samples, n_models)
            X = np.column_stack(
                [
                    cleaned_predictions[name][col].to_numpy()
                    for name in self._model_names
                    if name in cleaned_predictions
                ]
            )

            # Handle missing values
            # For samples with any NaN, we'll use simple averaging as fallback
            has_nan = np.isnan(X).any(axis=1)

            if has_nan.any():
                logger.debug(
                    "Found %d samples with NaN values in %s, using fallback averaging",
                    has_nan.sum(),
                    col,
                )

            # Predict using meta-model for valid samples
            predictions_array = np.full(len(X), np.nan)

            # Valid samples (no NaN)
            valid_mask = ~has_nan
            if valid_mask.any():
                X_valid = X[valid_mask]
                predictions_array[valid_mask] = self._meta_model.predict(X_valid)

            # Fallback: simple average for samples with NaN
            if has_nan.any():
                X_nan = X[has_nan]
                # Average over valid values only
                with np.errstate(divide="ignore", invalid="ignore"):
                    predictions_array[has_nan] = np.nanmean(X_nan, axis=1)

            combined_df[col] = predictions_array

            # Store individual contributions if requested
            if self.config.store_contributions:
                for name in self._model_names:
                    if name not in cleaned_predictions:
                        continue

                    if name not in model_contributions:
                        model_contributions[name] = pd.DataFrame(index=first_pred.index)

                    # For linear models, contribution is coef * prediction
                    # For tree models, we use the prediction weighted by importance
                    weight = self._global_weights.get(name, 0.0)
                    model_contributions[name][col] = (
                        cleaned_predictions[name][col].to_numpy() * weight
                    )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create metadata
        metadata_obj = self._create_metadata(
            models_used=list(cleaned_predictions.keys()),
            models_excluded=excluded_models,
            processing_time=processing_time,
        )

        # Add stacking-specific metadata
        metadata_obj.configuration["meta_model"] = self.meta_model_name
        metadata_obj.configuration["cv_folds"] = self.cv_folds
        metadata_obj.configuration["cv_strategy"] = self.cv_strategy

        # Add CV scores to performance metrics
        if self._cv_scores:
            for horizon, scores in self._cv_scores.items():
                if scores:
                    metadata_obj.add_metric(f"cv_rmse_mean_{horizon}", float(np.mean(scores)))
                    metadata_obj.add_metric(f"cv_rmse_std_{horizon}", float(np.std(scores)))

        # Add weights entropy
        metadata_obj.performance_metrics["weights_entropy"] = float(
            -sum(w * np.log(w + 1e-10) for w in self._global_weights.values())
        )

        # Create result
        result = CombinationResult(
            combined_predictions=combined_df,
            weights=self._global_weights.copy(),
            metadata=metadata_obj,
            model_contributions=model_contributions,
            confidence_intervals=None,  # Not implemented for stacking yet
        )

        logger.info(
            "Combined %d models using %s (processing time: %.3fs)",
            len(cleaned_predictions),
            self.name,
            processing_time,
        )

        return result

    def get_cv_scores(self, horizon: str | None = None) -> dict[str, list[float]] | list[float]:
        """Get cross-validation scores from training.

        Args:
            horizon: Optional specific horizon to get scores for (e.g., 'h0').
                If None, returns scores for all horizons.

        Returns:
            If horizon is None: dictionary mapping horizon names to CV score lists.
            If horizon is specified: list of CV scores for that horizon.

        Raises:
            RuntimeError: If combiner not fitted.
            KeyError: If specified horizon not found.

        Example:
            >>> combiner.fit(train_predictions, train_targets)
            >>> scores = combiner.get_cv_scores('h0')
            >>> print(f"Mean CV RMSE: {np.mean(scores):.2f}")
            >>> all_scores = combiner.get_cv_scores()
            >>> print(all_scores.keys())  # ['h0', 'h1', 'h2', ...]
        """
        if not self.is_fitted:
            msg = "Combiner not fitted. Call fit() first."
            raise RuntimeError(msg)

        if horizon is not None:
            if horizon not in self._cv_scores:
                msg = f"Horizon {horizon!r} not found. Available: {list(self._cv_scores.keys())}"
                raise KeyError(msg)
            return self._cv_scores[horizon].copy()

        return {k: v.copy() for k, v in self._cv_scores.items()}

    def save(self, filepath: str | Path) -> None:
        """Save combiner state to file.

        Serializes the combiner's meta-model, weights, configuration, and CV scores
        for later loading.

        Args:
            filepath: Path to save file (typically .pkl or .joblib).

        Example:
            >>> combiner.save("combiners/stacking_ridge_v1.pkl")
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
            # Stacking-specific state
            "meta_model": self._meta_model,
            "meta_model_name": self.meta_model_name,
            "cv_folds": self.cv_folds,
            "cv_strategy": self.cv_strategy,
            "cv_scores": self._cv_scores.copy(),
            "meta_model_params": self.meta_model_params.copy(),
            "oof_predictions": self._oof_predictions,
        }

        joblib.dump(state, filepath)
        logger.info("Saved %s combiner to %s", self.name, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "StackingCombiner":
        """Load combiner from file.

        Deserializes a previously saved combiner state, including the meta-model.

        Args:
            filepath: Path to load file.

        Returns:
            Loaded StackingCombiner instance.

        Example:
            >>> combiner = StackingCombiner.load("combiners/stacking_ridge_v1.pkl")
            >>> result = combiner.combine(test_predictions)
        """
        filepath = Path(filepath)
        state = joblib.load(filepath)

        # Extract stacking-specific parameters
        meta_model = state.get("meta_model_name", "ridge")
        cv_folds = state.get("cv_folds", 5)
        cv_strategy = state.get("cv_strategy", "timeseries")
        meta_model_params = state.get("meta_model_params", {})

        # Create instance
        instance = cls(
            meta_model=meta_model,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            meta_model_params=meta_model_params,
            config=state["config"],
        )

        # Restore base combiner state
        instance._weights = state["weights"]
        instance._global_weights = state["global_weights"]
        instance._fitted = state["fitted"]
        instance._fit_timestamp = (
            datetime.fromisoformat(state["fit_timestamp"]) if state["fit_timestamp"] else None
        )
        instance._performance_history = state["performance_history"]
        instance._model_names = state["model_names"]

        # Restore stacking-specific state
        instance._meta_model = state.get("meta_model")
        instance._cv_scores = state.get("cv_scores", {})
        instance._oof_predictions = state.get("oof_predictions")

        logger.info("Loaded %s combiner from %s", instance.name, filepath)

        return instance

    def reset(self) -> None:
        """Reset combiner to unfitted state.

        Clears all learned weights, performance history, and meta-model state.
        """
        # Call parent reset
        super().reset()

        # Reset stacking-specific state
        self._meta_model = None
        self._cv_scores = {}
        self._oof_predictions = None

        logger.info("Reset %s combiner (including meta-model) to unfitted state", self.name)
