"""LightGBM model implementation for load forecasting.

This module provides the LGBMModel class for short-term electric load forecasting
using LightGBM gradient boosting. It supports:
- D+0 and D+1 forecast horizons
- Optuna hyperparameter optimization
- Feature importance and SHAP value analysis
- Custom serialization for LightGBM models

Example:
    ```python
    from src.models.end_to_end.lgbm_model import LGBMModel
    from src.models.config import LGBMConfig
    import pandas as pd

    # Create and train model
    model = LGBMModel()
    config = LGBMConfig(optimize_hyperparams=True, n_trials=50)
    model.fit(X_train, y_train, config.model_dump())

    # Make predictions
    predictions = model.predict(X_test, horizons=[0, 1])

    # Get feature importance
    importance = model.get_feature_importance()

    # Save/load
    model.save("models/lgbm_model.joblib")
    loaded_model = LGBMModel.load("models/lgbm_model.joblib")
    ```
"""

import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.models.base import BaseModel, register_model_class
from src.models.config.lgbm_config import LGBMConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@register_model_class("lgbm")
class LGBMModel(BaseModel):
    """LightGBM gradient boosting model for electric load forecasting.

    This model uses LightGBM for short-term load forecasting with support for:
    - D+0 (same-day) and D+1 (next-day) predictions
    - Automatic hyperparameter optimization via Optuna
    - Feature importance extraction (gain-based)
    - SHAP value computation for model interpretability
    - Horizon-specific adjustments for multi-step forecasting

    The model uses the native LightGBM API (lightgbm.train) rather than the
    scikit-learn wrapper for better control and performance.

    Attributes:
        _model: Trained LightGBM Booster object.
        _config: LGBMConfig instance with training configuration.
        _best_iteration: Best boosting round (with early stopping).
        _cv_scores: Cross-validation scores from hyperparameter optimization.
    """

    def __init__(self) -> None:
        """Initialize LGBMModel with empty state."""
        super().__init__()
        self._model: lgb.Booster | None = None
        self._config: LGBMConfig | None = None
        self._best_iteration: int = 0
        self._cv_scores: dict[str, float] = {}
        logger.debug("Initialized LGBMModel")

    @property
    def name(self) -> str:
        """Return model name identifier.

        Returns:
            Model name "lgbm".
        """
        return "lgbm"

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
            List containing [0, 1] for D+0 and D+1 forecasts.
        """
        return [0, 1]

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series | pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        """Train the LightGBM model on provided data.

        This method:
        1. Validates input data
        2. Parses configuration into LGBMConfig
        3. Optionally performs Optuna hyperparameter optimization
        4. Trains the model with early stopping
        5. Stores training metadata

        Args:
            X: Training features DataFrame with shape (n_samples, n_features).
            y: Training target as Series (single column) or DataFrame.
               If DataFrame, uses first column.
            config: Model configuration dictionary. Should match LGBMConfig schema.

        Raises:
            ValueError: If input data is invalid or incompatible.
            RuntimeError: If training fails.

        Example:
            ```python
            from src.models.config import LGBMConfig

            config = LGBMConfig(n_estimators=500, optimize_hyperparams=False)
            model.fit(X_train, y_train, config.model_dump())
            ```
        """
        logger.info(
            "Starting LGBMModel training with %d samples, %d features", len(X), len(X.columns)
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
            self._config = LGBMConfig(**config)
        except Exception as e:
            msg = f"Invalid configuration: {e}"
            raise ValueError(msg) from e

        # Store feature names
        self._feature_names = list(X.columns)

        # Prepare training data
        train_data = self._prepare_data(X, y)

        # Optimize hyperparameters if requested
        if self._config.optimize_hyperparams:
            logger.info(
                "Starting Optuna hyperparameter optimization with %d trials", self._config.n_trials
            )
            best_params = self._optimize_hyperparams(train_data)
            # Update config with optimized parameters by creating new config
            # (avoids validation issues when updating dependent fields one at a time)
            config_dict = self._config.model_dump()
            for key, value in best_params.items():
                if key in config_dict:
                    config_dict[key] = value
            self._config = LGBMConfig(**config_dict)
            logger.info("Hyperparameter optimization completed. Best params: %s", best_params)

        # Get final training parameters
        params = self._get_default_params()

        # Train model (without early stopping on single dataset)
        # Note: Early stopping requires validation set, which we don't have in basic training
        # Optuna optimization uses CV with early stopping internally
        try:
            logger.info("Training LightGBM model with %d estimators", self._config.n_estimators)
            self._model = lgb.train(
                params,
                train_data,
                num_boost_round=self._config.n_estimators,
            )
            self._best_iteration = self._config.n_estimators
            logger.info("Training completed at iteration: %d", self._best_iteration)
        except Exception as e:
            msg = f"Training failed: {e}"
            raise RuntimeError(msg) from e

        # Mark as fitted
        self._is_fitted = True

        # Store training metadata
        self._training_metadata = {
            "trained_at": datetime.now(UTC),
            "training_samples": len(X),
            "training_features": len(X.columns),
            "best_iteration": self._best_iteration,
            "cv_scores": self._cv_scores,
            "training_config": self._config.model_dump(),
        }

        logger.info("LGBMModel training completed successfully")

    def predict(
        self,
        X: pd.DataFrame,  # noqa: N803
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Generate predictions for specified horizons.

        This method validates the model is fitted, validates features match training,
        and generates predictions with optional horizon-specific adjustments.

        Args:
            X: Features DataFrame with shape (n_samples, n_features).
            horizons: List of horizons to predict [0, 1]. If None, predicts all
                     supported horizons. Must be subset of [0, 1].

        Returns:
            DataFrame with predictions, shape (n_samples, n_horizons).
            Columns named "h0", "h1" for each horizon.

        Raises:
            ValueError: If model not fitted or horizons are invalid.
            RuntimeError: If prediction fails.

        Example:
            ```python
            # Predict all supported horizons
            preds = model.predict(X_test)

            # Predict specific horizons
            preds = model.predict(X_test, horizons=[0])  # D+0 only
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

        # Validate features
        self.validate_features(X)

        # Select only training features in correct order
        X_pred = X[self._feature_names]  # noqa: N806

        logger.info("Generating predictions for %d samples, horizons %s", len(X), horizons)

        # Generate base predictions
        try:
            base_predictions = self._model.predict(X_pred, num_iteration=self._best_iteration)
        except Exception as e:
            msg = f"Prediction failed: {e}"
            raise RuntimeError(msg) from e

        # Create predictions dataframe with horizon-specific adjustments
        predictions = pd.DataFrame(index=X.index)

        for horizon in sorted(horizons):
            # Apply horizon adjustment
            adjusted_preds = self._calculate_horizon_adjustment(X_pred, base_predictions, horizon)
            predictions[f"h{horizon}"] = adjusted_preds

        logger.info("Predictions generated successfully for horizons %s", horizons)
        return predictions

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores based on gain.

        Returns:
            Dictionary mapping feature names to importance scores.
            Scores are normalized to sum to 1.0.

        Raises:
            ValueError: If model is not fitted.

        Example:
            ```python
            importance = model.get_feature_importance()
            # Returns: {"feature1": 0.35, "feature2": 0.25, ...}
            ```
        """
        self._check_is_fitted()

        # Get gain-based importance
        importance = self._model.feature_importance(importance_type="gain")

        # Create dictionary with feature names
        importance_dict = dict(zip(self._feature_names, importance, strict=False))

        # Normalize to sum to 1.0
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v / total for k, v in importance_dict.items()}

        logger.debug("Extracted feature importance for %d features", len(importance_dict))
        return importance_dict

    def get_shap_values(
        self,
        X: pd.DataFrame,  # noqa: N803
        sample_size: int | None = None,
    ) -> np.ndarray:
        """Calculate SHAP values for model interpretability.

        SHAP (SHapley Additive exPlanations) values explain individual predictions
        by computing the contribution of each feature.

        Args:
            X: Features DataFrame to compute SHAP values for.
            sample_size: Number of samples to use. If None, uses all samples.
                        Sampling recommended for large datasets (>10k samples).

        Returns:
            SHAP values array with shape (n_samples, n_features).

        Raises:
            ValueError: If model is not fitted.
            ImportError: If shap package is not installed.

        Example:
            ```python
            shap_values = model.get_shap_values(X_test, sample_size=1000)
            # Use with shap library for visualization
            ```
        """
        self._check_is_fitted()

        try:
            import shap  # noqa: PLC0415
        except ImportError as e:
            msg = "SHAP package required. Install with: pip install shap"
            raise ImportError(msg) from e

        # Validate features
        self.validate_features(X)
        X_shap = X[self._feature_names]  # noqa: N806

        # Sample if requested
        if sample_size is not None and len(X_shap) > sample_size:
            logger.info("Sampling %d of %d rows for SHAP computation", sample_size, len(X_shap))
            X_shap = X_shap.sample(n=sample_size, random_state=self._config.random_state)  # noqa: N806

        logger.info("Computing SHAP values for %d samples", len(X_shap))

        # Create SHAP explainer
        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(X_shap)

        logger.info("SHAP values computed successfully")
        return shap_values

    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> lgb.Dataset:  # noqa: N803
        """Prepare LightGBM Dataset from features and target.

        Args:
            X: Features DataFrame.
            y: Target Series.

        Returns:
            LightGBM Dataset object.
        """
        # Identify categorical features by dtype
        categorical_feature_indices = []
        if self._config.categorical_features:
            for feat in self._config.categorical_features:
                if feat in X.columns:
                    categorical_feature_indices.append(list(X.columns).index(feat))

        dataset = lgb.Dataset(
            data=X,
            label=y,
            categorical_feature=(
                categorical_feature_indices if categorical_feature_indices else "auto"
            ),
            free_raw_data=False,
        )

        logger.debug(
            "Prepared LightGBM Dataset with %d categorical features",
            len(categorical_feature_indices),
        )
        return dataset

    def _get_default_params(self) -> dict[str, Any]:
        """Get default LightGBM parameters from config.

        Returns:
            Dictionary of LightGBM parameters.
        """
        params = self._config.to_lgbm_params()
        # Remove n_estimators as it's passed separately to lgb.train
        params.pop("n_estimators", None)
        return params

    def _optimize_hyperparams(self, train_data: lgb.Dataset) -> dict[str, Any]:
        """Optimize hyperparameters using Optuna.

        Args:
            train_data: Training dataset for optimization.

        Returns:
            Dictionary of best hyperparameters found.
        """
        search_space = self._config.get_optuna_search_space()

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function for hyperparameter tuning.

            Args:
                trial: Optuna trial object.

            Returns:
                Mean cross-validation score (lower is better for MAE).
            """
            # Suggest hyperparameters
            params = self._get_default_params()

            # Update with trial suggestions
            params["learning_rate"] = trial.suggest_float(
                "learning_rate",
                search_space["learning_rate"][0],
                search_space["learning_rate"][1],
                log=True,
            )
            # Sample max_depth first, then constrain num_leaves to 2^max_depth
            params["max_depth"] = trial.suggest_int(
                "max_depth",
                search_space["max_depth"][0],
                search_space["max_depth"][1],
            )
            max_leaves = 2 ** params["max_depth"]
            params["num_leaves"] = trial.suggest_int(
                "num_leaves",
                min(search_space["num_leaves"][0], max_leaves),
                min(search_space["num_leaves"][1], max_leaves),
            )
            params["min_child_samples"] = trial.suggest_int(
                "min_child_samples",
                search_space["min_child_samples"][0],
                search_space["min_child_samples"][1],
            )
            params["subsample"] = trial.suggest_float(
                "bagging_fraction",
                search_space["bagging_fraction"][0],
                search_space["bagging_fraction"][1],
            )
            params["colsample_bytree"] = trial.suggest_float(
                "feature_fraction",
                search_space["feature_fraction"][0],
                search_space["feature_fraction"][1],
            )
            params["reg_alpha"] = trial.suggest_float(
                "lambda_l1",
                search_space["lambda_l1"][0],
                search_space["lambda_l1"][1],
            )
            params["reg_lambda"] = trial.suggest_float(
                "lambda_l2",
                search_space["lambda_l2"][0],
                search_space["lambda_l2"][1],
            )

            # Perform time series cross-validation
            cv_results = lgb.cv(
                params,
                train_data,
                num_boost_round=self._config.n_estimators,
                folds=TimeSeriesSplit(n_splits=self._config.cv_folds),
                callbacks=[
                    lgb.early_stopping(self._config.early_stopping_rounds or 50, verbose=False)
                ],
                return_cvbooster=False,
            )

            # Return mean of best validation score
            # LightGBM cv returns keys like "valid l2-mean" or "valid l1-mean"
            # Map metric names to LightGBM internal names
            metric_mapping = {
                "mae": "l1",
                "mse": "l2",
                "rmse": "rmse",
                "mape": "mape",
            }
            lgb_metric = metric_mapping.get(self._config.metric, self._config.metric)
            metric_key = f"valid {lgb_metric}-mean"

            if metric_key not in cv_results:
                # Fallback: try to find any valid metric key
                valid_keys = [k for k in cv_results if "valid" in k and "mean" in k]
                if valid_keys:
                    metric_key = valid_keys[0]
                    logger.warning(
                        "Metric key %s not found, using %s instead",
                        f"valid {lgb_metric}-mean",
                        metric_key,
                    )
                else:
                    msg = f"No valid metric found in CV results. Available keys: {list(cv_results.keys())}"
                    raise RuntimeError(msg)

            return min(cv_results[metric_key])

        # Create Optuna study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self._config.random_state),
        )

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        study.optimize(objective, n_trials=self._config.n_trials, show_progress_bar=False)

        # Store CV scores
        self._cv_scores = {
            "best_value": study.best_value,
            "n_trials": len(study.trials),
        }

        logger.info("Optuna optimization completed. Best score: %.4f", study.best_value)
        return study.best_params

    def _calculate_horizon_adjustment(
        self,
        X: pd.DataFrame,  # noqa: N803, ARG002
        base_predictions: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """Calculate horizon-specific adjustments to predictions.

        For D+1 predictions, applies a slight adjustment factor to account for
        increased uncertainty. D+0 predictions use base predictions unchanged.

        Args:
            X: Features DataFrame (unused currently, reserved for future feature-based adjustments).
            base_predictions: Base model predictions.
            horizon: Forecast horizon (0 or 1).

        Returns:
            Adjusted predictions array.

        Note:
            The X parameter is currently unused but reserved for future enhancements
            where horizon adjustments may depend on input features (e.g., time of day,
            season, or other contextual information).
        """
        if horizon == 0:
            # D+0: no adjustment
            return base_predictions

        if horizon == 1:
            # D+1: slight adjustment for next-day uncertainty
            # This is a placeholder - in production, could be learned from validation data
            adjustment_factor = 1.0  # No adjustment for now
            return base_predictions * adjustment_factor

        # Should not reach here due to horizon validation
        return base_predictions

    def save(self, path: str | Path) -> None:
        """Save model to disk with custom LightGBM serialization.

        This method saves:
        1. LightGBM Booster model (native format)
        2. Model metadata (feature names, config, training info) via pickle

        Args:
            path: File path to save the model. Will create two files:
                 - {path}.lgb for the LightGBM model
                 - {path}.meta for metadata

        Raises:
            ValueError: If model is not fitted.
            OSError: If files cannot be written.
        """
        self._check_is_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save LightGBM model in native format
        lgb_path = path.with_suffix(".lgb")
        try:
            self._model.save_model(str(lgb_path))
            logger.debug("Saved LightGBM model to %s", lgb_path)
        except Exception as e:
            msg = f"Failed to save LightGBM model to {lgb_path}: {e}"
            raise OSError(msg) from e

        # Save metadata (everything except the model)
        meta_path = path.with_suffix(".meta")
        metadata = {
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
            "training_metadata": self._training_metadata,
            "config": self._config.model_dump() if self._config else {},
            "best_iteration": self._best_iteration,
            "cv_scores": self._cv_scores,
        }

        try:
            with meta_path.open("wb") as f:
                pickle.dump(metadata, f)
            logger.debug("Saved model metadata to %s", meta_path)
        except Exception as e:
            msg = f"Failed to save metadata to {meta_path}: {e}"
            raise OSError(msg) from e

        logger.info("Saved LGBMModel to %s (model: %s, metadata: %s)", path, lgb_path, meta_path)

    @classmethod
    def load(cls, path: str | Path) -> "LGBMModel":
        """Load model from disk with custom LightGBM deserialization.

        Args:
            path: File path to load the model from. Expects two files:
                 - {path}.lgb for the LightGBM model
                 - {path}.meta for metadata

        Returns:
            Loaded LGBMModel instance.

        Raises:
            FileNotFoundError: If model files don't exist.
            OSError: If files cannot be read.
        """
        path = Path(path)

        lgb_path = path.with_suffix(".lgb")
        meta_path = path.with_suffix(".meta")

        if not lgb_path.exists():
            msg = f"LightGBM model file not found: {lgb_path}"
            raise FileNotFoundError(msg)

        if not meta_path.exists():
            msg = f"Metadata file not found: {meta_path}"
            raise FileNotFoundError(msg)

        # Load LightGBM model
        try:
            lgb_model = lgb.Booster(model_file=str(lgb_path))
            logger.debug("Loaded LightGBM model from %s", lgb_path)
        except Exception as e:
            msg = f"Failed to load LightGBM model from {lgb_path}: {e}"
            raise OSError(msg) from e

        # Load metadata
        try:
            with meta_path.open("rb") as f:
                metadata = pickle.load(f)  # noqa: S301
            logger.debug("Loaded model metadata from %s", meta_path)
        except Exception as e:
            msg = f"Failed to load metadata from {meta_path}: {e}"
            raise OSError(msg) from e

        # Create model instance and restore state
        model = cls()
        model._model = lgb_model
        model._feature_names = metadata["feature_names"]
        model._is_fitted = metadata["is_fitted"]
        model._training_metadata = metadata["training_metadata"]
        model._best_iteration = metadata["best_iteration"]
        model._cv_scores = metadata["cv_scores"]

        # Restore config
        if metadata["config"]:
            model._config = LGBMConfig(**metadata["config"])

        logger.info("Loaded LGBMModel from %s", path)
        return model
