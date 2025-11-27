"""SVM Profile model for semi-hourly load profile forecasting.

This module implements the SVMProfileModel class for forecasting semi-hourly
load profiles using 48 independent Support Vector Regression (SVR) models.
Each model predicts the profile ratio (load / daily_demand_mean) for a
specific semi-hourly period (0-47).

Example:
    ```python
    from src.models.profile.svm_profile import SVMProfileModel
    from src.models.config.svm_config import SVMProfileConfig
    import pandas as pd

    # Create model
    model = SVMProfileModel()

    # Prepare data with semi-hourly timestamps
    # X should have DatetimeIndex at semi-hourly frequency
    # y should be profile ratio values (load / daily_demand_mean)
    config = SVMProfileConfig(
        kernel="rbf",
        optimize_hyperparams=True,
        cv_folds=5
    )

    # Train 48 models
    model.fit(X_train, y_train, config={"svm_config": config})

    # Generate profile predictions for current period (D+0)
    predictions = model.predict(X_test, horizons=[0])
    ```
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.models.base.model import BaseModel
from src.models.config.svm_config import SVMProfileConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SVMProfileModel(BaseModel):
    """SVM model for semi-hourly profile ratio forecasting.

    This model trains 48 independent SVR models (one per semi-hourly period 0-47)
    to predict profile ratios. Each model:
    - Uses RBF kernel by default for non-linear patterns
    - Optionally optimizes hyperparameters via GridSearchCV
    - Applies feature scaling for better SVM performance
    - Bounds predictions using training statistics

    Profile ratios represent the ratio of semi-hourly load to daily demand mean.
    They typically range from 0.5 to 2.0, with values:
    - < 1: Below daily average (nighttime, low demand)
    - = 1: At daily average
    - > 1: Above daily average (peak hours)

    Attributes:
        _svm_config: SVMProfileConfig instance with model hyperparameters.
        _models: Dictionary mapping period (0-47) to fitted SVR model.
        _scalers: Dictionary mapping period to StandardScaler for features.
        _profile_statistics: Statistics per period for bounding predictions.
        _period_feature_names: Feature names used for training.
    """

    def __init__(self) -> None:
        """Initialize SVM profile model."""
        super().__init__()
        self._svm_config: SVMProfileConfig | None = None
        self._models: dict[int, SVR] = {}
        self._scalers: dict[int, StandardScaler] = {}
        self._profile_statistics: dict[int, dict[str, float]] = {}
        self._period_feature_names: list[str] = []
        logger.debug("Initialized SVMProfileModel")

    @property
    def name(self) -> str:
        """Return the model name identifier.

        Returns:
            Model name string.
        """
        return "svm_profile"

    @property
    def version(self) -> str:
        """Return the model version string.

        Returns:
            Version string in semantic versioning format.
        """
        return "1.0.0"

    @property
    def supported_horizons(self) -> list[int]:
        """Return list of forecast horizons this model supports.

        SVM profile models predict the current period only (D+0), as they
        forecast the profile ratio for the same day based on temporal features.

        Returns:
            List containing only [0] for D+0 horizon.
        """
        return [0]  # Profile models predict current period only

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        """Train 48 SVM models for semi-hourly profile forecasting.

        This method:
        1. Validates and processes input data
        2. Extracts semi-hourly period (0-47) from timestamps
        3. Computes profile statistics per period
        4. Trains one SVR model per period with optional hyperparameter tuning
        5. Stores fitted models, scalers, and metadata

        Args:
            X: Training features DataFrame with DatetimeIndex at semi-hourly frequency.
                Can include temporal features or other covariates.
            y: Training target (profile ratios) as Series or single-column DataFrame.
                Should be profile ratio values (load / daily_demand_mean).
            config: Training configuration dictionary. Should contain:
                - "svm_config": SVMProfileConfig instance or dict

        Raises:
            ValueError: If input data is invalid or insufficient.
            RuntimeError: If SVM training fails.

        Example:
            ```python
            config = {
                "svm_config": SVMProfileConfig(
                    kernel="rbf",
                    optimize_hyperparams=True,
                    min_samples_per_period=50
                )
            }
            model.fit(X_train, y_train, config)
            ```
        """
        logger.info("Starting SVM profile model training")

        # Parse config
        svm_config_input = config.get("svm_config", {})
        if isinstance(svm_config_input, dict):
            self._svm_config = SVMProfileConfig(**svm_config_input)
        elif isinstance(svm_config_input, SVMProfileConfig):
            self._svm_config = svm_config_input
        else:
            msg = f"Invalid svm_config type: {type(svm_config_input)}"
            raise ValueError(msg)

        # Validate data
        if not isinstance(X.index, pd.DatetimeIndex):
            msg = "X must have DatetimeIndex"
            raise ValueError(msg)

        # Convert y to Series if DataFrame
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                msg = f"y must be single column, got {y.shape[1]} columns"
                raise ValueError(msg)
            y_series = y.iloc[:, 0]
        else:
            y_series = y

        if len(X) != len(y_series):
            msg = f"X and y length mismatch: {len(X)} vs {len(y_series)}"
            raise ValueError(msg)

        logger.info("Training on %d semi-hourly observations", len(X))

        # Prepare data grouped by period
        period_data = self._prepare_profile_data(X, y_series)

        # Store feature names and whether we auto-generated them
        if len(X.columns) > 0:
            self._period_feature_names = list(X.columns)
            self._uses_auto_temporal_features = False
        else:
            # If no features provided, extract temporal features
            logger.info("No features provided, extracting temporal features")
            temporal_features = self._prepare_period_features(X.index)
            self._period_feature_names = list(temporal_features.columns)
            self._uses_auto_temporal_features = True

        # Train models for each period
        n_periods_trained = 0
        n_periods_skipped = 0

        for period in range(48):
            if period not in period_data:
                logger.warning("Period %d has no data, skipping", period)
                n_periods_skipped += 1
                continue

            data = period_data[period]
            n_samples = len(data["X"])

            if n_samples < self._svm_config.min_samples_per_period:
                logger.warning(
                    "Period %d has only %d samples (min required: %d), skipping",
                    period,
                    n_samples,
                    self._svm_config.min_samples_per_period,
                )
                n_periods_skipped += 1
                continue

            # Train model for this period
            try:
                self._train_period_model(period, data)
                n_periods_trained += 1

                if (period + 1) % 12 == 0:
                    logger.info(
                        "Progress: trained %d/%d periods",
                        n_periods_trained,
                        48,
                    )

            except Exception as e:
                logger.error(
                    "Failed to train model for period %d: %s",
                    period,
                    e,
                    exc_info=True,
                )
                n_periods_skipped += 1

        if n_periods_trained == 0:
            msg = "No models were successfully trained"
            raise RuntimeError(msg)

        logger.info(
            "Training completed: %d models trained, %d periods skipped",
            n_periods_trained,
            n_periods_skipped,
        )

        # Store metadata
        self._feature_names = self._period_feature_names
        self._training_metadata = {
            "trained_at": datetime.now(UTC),
            "training_config": config,
            "n_samples_total": len(X),
            "n_periods_trained": n_periods_trained,
            "n_periods_skipped": n_periods_skipped,
            "svm_config": self._svm_config.model_dump(),
        }
        self._is_fitted = True

    def predict(
        self,
        X: pd.DataFrame,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Generate profile ratio predictions.

        Args:
            X: Features DataFrame with DatetimeIndex at semi-hourly frequency.
            horizons: List of horizons to predict. Must be [0] or None.
                Profile models only support D+0 (current period).

        Returns:
            DataFrame with predictions. Shape (n_samples, 1).
            Column is named "h0" for horizon 0.

        Raises:
            ValueError: If model is not fitted or horizons are invalid.
            RuntimeError: If prediction fails.

        Example:
            ```python
            # Predict current period profile ratios
            predictions = model.predict(X_test, horizons=[0])
            # Result shape: (n_samples, 1)
            # Column: "h0"
            ```
        """
        self._check_is_fitted()

        if horizons is None:
            horizons = self.supported_horizons
        else:
            # Validate horizons
            invalid = [h for h in horizons if h not in self.supported_horizons]
            if invalid:
                msg = f"Invalid horizons: {invalid}. Supported: {self.supported_horizons}"
                raise ValueError(msg)

        if not isinstance(X.index, pd.DatetimeIndex):
            msg = "X must have DatetimeIndex"
            raise ValueError(msg)

        logger.debug("Generating predictions for %d samples", len(X))

        try:
            # Extract periods from timestamps
            periods = self._extract_periods(X.index)

            # Prepare features - use the same approach as during training
            uses_auto_temporal = getattr(self, "_uses_auto_temporal_features", False)

            if uses_auto_temporal:
                # Training used auto-generated temporal features, generate them again
                X_features = self._prepare_period_features(X.index)
            elif len(X.columns) == 0:
                # No features provided but training used provided features
                # We need the same features that were used during training
                msg = (
                    f"Model was trained with features {self._period_feature_names} "
                    "but no features provided for prediction."
                )
                raise ValueError(msg)
            else:
                # Use provided features
                X_features = X

            # Generate predictions for each sample
            predictions = np.zeros(len(X))

            for i, period in enumerate(periods):
                if period not in self._models:
                    # Use mean ratio from available periods if this period wasn't trained
                    if self._profile_statistics:
                        # Use global mean from all periods
                        all_means = [
                            stats["mean"]
                            for stats in self._profile_statistics.values()
                        ]
                        predictions[i] = np.mean(all_means)
                        if i == 0:
                            logger.warning(
                                "Period %d has no trained model, using global mean",
                                period,
                            )
                    else:
                        predictions[i] = 1.0  # Default to average
                    continue

                # Get model and scaler for this period
                model = self._models[period]
                scaler = self._scalers[period]

                # Extract features for this sample
                X_sample = X_features.iloc[[i]]

                # Scale features
                X_scaled = scaler.transform(X_sample)

                # Predict
                pred = model.predict(X_scaled)[0]

                # Apply bounds
                pred_bounded = self._apply_profile_bounds(pred, period)

                predictions[i] = pred_bounded

            # Create result DataFrame
            result_df = pd.DataFrame(
                {"h0": predictions},
                index=X.index,
            )

            logger.debug("Generated predictions with shape %s", result_df.shape)

            return result_df

        except Exception as e:
            msg = f"SVM profile prediction failed: {e}"
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from e

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores.

        SVM models don't provide direct feature importance like tree-based
        models. This returns an empty dict.

        Returns:
            Empty dictionary (SVM doesn't have feature importance).

        Raises:
            ValueError: If model is not fitted.
        """
        self._check_is_fitted()
        logger.debug("SVM models do not provide feature importance")
        return {}

    def _prepare_profile_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[int, dict[str, pd.DataFrame | pd.Series]]:
        """Group data by semi-hourly period.

        Args:
            X: Features DataFrame with DatetimeIndex.
            y: Target Series with profile ratios.

        Returns:
            Dictionary mapping period (0-47) to dict with "X" and "y" data.
        """
        periods = self._extract_periods(X.index)

        # Group by period
        period_data: dict[int, dict[str, pd.DataFrame | pd.Series]] = {}

        for period in range(48):
            mask = periods == period
            n_samples = mask.sum()

            if n_samples == 0:
                continue

            # Extract data for this period
            X_period = X[mask].copy()
            y_period = y[mask].copy()

            # If X has no columns, create temporal features
            if len(X_period.columns) == 0:
                X_period = self._prepare_period_features(X_period.index)

            period_data[period] = {
                "X": X_period,
                "y": y_period,
            }

            # Compute statistics for this period
            self._profile_statistics[period] = {
                "mean": float(y_period.mean()),
                "std": float(y_period.std()),
                "min": float(y_period.min()),
                "max": float(y_period.max()),
                "q25": float(y_period.quantile(0.25)),
                "q75": float(y_period.quantile(0.75)),
                "n_samples": n_samples,
            }

        logger.debug(
            "Prepared data for %d periods (out of 48 possible)",
            len(period_data),
        )

        return period_data

    def _prepare_period_features(self, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """Extract temporal and calendar features from timestamps.

        Args:
            timestamps: DatetimeIndex with semi-hourly timestamps.

        Returns:
            DataFrame with temporal features.
        """
        features = pd.DataFrame(index=timestamps)

        # Basic temporal features
        features["dayofweek"] = timestamps.dayofweek  # 0-6
        features["day"] = timestamps.day  # 1-31
        features["month"] = timestamps.month  # 1-12
        features["quarter"] = timestamps.quarter  # 1-4
        features["is_weekend"] = (timestamps.dayofweek >= 5).astype(int)  # 0/1

        # Cyclical encoding for month (optional, helps SVM)
        features["month_sin"] = np.sin(2 * np.pi * timestamps.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * timestamps.month / 12)

        # Cyclical encoding for day of week
        features["dow_sin"] = np.sin(2 * np.pi * timestamps.dayofweek / 7)
        features["dow_cos"] = np.cos(2 * np.pi * timestamps.dayofweek / 7)

        return features

    def _train_period_model(
        self,
        period: int,
        data: dict[str, pd.DataFrame | pd.Series],
    ) -> None:
        """Train SVM model for a specific period.

        Args:
            period: Semi-hourly period (0-47).
            data: Dictionary with "X" and "y" data for this period.
        """
        X_period = data["X"]
        y_period = data["y"]

        # Create scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_period)

        # Train model
        if self._svm_config.optimize_hyperparams:
            # Use GridSearchCV
            model = self._optimize_svm_hyperparams(X_scaled, y_period)
        else:
            # Use fixed parameters
            model = SVR(**self._svm_config.to_svr_params())
            model.fit(X_scaled, y_period)

        # Store model and scaler
        self._models[period] = model
        self._scalers[period] = scaler

        logger.debug(
            "Trained SVM for period %d: %d samples, mean=%.3f, std=%.3f",
            period,
            len(y_period),
            y_period.mean(),
            y_period.std(),
        )

    def _optimize_svm_hyperparams(
        self,
        X: np.ndarray,
        y: pd.Series,
    ) -> SVR:
        """Optimize SVM hyperparameters using GridSearchCV.

        Args:
            X: Scaled feature matrix.
            y: Target values.

        Returns:
            Fitted SVR model with best parameters.
        """
        # Create base SVR
        svr = SVR(kernel=self._svm_config.kernel)

        # Get parameter grid
        param_grid = self._svm_config.get_param_grid()

        # Create GridSearchCV
        grid_search = GridSearchCV(
            svr,
            param_grid,
            cv=self._svm_config.cv_folds,
            scoring="neg_mean_squared_error",
            n_jobs=self._svm_config.n_jobs,
            verbose=0,
        )

        # Fit
        grid_search.fit(X, y)

        logger.debug(
            "GridSearchCV best params: %s (score: %.4f)",
            grid_search.best_params_,
            -grid_search.best_score_,  # Negative because scoring is neg_mse
        )

        return grid_search.best_estimator_

    def _apply_profile_bounds(self, prediction: float, period: int) -> float:
        """Apply bounds to profile ratio prediction.

        Uses training statistics to bound predictions within reasonable range.

        Args:
            prediction: Raw prediction from SVM.
            period: Semi-hourly period (0-47).

        Returns:
            Bounded prediction.
        """
        if period not in self._profile_statistics:
            # Use global bounds
            lower = self._svm_config.min_profile_ratio
            upper = self._svm_config.max_profile_ratio
        else:
            stats = self._profile_statistics[period]

            # Use config bounds with tightening based on training statistics
            # Lower bound: max of config min and (Q25 - 2*std)
            lower = max(
                self._svm_config.min_profile_ratio,
                stats["q25"] - 2 * stats["std"],
            )

            # Upper bound: min of config max and (Q75 + 2*std)
            upper = min(
                self._svm_config.max_profile_ratio,
                stats["q75"] + 2 * stats["std"],
            )

            # Ensure bounds are valid
            lower = max(lower, 0.01)  # Never go below 1%
            upper = max(upper, lower + 0.1)  # Ensure upper > lower

        # Clip prediction
        bounded = np.clip(prediction, lower, upper)

        return bounded

    def _extract_periods(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Extract semi-hourly period (0-47) from timestamps.

        Args:
            timestamps: DatetimeIndex with semi-hourly timestamps.

        Returns:
            Array of periods (0-47).
        """
        # Period = hour * 2 + (minute // 30)
        periods = timestamps.hour * 2 + (timestamps.minute // 30)
        return periods.to_numpy()

    def get_diagnostics(self) -> dict[str, Any]:
        """Get model diagnostics and statistics.

        Returns:
            Dictionary with diagnostic information.
        """
        self._check_is_fitted()

        diagnostics = {
            "model_name": self.name,
            "model_version": self.version,
            "config": self._svm_config.model_dump(),
            "n_periods_trained": len(self._models),
            "trained_periods": sorted(self._models.keys()),
            "profile_statistics": self._profile_statistics,
            "training_metadata": self._training_metadata,
        }

        logger.debug("Retrieved model diagnostics")
        return diagnostics

    def save(self, path: str | Path) -> None:
        """Save model to disk using joblib.

        Args:
            path: File path to save the model.

        Raises:
            ValueError: If model is not fitted.
            OSError: If file cannot be written.
        """
        self._check_is_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            model_data = {
                "models": self._models,
                "scalers": self._scalers,
                "profile_statistics": self._profile_statistics,
                "config": self._svm_config.model_dump(),
                "training_metadata": self._training_metadata,
                "feature_names": self._feature_names,
                "period_feature_names": self._period_feature_names,
                "is_fitted": self._is_fitted,
                "uses_auto_temporal_features": getattr(self, "_uses_auto_temporal_features", False),
            }
            joblib.dump(model_data, path)
            logger.info("Saved SVM profile model to %s", path)
        except Exception as e:
            msg = f"Failed to save model to {path}: {e}"
            raise OSError(msg) from e

    @classmethod
    def load(cls, path: str | Path) -> "SVMProfileModel":
        """Load model from disk using joblib.

        Args:
            path: File path to load the model from.

        Returns:
            Loaded model instance.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            ValueError: If loaded object is not valid.
            OSError: If file cannot be read.
        """
        path = Path(path)

        if not path.exists():
            msg = f"Model file not found: {path}"
            raise FileNotFoundError(msg)

        try:
            model_data = joblib.load(path)
        except Exception as e:
            msg = f"Failed to load model from {path}: {e}"
            raise OSError(msg) from e

        # Create new instance
        model = cls()

        # Restore state
        model._models = model_data["models"]
        model._scalers = model_data["scalers"]
        model._profile_statistics = model_data["profile_statistics"]
        model._svm_config = SVMProfileConfig(**model_data["config"])
        model._training_metadata = model_data["training_metadata"]
        model._feature_names = model_data["feature_names"]
        model._period_feature_names = model_data["period_feature_names"]
        model._is_fitted = model_data["is_fitted"]
        model._uses_auto_temporal_features = model_data.get("uses_auto_temporal_features", False)

        logger.info("Loaded SVM profile model from %s", path)
        return model
