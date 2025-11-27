"""Holt-Winters profile model for semi-hourly load profile forecasting.

This module implements the HoltWintersProfileModel class for forecasting
semi-hourly load profiles using 48 independent Holt-Winters exponential
smoothing models. Each model predicts the profile ratio (load / daily_demand_mean)
for a specific semi-hourly period (0-47).

Example:
    ```python
    from src.models.profile.holt_winters_profile import HoltWintersProfileModel
    from src.models.config.holtwinters_config import HoltWintersProfileConfig
    import pandas as pd

    # Create model
    model = HoltWintersProfileModel()

    # Prepare data with semi-hourly timestamps
    # X should have DatetimeIndex at semi-hourly frequency
    # y should be profile ratio values (load / daily_demand_mean)
    config = HoltWintersProfileConfig(
        seasonal="add",
        seasonal_periods=7,
        trend=None
    )

    # Train 48 models
    model.fit(X_train, y_train, config={"holtwinters_config": config})

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
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.models.base.model import BaseModel
from src.models.config.holtwinters_config import HoltWintersProfileConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HoltWintersProfileModel(BaseModel):
    """Holt-Winters model for semi-hourly profile ratio forecasting.

    This model trains 48 independent Holt-Winters exponential smoothing models
    (one per semi-hourly period 0-47) to predict profile ratios. Each model:
    - Captures weekly seasonality (day-of-week patterns)
    - Optionally captures trend in profile evolution
    - Uses exponential smoothing for robust forecasting
    - Provides interpretable level and seasonal components

    Profile ratios represent the ratio of semi-hourly load to daily demand mean.
    They typically range from 0.5 to 2.0, with values:
    - < 1: Below daily average (nighttime, low demand)
    - = 1: At daily average
    - > 1: Above daily average (peak hours)

    Attributes:
        _holtwinters_config: HoltWintersProfileConfig instance with hyperparameters.
        _models: Dictionary mapping period (0-47) to fitted ExponentialSmoothing result.
        _profile_statistics: Statistics per period for bounding predictions.
        _period_feature_names: Feature names used for training (empty for Holt-Winters).
    """

    def __init__(self) -> None:
        """Initialize Holt-Winters profile model."""
        super().__init__()
        self._holtwinters_config: HoltWintersProfileConfig | None = None
        self._models: dict[int, Any] = {}  # period -> fitted result
        self._profile_statistics: dict[int, dict[str, float]] = {}
        self._period_feature_names: list[str] = []
        logger.debug("Initialized HoltWintersProfileModel")

    @property
    def name(self) -> str:
        """Return the model name identifier.

        Returns:
            Model name string.
        """
        return "holt_winters_profile"

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

        Holt-Winters profile models predict the current period only (D+0), as they
        forecast the profile ratio for the same day based on seasonal patterns.

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
        """Train 48 Holt-Winters models for semi-hourly profile forecasting.

        This method:
        1. Validates and processes input data
        2. Extracts semi-hourly period (0-47) from timestamps
        3. Computes profile statistics per period
        4. Trains one Holt-Winters model per period with exponential smoothing
        5. Stores fitted models and metadata

        Args:
            X: Training features DataFrame with DatetimeIndex at semi-hourly frequency.
                Not used directly (Holt-Winters uses time patterns).
            y: Training target (profile ratios) as Series or single-column DataFrame.
                Should be profile ratio values (load / daily_demand_mean).
            config: Training configuration dictionary. Should contain:
                - "holtwinters_config": HoltWintersProfileConfig instance or dict

        Raises:
            ValueError: If input data is invalid or insufficient.
            RuntimeError: If Holt-Winters training fails for all periods.

        Example:
            ```python
            config = {
                "holtwinters_config": HoltWintersProfileConfig(
                    seasonal="add",
                    seasonal_periods=7,
                    trend=None,
                    min_samples_per_period=14
                )
            }
            model.fit(X_train, y_train, config)
            ```
        """
        logger.info("Starting Holt-Winters profile model training")

        # Parse config
        hw_config_input = config.get("holtwinters_config", {})
        if isinstance(hw_config_input, dict):
            self._holtwinters_config = HoltWintersProfileConfig(**hw_config_input)
        elif isinstance(hw_config_input, HoltWintersProfileConfig):
            self._holtwinters_config = hw_config_input
        else:
            msg = f"Invalid holtwinters_config type: {type(hw_config_input)}"
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

        # Train models for each period
        n_periods_trained = 0
        n_periods_skipped = 0

        for period in range(48):
            if period not in period_data:
                logger.warning("Period %d has no data, skipping", period)
                n_periods_skipped += 1
                continue

            data = period_data[period]
            n_samples = len(data["y"])

            if n_samples < self._holtwinters_config.min_samples_per_period:
                logger.warning(
                    "Period %d has only %d samples (min required: %d), skipping",
                    period,
                    n_samples,
                    self._holtwinters_config.min_samples_per_period,
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
        self._feature_names = []  # Holt-Winters doesn't use features
        self._training_metadata = {
            "trained_at": datetime.now(UTC),
            "training_config": config,
            "n_samples_total": len(X),
            "n_periods_trained": n_periods_trained,
            "n_periods_skipped": n_periods_skipped,
            "holtwinters_config": self._holtwinters_config.model_dump(),
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

            # Generate predictions for each sample
            predictions = np.zeros(len(X))

            for i, period in enumerate(periods):
                if period not in self._models:
                    # Use mean ratio from statistics if this period wasn't trained
                    if period in self._profile_statistics:
                        predictions[i] = self._profile_statistics[period]["mean"]
                    elif self._profile_statistics:
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

                # Get fitted model for this period
                fitted_model = self._models[period]

                # Predict next step (forecast 1 period ahead)
                pred = fitted_model.forecast(steps=1).iloc[0]

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
            msg = f"Holt-Winters profile prediction failed: {e}"
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from e

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores.

        Holt-Winters models don't use features in the traditional sense, so
        this returns an empty dict.

        Returns:
            Empty dictionary (Holt-Winters doesn't have feature importance).

        Raises:
            ValueError: If model is not fitted.
        """
        self._check_is_fitted()
        logger.debug("Holt-Winters models do not provide feature importance")
        return {}

    def _prepare_profile_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[int, dict[str, pd.Series]]:
        """Group data by semi-hourly period.

        Args:
            X: Features DataFrame with DatetimeIndex.
            y: Target Series with profile ratios.

        Returns:
            Dictionary mapping period (0-47) to dict with "y" data.
        """
        periods = self._extract_periods(X.index)

        # Group by period
        period_data: dict[int, dict[str, pd.Series]] = {}

        for period in range(48):
            mask = periods == period
            n_samples = mask.sum()

            if n_samples == 0:
                continue

            # Extract data for this period
            y_period = y[mask].copy()

            # Sort by date to ensure temporal order
            y_period = y_period.sort_index()

            period_data[period] = {
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

    def _train_period_model(
        self,
        period: int,
        data: dict[str, pd.Series],
    ) -> None:
        """Train Holt-Winters model for a specific period.

        Args:
            period: Semi-hourly period (0-47).
            data: Dictionary with "y" data for this period.
        """
        y_period = data["y"]

        # Check for non-positive values if using multiplicative components
        if self._holtwinters_config.seasonal == "mul" or self._holtwinters_config.trend == "mul":
            if (y_period <= 0).any():
                logger.warning(
                    "Period %d has non-positive values, cannot use multiplicative model. Switching to additive.",
                    period
                )
                # Temporarily override config for this period
                sm_params = self._holtwinters_config.to_statsmodels_params()
                if self._holtwinters_config.seasonal == "mul":
                    sm_params["seasonal"] = "add"
                if self._holtwinters_config.trend == "mul":
                    sm_params["trend"] = "add"
            else:
                sm_params = self._holtwinters_config.to_statsmodels_params()
        else:
            sm_params = self._holtwinters_config.to_statsmodels_params()

        # Create and fit model
        try:
            model = ExponentialSmoothing(
                y_period,
                **sm_params
            )

            fitted_model = model.fit(optimized=True)

            # Store fitted model
            self._models[period] = fitted_model

            logger.debug(
                "Trained Holt-Winters for period %d: %d samples, mean=%.3f, std=%.3f",
                period,
                len(y_period),
                y_period.mean(),
                y_period.std(),
            )

        except Exception as e:
            # If fitting fails, try with simpler model (no trend)
            logger.warning("Period %d: Failed with configured params, trying simpler model: %s", period, e)

            try:
                # Fallback to simplest model (level + seasonal)
                simple_params = {
                    "trend": None,
                    "seasonal": "add" if self._holtwinters_config.seasonal else None,
                    "seasonal_periods": self._holtwinters_config.seasonal_periods if self._holtwinters_config.seasonal else None,
                    "damped_trend": False,
                    "use_boxcox": False,
                    "initialization_method": "heuristic",
                }

                model = ExponentialSmoothing(y_period, **simple_params)
                fitted_model = model.fit(optimized=False)
                self._models[period] = fitted_model

                logger.debug("Period %d: Fallback model trained successfully", period)

            except Exception as e2:
                logger.error("Period %d: Fallback model also failed: %s", period, e2)
                raise

    def _apply_profile_bounds(self, prediction: float, period: int) -> float:
        """Apply bounds to profile ratio prediction.

        Uses training statistics to bound predictions within reasonable range.

        Args:
            prediction: Raw prediction from Holt-Winters.
            period: Semi-hourly period (0-47).

        Returns:
            Bounded prediction.
        """
        if period not in self._profile_statistics:
            # Use global bounds
            lower = self._holtwinters_config.min_profile_ratio
            upper = self._holtwinters_config.max_profile_ratio
        else:
            stats = self._profile_statistics[period]

            # Use config bounds with tightening based on training statistics
            # Lower bound: max of config min and (Q25 - 2*std)
            lower = max(
                self._holtwinters_config.min_profile_ratio,
                stats["q25"] - 2 * stats["std"],
            )

            # Upper bound: min of config max and (Q75 + 2*std)
            upper = min(
                self._holtwinters_config.max_profile_ratio,
                stats["q75"] + 2 * stats["std"],
            )

            # Ensure bounds are valid
            lower = max(lower, 0.01)  # Never go below 1%
            upper = max(upper, lower + 0.1)  # Ensure upper > lower

        # Clip prediction
        return np.clip(prediction, lower, upper)

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
            "config": self._holtwinters_config.model_dump(),
            "n_periods_trained": len(self._models),
            "trained_periods": sorted(self._models.keys()),
            "profile_statistics": self._profile_statistics,
            "training_metadata": self._training_metadata,
        }

        # Add sample model parameters from first few periods
        sample_periods = sorted(self._models.keys())[:3]
        diagnostics["sample_model_params"] = {}

        for period in sample_periods:
            fitted_model = self._models[period]
            diagnostics["sample_model_params"][period] = {
                "smoothing_level": float(fitted_model.params["smoothing_level"]),
                "smoothing_trend": float(fitted_model.params.get("smoothing_trend", 0)),
                "smoothing_seasonal": float(fitted_model.params.get("smoothing_seasonal", 0)),
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
                "profile_statistics": self._profile_statistics,
                "config": self._holtwinters_config.model_dump(),
                "training_metadata": self._training_metadata,
                "feature_names": self._feature_names,
                "period_feature_names": self._period_feature_names,
                "is_fitted": self._is_fitted,
            }
            joblib.dump(model_data, path)
            logger.info("Saved Holt-Winters profile model to %s", path)
        except Exception as e:
            msg = f"Failed to save model to {path}: {e}"
            raise OSError(msg) from e

    @classmethod
    def load(cls, path: str | Path) -> "HoltWintersProfileModel":
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
        model._profile_statistics = model_data["profile_statistics"]
        model._holtwinters_config = HoltWintersProfileConfig(**model_data["config"])
        model._training_metadata = model_data["training_metadata"]
        model._feature_names = model_data["feature_names"]
        model._period_feature_names = model_data["period_feature_names"]
        model._is_fitted = model_data["is_fitted"]

        logger.info("Loaded Holt-Winters profile model from %s", path)
        return model
