"""Holt-Winters exponential smoothing model for demand mean forecasting.

This module implements the HoltWintersDemandMeanModel class for forecasting
daily demand mean using Holt-Winters exponential smoothing from statsmodels.
It supports additive and multiplicative trend/seasonality, damped trend, and
provides prediction intervals.

Example:
    ```python
    from src.models.demand_mean.holt_winters import HoltWintersDemandMeanModel
    from src.models.config.holtwinters_config import HoltWintersConfig
    import pandas as pd

    # Create model
    model = HoltWintersDemandMeanModel()

    # Prepare daily demand data
    # X should have DatetimeIndex at daily frequency
    # y should be daily demand mean values
    config = HoltWintersConfig(
        trend="add",
        seasonal="add",
        seasonal_periods=7
    )

    # Train model
    model.fit(X_daily, y_daily, config={"holtwinters_config": config})

    # Generate forecasts for D+0 to D+8
    predictions = model.predict(X_test, horizons=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    # Get prediction intervals
    intervals = model.get_prediction_intervals(X_test, horizons=[0, 1])

    # Get seasonal decomposition
    decomp = model.get_seasonal_decomposition()
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
from src.models.config.holtwinters_config import HoltWintersConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HoltWintersDemandMeanModel(BaseModel):
    """Holt-Winters exponential smoothing for daily demand mean forecasting.

    This model uses exponential smoothing to capture:
    - Level: baseline demand
    - Trend: long-term increase/decrease
    - Seasonality: weekly patterns

    The Holt-Winters method automatically estimates smoothing parameters
    (alpha, beta, gamma) that minimize forecast error. It is particularly
    effective for data with clear trend and seasonal patterns.

    Advantages:
    - Simple, interpretable components (level, trend, seasonal)
    - Automatic parameter estimation
    - Fast training and prediction
    - Provides prediction intervals
    - Handles multiple seasonal patterns

    Attributes:
        _holtwinters_config: HoltWintersConfig instance with hyperparameters.
        _model: statsmodels ExponentialSmoothing instance.
        _fitted_result: Fitted model result with parameters and forecasts.
        _training_series: Training data used for fitting (for diagnostics).
    """

    def __init__(self) -> None:
        """Initialize Holt-Winters demand mean model."""
        super().__init__()
        self._holtwinters_config: HoltWintersConfig | None = None
        self._model: ExponentialSmoothing | None = None
        self._fitted_result: Any | None = None
        self._training_series: pd.Series | None = None
        logger.debug("Initialized HoltWintersDemandMeanModel")

    @property
    def name(self) -> str:
        """Return the model name identifier.

        Returns:
            Model name string.
        """
        return "holt_winters_demand_mean"

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

        Returns:
            List of integers from 0 to 8 representing D+0 to D+8 horizons.
        """
        return list(range(9))  # D+0 through D+8

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803, ARG002
        y: pd.Series | pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        """Train the Holt-Winters model on daily demand mean data.

        This method:
        1. Validates and processes input data
        2. Aggregates semi-hourly data to daily mean if needed
        3. Fits Holt-Winters exponential smoothing model
        4. Stores fitted model and training metadata

        Args:
            X: Training features DataFrame with DatetimeIndex.
                Can be at any frequency; will be aggregated to daily.
            y: Training target (demand) as Series or single-column DataFrame.
                Will be aggregated to daily mean.
            config: Training configuration dictionary. Should contain:
                - "holtwinters_config": HoltWintersConfig instance or dict

        Raises:
            ValueError: If input data is invalid or insufficient.
            RuntimeError: If Holt-Winters fitting fails.

        Example:
            ```python
            config = {
                "holtwinters_config": HoltWintersConfig(
                    trend="add",
                    seasonal="add",
                    seasonal_periods=7,
                    damped_trend=False
                )
            }
            model.fit(X_train, y_train, config)
            ```
        """
        logger.info("Starting Holt-Winters demand mean model training")

        # Parse config
        hw_config_input = config.get("holtwinters_config", {})
        if isinstance(hw_config_input, dict):
            self._holtwinters_config = HoltWintersConfig(**hw_config_input)
        elif isinstance(hw_config_input, HoltWintersConfig):
            self._holtwinters_config = hw_config_input
        else:
            msg = f"Invalid holtwinters_config type: {type(hw_config_input)}"
            raise ValueError(msg)

        # Convert y to Series if DataFrame
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                msg = f"y must be single column, got {y.shape[1]} columns"
                raise ValueError(msg)
            y_series = y.iloc[:, 0]
        else:
            y_series = y

        # Aggregate to daily mean if needed
        logger.debug("Aggregating data to daily mean")
        y_daily = self._convert_to_daily_mean(y_series)

        # Validate sufficient data
        min_samples = self._holtwinters_config.min_training_days
        if len(y_daily) < min_samples:
            msg = (
                f"Insufficient data for training: need at least {min_samples} days, "
                f"got {len(y_daily)} days"
            )
            raise ValueError(msg)

        # Check for non-positive values (required for multiplicative models)
        if (self._holtwinters_config.seasonal == "mul" or self._holtwinters_config.trend == "mul") and (y_daily <= 0).any():
            msg = (
                "Multiplicative trend/seasonal requires all positive values. "
                f"Found {(y_daily <= 0).sum()} non-positive values."
            )
            raise ValueError(msg)

        logger.info(
            "Training on %d daily observations (mean=%.2f, std=%.2f)",
            len(y_daily),
            y_daily.mean(),
            y_daily.std(),
        )

        # Create ExponentialSmoothing model
        sm_params = self._holtwinters_config.to_statsmodels_params()

        logger.info(
            "Fitting Holt-Winters: trend=%s, seasonal=%s, seasonal_periods=%s, damped=%s",
            self._holtwinters_config.trend,
            self._holtwinters_config.seasonal,
            self._holtwinters_config.seasonal_periods if self._holtwinters_config.seasonal else None,
            self._holtwinters_config.damped_trend,
        )

        try:
            self._model = ExponentialSmoothing(
                y_daily,
                **sm_params
            )

            # Fit model with optimized parameters
            self._fitted_result = self._model.fit(optimized=True)

            logger.info(
                "Holt-Winters model fitted successfully. Smoothing level: %.4f",
                self._fitted_result.params["smoothing_level"]
            )

        except Exception as e:
            msg = f"Holt-Winters model fitting failed: {e}"
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from e

        # Store training data and metadata
        self._training_series = y_daily
        self._feature_names = []  # Holt-Winters doesn't use features
        self._training_metadata = {
            "trained_at": datetime.now(UTC),
            "training_config": config,
            "n_samples": len(y_daily),
            "data_mean": float(y_daily.mean()),
            "data_std": float(y_daily.std()),
            "data_min": float(y_daily.min()),
            "data_max": float(y_daily.max()),
            "holtwinters_config": self._holtwinters_config.model_dump(),
            "smoothing_params": {
                "smoothing_level": float(self._fitted_result.params["smoothing_level"]),
                "smoothing_trend": float(self._fitted_result.params.get("smoothing_trend", 0)),
                "smoothing_seasonal": float(self._fitted_result.params.get("smoothing_seasonal", 0)),
                "damping_trend": float(self._fitted_result.params.get("damping_trend", 0)),
            }
        }
        self._is_fitted = True

        logger.info(
            "Holt-Winters training completed: %d samples, mean=%.2f, std=%.2f",
            len(y_daily),
            y_daily.mean(),
            y_daily.std(),
        )

    def predict(
        self,
        X: pd.DataFrame,  # noqa: N803, ARG002
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Generate demand mean forecasts for specified horizons.

        Args:
            X: Features DataFrame with DatetimeIndex. Used for timestamp info
                but not as predictors (Holt-Winters uses past values only).
            horizons: List of horizons to predict (0-8). If None, predicts all
                     supported horizons.

        Returns:
            DataFrame with predictions. Shape (1, n_horizons).
            Columns are named "h0", "h1", etc. for each horizon.
            Returns single row since Holt-Winters produces forecasts from
            the end of training data.

        Raises:
            ValueError: If model is not fitted or horizons are invalid.
            RuntimeError: If prediction fails.

        Example:
            ```python
            # Predict D+0 through D+3
            predictions = model.predict(X_test, horizons=[0, 1, 2, 3])
            # Result shape: (1, 4)
            # Columns: ["h0", "h1", "h2", "h3"]
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

        max_horizon = max(horizons)
        logger.debug("Generating forecasts for horizons: %s (max: D+%d)", horizons, max_horizon)

        try:
            # Generate forecasts
            # steps = max_horizon + 1 because we need forecasts from h0 to h{max_horizon}
            forecasts = self._fitted_result.forecast(steps=max_horizon + 1)

            # Extract point forecasts for requested horizons
            result_data = {}
            for horizon in horizons:
                result_data[f"h{horizon}"] = forecasts.iloc[horizon]

            # Create result DataFrame with single row
            result_df = pd.DataFrame([result_data])

            logger.debug(
                "Generated forecasts: %d horizons, shape %s",
                len(horizons),
                result_df.shape,
            )

            return result_df

        except Exception as e:
            msg = f"Holt-Winters prediction failed: {e}"
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from e

    def get_prediction_intervals(
        self,
        X: pd.DataFrame,  # noqa: N803, ARG002
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Generate prediction intervals for forecasts.

        Returns confidence intervals at the levels specified in HoltWintersConfig.

        Args:
            X: Features DataFrame with DatetimeIndex.
            horizons: List of horizons to predict (0-8). If None, predicts all
                     supported horizons.

        Returns:
            DataFrame with prediction intervals. Columns include:
                - "h0", "h1", ... : Point forecasts
                - "h0_lo80", "h0_hi80", ... : 80% prediction intervals
                - "h0_lo95", "h0_hi95", ... : 95% prediction intervals
                (depending on configured interval levels)

        Raises:
            ValueError: If model is not fitted or horizons are invalid.
            RuntimeError: If prediction fails.

        Example:
            ```python
            intervals = model.get_prediction_intervals(X_test, horizons=[0, 1, 2])
            # Columns: h0, h1, h2, h0_lo80, h0_hi80, h0_lo95, h0_hi95, ...
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

        max_horizon = max(horizons)
        levels = self._holtwinters_config.prediction_intervals

        logger.debug(
            "Generating forecasts with intervals for horizons: %s, levels: %s",
            horizons,
            levels,
        )

        try:
            # Generate forecasts with prediction intervals using simulation
            # statsmodels uses alpha values (e.g., 0.05 for 95% interval)
            # We need to convert our percentiles to alphas
            alphas = [(100 - level) / 100.0 for level in levels]

            # Get forecasts with intervals
            forecast_result = self._fitted_result.forecast(steps=max_horizon + 1)

            # Create result data
            result_data = {}

            # Add point forecasts
            for horizon in horizons:
                result_data[f"h{horizon}"] = forecast_result.iloc[horizon]

            # Simulate prediction intervals using standard error approach
            # This is a simplified approach; more sophisticated methods use simulation
            if hasattr(self._fitted_result, "sse") and self._fitted_result.sse is not None:
                # Calculate standard error
                sse = self._fitted_result.sse
                n = len(self._training_series)
                mse = sse / n
                std_error = np.sqrt(mse)

                # For each horizon, calculate intervals
                for horizon in horizons:
                    point_forecast = forecast_result.iloc[horizon]

                    # Standard error increases with horizon (simplified model)
                    horizon_std_error = std_error * np.sqrt(1 + horizon * 0.1)

                    # Calculate intervals for each level
                    for level, alpha in zip(levels, alphas):
                        # Use normal approximation for confidence intervals
                        # z-score for the confidence level
                        z_score = 1.96 if level == 95 else 1.28  # 95% or 80%
                        if level not in [80, 95]:
                            # For other levels, use approximation
                            from scipy import stats
                            z_score = stats.norm.ppf(1 - alpha / 2)

                        margin = z_score * horizon_std_error
                        result_data[f"h{horizon}_lo{level}"] = max(0, point_forecast - margin)
                        result_data[f"h{horizon}_hi{level}"] = point_forecast + margin
            else:
                logger.warning("SSE not available, cannot compute prediction intervals")

            # Create result DataFrame
            result_df = pd.DataFrame([result_data])

            logger.debug(
                "Generated forecasts with intervals: %d horizons, %d levels, shape %s",
                len(horizons),
                len(levels),
                result_df.shape,
            )

            return result_df

        except Exception as e:
            msg = f"Holt-Winters prediction with intervals failed: {e}"
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from e

    def get_seasonal_decomposition(self) -> dict[str, Any]:
        """Get seasonal decomposition components from fitted model.

        Extracts the level, trend, and seasonal components estimated by
        the Holt-Winters model. Useful for understanding model behavior
        and diagnosing fit quality.

        Returns:
            Dictionary containing:
                - "level": Series with level component
                - "trend": Series with trend component (None if no trend)
                - "seasonal": Series with seasonal component (None if no seasonal)
                - "fitted_values": Series with in-sample fitted values
                - "residuals": Series with in-sample residuals

        Raises:
            ValueError: If model is not fitted.

        Example:
            ```python
            decomp = model.get_seasonal_decomposition()
            level = decomp["level"]
            trend = decomp["trend"]
            seasonal = decomp["seasonal"]
            ```
        """
        self._check_is_fitted()

        logger.debug("Extracting seasonal decomposition components")

        decomposition = {
            "level": pd.Series(
                self._fitted_result.level,
                index=self._training_series.index
            ) if hasattr(self._fitted_result, "level") else None,
            "trend": pd.Series(
                self._fitted_result.trend,
                index=self._training_series.index
            ) if hasattr(self._fitted_result, "trend") and self._holtwinters_config.trend else None,
            "seasonal": pd.Series(
                self._fitted_result.season,
                index=self._training_series.index
            ) if hasattr(self._fitted_result, "season") and self._holtwinters_config.seasonal else None,
            "fitted_values": pd.Series(
                self._fitted_result.fittedvalues,
                index=self._training_series.index
            ),
            "residuals": pd.Series(
                self._fitted_result.resid,
                index=self._training_series.index
            ),
        }

        logger.info(
            "Decomposition extracted: level=%s, trend=%s, seasonal=%s",
            decomposition["level"] is not None,
            decomposition["trend"] is not None,
            decomposition["seasonal"] is not None,
        )

        return decomposition

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores.

        Holt-Winters models don't use features in the traditional sense (they
        use past values and time components only), so this returns an empty dict.

        Returns:
            Empty dictionary (Holt-Winters doesn't have feature importance).

        Raises:
            ValueError: If model is not fitted.
        """
        self._check_is_fitted()
        logger.debug("Holt-Winters model has no feature importance (uses past values only)")
        return {}

    def get_diagnostics(self) -> dict[str, Any]:
        """Get model diagnostics and statistics.

        Returns information about the fitted Holt-Winters model including
        smoothing parameters, information criteria, and training data statistics.

        Returns:
            Dictionary with diagnostic information:
                - "model_params": Fitted smoothing parameters
                - "training_stats": Statistics from training data
                - "config": HoltWintersConfig used for training
                - "aic": Akaike Information Criterion
                - "bic": Bayesian Information Criterion
                - "aicc": Corrected AIC

        Raises:
            ValueError: If model is not fitted.

        Example:
            ```python
            diagnostics = model.get_diagnostics()
            print(f"Smoothing level: {diagnostics['model_params']['smoothing_level']}")
            print(f"AIC: {diagnostics['aic']}")
            ```
        """
        self._check_is_fitted()

        diagnostics = {
            "model_name": self.name,
            "model_version": self.version,
            "config": self._holtwinters_config.model_dump(),
            "model_params": self._training_metadata.get("smoothing_params"),
            "training_stats": {
                "n_samples": self._training_metadata.get("n_samples"),
                "mean": self._training_metadata.get("data_mean"),
                "std": self._training_metadata.get("data_std"),
                "min": self._training_metadata.get("data_min"),
                "max": self._training_metadata.get("data_max"),
            },
            "trained_at": self._training_metadata.get("trained_at"),
        }

        # Add information criteria if available
        if hasattr(self._fitted_result, "aic"):
            diagnostics["aic"] = float(self._fitted_result.aic)
        if hasattr(self._fitted_result, "bic"):
            diagnostics["bic"] = float(self._fitted_result.bic)
        if hasattr(self._fitted_result, "aicc"):
            diagnostics["aicc"] = float(self._fitted_result.aicc)

        logger.debug("Retrieved model diagnostics")
        return diagnostics

    def _convert_to_daily_mean(self, series: pd.Series) -> pd.Series:
        """Convert semi-hourly or hourly data to daily mean.

        Aggregates input series to daily frequency by computing the mean
        for each day.

        Args:
            series: Input series with DatetimeIndex at any frequency.

        Returns:
            Series with daily frequency, indexed by date.

        Example:
            ```python
            # Input: 48 semi-hourly values per day
            # Output: 1 daily mean per day
            daily_mean = model._convert_to_daily_mean(semi_hourly_series)
            ```
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            msg = "Series must have DatetimeIndex"
            raise ValueError(msg)

        # Check if already daily
        if len(series) > 1:
            time_diff = (series.index[1] - series.index[0]).total_seconds()
            if time_diff >= 24 * 3600:  # Already daily or coarser
                logger.debug("Data is already at daily or coarser frequency")
                return series

        # Group by date and compute mean
        daily_mean = series.groupby(series.index.date).mean()

        # Convert index back to DatetimeIndex
        daily_mean.index = pd.to_datetime(daily_mean.index)

        logger.debug(
            "Aggregated to daily mean: %d observations -> %d days",
            len(series),
            len(daily_mean),
        )

        return daily_mean

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
            joblib.dump(self, path)
            logger.info("Saved Holt-Winters model to %s", path)
        except Exception as e:
            msg = f"Failed to save model to {path}: {e}"
            raise OSError(msg) from e

    @classmethod
    def load(cls, path: str | Path) -> "HoltWintersDemandMeanModel":
        """Load model from disk using joblib.

        Args:
            path: File path to load the model from.

        Returns:
            Loaded model instance.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            ValueError: If loaded object is not a HoltWintersDemandMeanModel instance.
            OSError: If file cannot be read.
        """
        path = Path(path)

        if not path.exists():
            msg = f"Model file not found: {path}"
            raise FileNotFoundError(msg)

        try:
            model = joblib.load(path)
        except Exception as e:
            msg = f"Failed to load model from {path}: {e}"
            raise OSError(msg) from e

        if not isinstance(model, HoltWintersDemandMeanModel):
            msg = f"Loaded object is not a HoltWintersDemandMeanModel instance: {type(model)}"
            raise ValueError(msg)

        logger.info("Loaded Holt-Winters model from %s", path)
        return model
