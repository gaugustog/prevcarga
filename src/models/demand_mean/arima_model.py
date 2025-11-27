"""ARIMA/SARIMA model for demand mean forecasting.

This module implements the ARIMADemandMeanModel class for forecasting daily
demand mean using AutoARIMA from the statsforecast library. It supports both
non-seasonal ARIMA and seasonal ARIMA (SARIMA) models with automatic parameter
selection.

Example:
    ```python
    from src.models.demand_mean.arima_model import ARIMADemandMeanModel
    from src.models.config.arima_config import ARIMAConfig
    import pandas as pd

    # Create model
    model = ARIMADemandMeanModel()

    # Prepare daily demand data
    # X should have DatetimeIndex at daily frequency
    # y should be daily demand mean values
    config = ARIMAConfig(seasonal=True, season_length=7)

    # Train model
    model.fit(X_daily, y_daily, config={"arima_config": config})

    # Generate forecasts for D+0 to D+8
    predictions = model.predict(X_test, horizons=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    # Get prediction intervals
    intervals = model.get_prediction_intervals(X_test, horizons=[0, 1])
    ```
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from src.models.base.model import BaseModel
from src.models.config.arima_config import ARIMAConfig
from src.models.demand_mean.demand_processor import DemandProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ARIMADemandMeanModel(BaseModel):
    """ARIMA/SARIMA model for daily demand mean forecasting.

    This model uses statsforecast's AutoARIMA to automatically select optimal
    (p,d,q) or (p,d,q)(P,D,Q)s parameters for forecasting demand mean.
    It handles:
    - Automatic parameter selection via information criteria (AIC/BIC)
    - Seasonal patterns (e.g., weekly seasonality with period=7)
    - Multi-horizon forecasting (D+0 through D+8)
    - Prediction intervals at multiple confidence levels

    The model expects daily aggregated data and is typically used as the first
    stage in hierarchical forecasting models.

    Attributes:
        _arima_config: ARIMAConfig instance with model hyperparameters.
        _statsforecast: Fitted StatsForecast instance with AutoARIMA model.
        _processor: DemandProcessor for data cleaning.
        _training_series: Training data used for fitting (for diagnostics).
    """

    def __init__(self) -> None:
        """Initialize ARIMA demand mean model."""
        super().__init__()
        self._arima_config: ARIMAConfig | None = None
        self._statsforecast: StatsForecast | None = None
        self._processor = DemandProcessor()
        self._training_series: pd.Series | None = None
        self._training_df: pd.DataFrame | None = None  # For statsforecast 2.0+ API
        logger.debug("Initialized ARIMADemandMeanModel")

    @property
    def name(self) -> str:
        """Return the model name identifier.

        Returns:
            Model name string.
        """
        return "arima_demand_mean"

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
        """Train the ARIMA model on daily demand mean data.

        This method:
        1. Validates and processes input data
        2. Aggregates semi-hourly data to daily mean if needed
        3. Fits AutoARIMA model with automatic parameter selection
        4. Stores fitted model and training metadata

        Args:
            X: Training features DataFrame with DatetimeIndex.
                Can be at any frequency; will be aggregated to daily.
            y: Training target (demand) as Series or single-column DataFrame.
                Will be aggregated to daily mean.
            config: Training configuration dictionary. Should contain:
                - "arima_config": ARIMAConfig instance or dict

        Raises:
            ValueError: If input data is invalid or insufficient.
            RuntimeError: If ARIMA fitting fails.

        Example:
            ```python
            config = {
                "arima_config": ARIMAConfig(
                    seasonal=True,
                    season_length=7,
                    min_training_days=14
                )
            }
            model.fit(X_train, y_train, config)
            ```
        """
        logger.info("Starting ARIMA demand mean model training")

        # Parse config
        arima_config_input = config.get("arima_config", {})
        if isinstance(arima_config_input, dict):
            self._arima_config = ARIMAConfig(**arima_config_input)
        elif isinstance(arima_config_input, ARIMAConfig):
            self._arima_config = arima_config_input
        else:
            msg = f"Invalid arima_config type: {type(arima_config_input)}"
            raise ValueError(msg)

        # Convert y to Series if DataFrame
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                msg = f"y must be single column, got {y.shape[1]} columns"
                raise ValueError(msg)
            y_series = y.iloc[:, 0]
        else:
            y_series = y

        # Aggregate to daily mean
        logger.debug("Aggregating data to daily mean")
        y_daily = self._convert_to_daily_mean(y_series)

        # Validate data
        validation = self._processor.validate_for_arima(
            y_daily,
            min_observations=self._arima_config.min_training_days,
        )

        if not validation["is_valid"]:
            issues_str = "; ".join(validation["issues"])
            msg = f"Data validation failed: {issues_str}"
            raise ValueError(msg)

        logger.info(
            "Training on %d daily observations (%.1f days)",
            validation["n_observations"],
            validation["n_observations"],
        )

        # Process demand series
        y_processed = self._processor.process_demand_series(
            y_daily,
            fill_missing=True,
            handle_outliers=True,
            outlier_z_threshold=3.5,
        )

        # Prepare data in statsforecast format
        # statsforecast expects: unique_id, ds (datetime), y (target)
        sf_df = pd.DataFrame(
            {
                "unique_id": "demand_mean",
                "ds": y_processed.index,
                "y": y_processed.to_numpy(),
            }
        )

        # Create AutoARIMA model
        sf_params = self._arima_config.to_statsforecast_params()
        auto_arima = AutoARIMA(**sf_params)

        logger.info(
            "Fitting AutoARIMA: seasonal=%s, season_length=%d, stepwise=%s",
            self._arima_config.seasonal,
            self._arima_config.season_length,
            self._arima_config.stepwise,
        )

        # Fit model
        try:
            self._statsforecast = StatsForecast(
                models=[auto_arima],
                freq="D",  # Daily frequency
                n_jobs=1,
            )

            self._statsforecast.fit(sf_df)

            logger.info("AutoARIMA model fitted successfully")

        except Exception as e:
            msg = f"ARIMA model fitting failed: {e}"
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from e

        # Store training data and metadata
        self._training_series = y_processed
        self._training_df = sf_df  # Store for prediction (statsforecast 2.0+ requirement)
        self._feature_names = []  # ARIMA doesn't use features
        self._training_metadata = {
            "trained_at": datetime.now(UTC),
            "training_config": config,
            "n_samples": len(y_processed),
            "data_mean": float(y_processed.mean()),
            "data_std": float(y_processed.std()),
            "data_min": float(y_processed.min()),
            "data_max": float(y_processed.max()),
            "arima_config": self._arima_config.model_dump(),
        }
        self._is_fitted = True

        logger.info(
            "ARIMA training completed: %d samples, mean=%.2f, std=%.2f",
            len(y_processed),
            y_processed.mean(),
            y_processed.std(),
        )

    def predict(
        self,
        X: pd.DataFrame,  # noqa: N803, ARG002
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Generate demand mean forecasts for specified horizons.

        Args:
            X: Features DataFrame with DatetimeIndex. Used for timestamp info
                but not as predictors (ARIMA uses past values only).
            horizons: List of horizons to predict (0-8). If None, predicts all
                     supported horizons.

        Returns:
            DataFrame with predictions. Shape (n_samples, n_horizons).
            Columns are named "h0", "h1", etc. for each horizon.
            For daily forecasts, n_samples will be the number of unique dates.

        Raises:
            ValueError: If model is not fitted or horizons are invalid.
            RuntimeError: If prediction fails.

        Example:
            ```python
            # Predict D+0 through D+3
            predictions = model.predict(X_test, horizons=[0, 1, 2, 3])
            # Result shape: (n_dates, 4)
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
            # h = max_horizon + 1 because statsforecast uses 1-based indexing
            # statsforecast 2.0+ requires df parameter
            forecasts = self._statsforecast.forecast(df=self._training_df, h=max_horizon + 1)

            # Extract point forecasts for requested horizons
            # forecasts has columns: unique_id, ds, AutoARIMA
            result_data = {}

            for horizon in horizons:
                # statsforecast returns all horizons in sequence
                # We need to extract the specific horizon
                # For simplicity, we'll take the (horizon+1)th row for each forecast origin
                result_data[f"h{horizon}"] = forecasts.iloc[horizon, :]["AutoARIMA"]

            # Create result DataFrame
            # For now, return a single-row DataFrame with forecasts
            result_df = pd.DataFrame([result_data])

            logger.debug(
                "Generated forecasts: %d horizons, shape %s",
                len(horizons),
                result_df.shape,
            )

            return result_df

        except Exception as e:
            msg = f"ARIMA prediction failed: {e}"
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from e

    def get_prediction_intervals(
        self,
        X: pd.DataFrame,  # noqa: N803, ARG002
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Generate prediction intervals for forecasts.

        Returns confidence intervals at the levels specified in ARIMAConfig.

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
        levels = self._arima_config.prediction_intervals

        logger.debug(
            "Generating forecasts with intervals for horizons: %s, levels: %s",
            horizons,
            levels,
        )

        try:
            # Generate forecasts with prediction intervals
            # statsforecast 2.0+ requires df parameter
            forecasts = self._statsforecast.forecast(df=self._training_df, h=max_horizon + 1, level=levels)

            # Extract forecasts and intervals for requested horizons
            result_data = {}

            for horizon in horizons:
                # Point forecast
                result_data[f"h{horizon}"] = forecasts.iloc[horizon, :]["AutoARIMA"]

                # Prediction intervals
                for level in levels:
                    lo_col = f"AutoARIMA-lo-{level}"
                    hi_col = f"AutoARIMA-hi-{level}"

                    if lo_col in forecasts.columns and hi_col in forecasts.columns:
                        result_data[f"h{horizon}_lo{level}"] = forecasts.iloc[horizon, :][lo_col]
                        result_data[f"h{horizon}_hi{level}"] = forecasts.iloc[horizon, :][hi_col]

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
            msg = f"ARIMA prediction with intervals failed: {e}"
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from e

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores.

        ARIMA models don't use features in the traditional sense (they use
        past values only), so this returns an empty dict.

        Returns:
            Empty dictionary (ARIMA doesn't have feature importance).

        Raises:
            ValueError: If model is not fitted.
        """
        self._check_is_fitted()
        logger.debug("ARIMA model has no feature importance (uses past values only)")
        return {}

    def get_diagnostics(self) -> dict[str, Any]:
        """Get model diagnostics and statistics.

        Returns information about the fitted ARIMA model including parameters,
        information criteria, and training data statistics.

        Returns:
            Dictionary with diagnostic information:
                - "model_params": Fitted ARIMA parameters (p,d,q,P,D,Q)
                - "training_stats": Statistics from training data
                - "config": ARIMAConfig used for training

        Raises:
            ValueError: If model is not fitted.

        Example:
            ```python
            diagnostics = model.get_diagnostics()
            print(f"Fitted model: ARIMA{diagnostics['model_params']}")
            print(f"Training mean: {diagnostics['training_stats']['mean']}")
            ```
        """
        self._check_is_fitted()

        diagnostics = {
            "model_name": self.name,
            "model_version": self.version,
            "config": self._arima_config.model_dump(),
            "training_stats": {
                "n_samples": self._training_metadata.get("n_samples"),
                "mean": self._training_metadata.get("data_mean"),
                "std": self._training_metadata.get("data_std"),
                "min": self._training_metadata.get("data_min"),
                "max": self._training_metadata.get("data_max"),
            },
            "trained_at": self._training_metadata.get("trained_at"),
        }

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
            logger.info("Saved ARIMA model to %s", path)
        except Exception as e:
            msg = f"Failed to save model to {path}: {e}"
            raise OSError(msg) from e

    @classmethod
    def load(cls, path: str | Path) -> "ARIMADemandMeanModel":
        """Load model from disk using joblib.

        Args:
            path: File path to load the model from.

        Returns:
            Loaded model instance.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            ValueError: If loaded object is not an ARIMADemandMeanModel instance.
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

        if not isinstance(model, ARIMADemandMeanModel):
            msg = f"Loaded object is not an ARIMADemandMeanModel instance: {type(model)}"
            raise ValueError(msg)

        logger.info("Loaded ARIMA model from %s", path)
        return model
