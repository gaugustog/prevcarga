"""Holt-Winters hierarchical model for load forecasting.

This module implements the HoltWintersModel class that combines Holt-Winters demand mean
forecasting with Holt-Winters profile models in a two-stage hierarchical approach:

1. Stage 1: Holt-Winters models daily demand mean (average load per day)
2. Stage 2: Holt-Winters models semi-hourly profile ratios (load / demand_mean) for each period
3. Combination: Semi-hourly load = demand_mean * profile_ratio

The Holt-Winters methodology uses exponential smoothing to capture trend and
seasonal patterns at both the daily and intraday levels.

Example:
    ```python
    from src.models.hierarchical.holt_winters import HoltWintersModel
    from src.models.config.holtwinters_config import (
        HoltWintersConfig,
        HoltWintersProfileConfig
    )
    import pandas as pd

    # Create model
    model = HoltWintersModel()

    # Prepare configuration
    config = {
        "demand_mean_config": {
            "holtwinters_config": HoltWintersConfig(
                trend="add",
                seasonal="add",
                seasonal_periods=7
            )
        },
        "profile_config": {
            "holtwinters_config": HoltWintersProfileConfig(
                seasonal="add",
                seasonal_periods=7,
                trend=None
            )
        }
    }

    # Train model (two-stage training)
    model.fit(X_train, y_train, config)

    # Generate forecasts for horizons D+0 to D+3
    predictions = model.predict(X_test, horizons=[0, 1, 2, 3])

    # Get seasonal decomposition for analysis
    decomp = model.get_seasonal_decomposition()
    print(decomp["demand_mean"]["level"])     # Daily level component
    print(decomp["demand_mean"]["seasonal"])  # Daily seasonal component
    ```
"""

from typing import Any

import numpy as np
import pandas as pd

from src.models.demand_mean.holt_winters import HoltWintersDemandMeanModel
from src.models.hierarchical.base_hierarchical import BaseHierarchicalModel
from src.models.profile.holt_winters_profile import HoltWintersProfileModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
EPSILON = 1e-8  # Small constant to avoid division by zero
MIN_PROFILE_RATIO = 0.01  # Minimum profile ratio (1%)
MAX_PROFILE_RATIO = 10.0  # Maximum profile ratio (1000%)
PERIODS_PER_DAY = 48  # Semi-hourly periods in a day


class HoltWintersModel(BaseHierarchicalModel):
    """Holt-Winters hierarchical model for load forecasting.

    This model implements a hierarchical forecasting approach using exponential
    smoothing at both levels:
    - Demand mean: Holt-Winters with trend and seasonal components
    - Profiles: Holt-Winters with seasonal components for each period

    Training Process:
        1. Aggregate semi-hourly data to daily demand mean
        2. Train Holt-Winters model on daily demand mean time series
        3. Calculate profile ratios (semi-hourly load / daily demand mean)
        4. Train 48 independent Holt-Winters models (one per semi-hourly period)

    Prediction Process:
        1. Predict daily demand mean using Holt-Winters
        2. Predict profile ratios for each period using Holt-Winters
        3. Combine: load[period] = demand_mean * profile_ratio[period]
        4. Validate energy conservation (daily sum ≈ demand_mean * 48)

    Attributes:
        _epsilon: Small constant to avoid division by zero in ratio calculation.
        _profile_model: HoltWintersProfileModel for all 48 periods.
    """

    def __init__(self) -> None:
        """Initialize Holt-Winters hierarchical model."""
        super().__init__()
        self._epsilon = EPSILON
        self._profile_model: HoltWintersProfileModel | None = None
        logger.debug("Initialized HoltWintersModel")

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series | pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        """Train the Holt-Winters hierarchical model.

        Overrides base class fit() to train a SINGLE HoltWintersProfileModel for all
        48 periods rather than 48 separate models. HoltWintersProfileModel internally
        handles all periods.

        Args:
            X: Training features DataFrame with semi-hourly resolution.
            y: Training target (load) as Series or single-column DataFrame.
            config: Training configuration with "demand_mean_config" and "profile_config".
        """
        logger.info("Starting Holt-Winters two-stage hierarchical model training")

        # Convert y to Series if DataFrame
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                msg = f"y must be single column, got {y.shape[1]} columns"
                raise ValueError(msg)
            y = y.iloc[:, 0]

        # Combine X and y for data preparation
        df = X.copy()
        df["load"] = y

        # Stage 1: Train demand mean model
        logger.info("Stage 1: Training Holt-Winters demand mean model")
        X_daily, y_daily = self._prepare_demand_mean_data(df)

        demand_mean_config = config.get("demand_mean_config", {})
        self.demand_mean_model = self.demand_mean_model_class()
        self.demand_mean_model.fit(X_daily, y_daily, demand_mean_config)

        logger.info(
            "Demand mean model trained on %d daily samples",
            len(y_daily),
        )

        # Get demand mean predictions for profile ratio calculation
        demand_mean_pred = self.demand_mean_model.predict(X_daily)
        if isinstance(demand_mean_pred, pd.DataFrame):
            demand_mean_series = demand_mean_pred.iloc[:, 0]
        else:
            demand_mean_series = demand_mean_pred

        # Stage 2: Train SINGLE HoltWintersProfileModel on ALL profile ratio data
        logger.info("Stage 2: Training Holt-Winters profile model (all 48 periods)")
        profile_config = config.get("profile_config", {})

        # Calculate profile ratios for all semi-hourly data
        profile_ratios = self._calculate_profile_ratios(y, demand_mean_series)

        # Create empty X for Holt-Winters (it uses time patterns, not features)
        X_profile = pd.DataFrame(index=y.index)

        # Train single HoltWintersProfileModel - it handles all 48 periods internally
        self._profile_model = HoltWintersProfileModel()
        self._profile_model.fit(X_profile, profile_ratios, profile_config)

        # Map profile model's internal models to our profile_models dict for compatibility
        self.profile_models = self._profile_model._models

        # Mark as fitted
        self._is_fitted = True
        self._feature_names = []

        # Store training metadata
        self._training_metadata = {
            "n_daily_samples": len(y_daily),
            "n_semi_hourly_samples": len(y),
            "n_profile_models": len(self.profile_models),
            "demand_mean_model": self.demand_mean_model.name,
            "profile_model": self._profile_model.name if self._profile_model else None,
        }

        logger.info(
            "Holt-Winters training complete: demand mean trained on %d daily samples, "
            "%d profile models trained",
            len(y_daily),
            len(self.profile_models),
        )

    def get_feature_importance(self) -> dict[str, float]:
        """Return combined feature importance from demand mean and profile models.

        Note: Holt-Winters models don't provide traditional feature importance.
        They use exponential smoothing on past values.

        Returns:
            Empty dictionary (no feature importance available).
        """
        self._check_is_fitted()

        logger.debug(
            "Feature importance not available for Holt-Winters model "
            "(uses exponential smoothing on past values)"
        )
        return {}

    @property
    def name(self) -> str:
        """Return the model name identifier.

        Returns:
            Model name string.
        """
        return "holt_winters"

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

    @property
    def demand_mean_model_class(self) -> type[HoltWintersDemandMeanModel]:
        """Return the class for demand mean forecasting.

        Returns:
            HoltWintersDemandMeanModel class for Holt-Winters demand mean forecasting.
        """
        return HoltWintersDemandMeanModel

    @property
    def profile_model_class(self) -> type[HoltWintersProfileModel]:
        """Return the class for profile forecasting.

        Returns:
            HoltWintersProfileModel class for Holt-Winters profile ratio forecasting.
        """
        return HoltWintersProfileModel

    def _prepare_demand_mean_data(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare data for demand mean model training.

        Aggregates semi-hourly data to daily resolution and computes the demand
        mean (daily average load).

        Args:
            df: Input DataFrame with semi-hourly data. Must have DatetimeIndex
               and optionally a "load" column (for training).

        Returns:
            Tuple of (X_daily, y_daily) where:
                - X_daily: Features aggregated to daily resolution (empty DataFrame)
                - y_daily: Daily demand mean (average load)

        Raises:
            ValueError: If DatetimeIndex is missing or load column is missing during training.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            msg = "DataFrame must have DatetimeIndex"
            raise ValueError(msg)

        logger.debug("Aggregating semi-hourly data to daily demand mean")

        # Check if this is training (has load column) or prediction
        has_load = "load" in df.columns

        if has_load:
            # Training: aggregate load to daily mean
            daily_y = df["load"].groupby(df.index.date).mean()

            # Convert date index to DatetimeIndex
            daily_y.index = pd.to_datetime(daily_y.index)

            # Create empty features DataFrame with same index
            # Holt-Winters doesn't use features, only past values
            daily_X = pd.DataFrame(index=daily_y.index)  # noqa: N806

            logger.debug(
                "Prepared demand mean data: %d daily samples (from %d semi-hourly)",
                len(daily_y),
                len(df),
            )

            return daily_X, daily_y

        # Prediction: create daily index from semi-hourly index
        # Group by date to get unique dates
        dates = df.index.date
        unique_dates = pd.Series(dates).unique()
        daily_index = pd.to_datetime(unique_dates)

        # Create empty DataFrame and Series
        daily_X = pd.DataFrame(index=daily_index)  # noqa: N806
        daily_y = pd.Series(index=daily_index, dtype=float)

        logger.debug(
            "Prepared demand mean data for prediction: %d daily samples",
            len(daily_X),
        )

        return daily_X, daily_y

    def _prepare_profile_data(
        self,
        df: pd.DataFrame,
        demand_mean: pd.Series,
        period: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare data for profile model training for a specific period.

        Filters data for the given semi-hourly period (0-47) and computes
        profile ratios as: profile_ratio = actual_load / demand_mean

        The returned data is indexed by DATE (not timestamp), so there is
        one row per day for this period.

        Args:
            df: Input DataFrame with semi-hourly data. Must have DatetimeIndex
               and optionally a "load" column (for training).
            demand_mean: Series of daily demand mean values, indexed by date.
            period: Semi-hourly period (0-47) to prepare data for.

        Returns:
            Tuple of (X_period, y_period) where:
                - X_period: Features for the specified period, indexed by date
                - y_period: Profile ratios (load / demand_mean), indexed by date

        Raises:
            ValueError: If period is invalid or DatetimeIndex is missing.
        """
        if not 0 <= period < PERIODS_PER_DAY:
            msg = f"Period must be in range [0, 47], got {period}"
            raise ValueError(msg)

        if not isinstance(df.index, pd.DatetimeIndex):
            msg = "DataFrame must have DatetimeIndex"
            raise ValueError(msg)

        logger.debug("Preparing profile data for period %d", period)

        # Extract semi-hourly period from timestamps
        # Period = hour * 2 + (minute // 30)
        periods = df.index.hour * 2 + (df.index.minute // 30)

        # Filter data for this period
        mask = periods == period
        df_period = df[mask].copy()

        if len(df_period) == 0:
            logger.warning("No data available for period %d", period)
            # Return empty DataFrame and Series with proper DatetimeIndex
            empty_index = pd.DatetimeIndex([])
            return pd.DataFrame(index=empty_index), pd.Series(dtype=float, index=empty_index)

        # Add date column
        df_period = df_period.copy()
        df_period["date"] = df_period.index.date

        # Check if this is training (has load column) or prediction
        has_load = "load" in df.columns

        if has_load:
            # Training: calculate profile ratios for each day
            # Group by date and compute profile ratio
            profile_data = []

            # Calculate hour and minute for this period
            # Period 0 = 00:00, Period 1 = 00:30, Period 2 = 01:00, etc.
            period_hour = period // 2
            period_minute = (period % 2) * 30

            for date in df_period["date"].unique():
                date_mask = df_period["date"] == date
                load_value = (
                    df_period.loc[date_mask, "load"].iloc[0] if date_mask.sum() > 0 else np.nan
                )

                # Convert date to Timestamp for demand_mean lookup
                date_ts = pd.Timestamp(date)

                if date_ts in demand_mean.index:
                    demand_mean_value = demand_mean.loc[date_ts]

                    # Calculate profile ratio
                    profile_ratio = load_value / (demand_mean_value + self._epsilon)
                    profile_ratio = np.clip(profile_ratio, MIN_PROFILE_RATIO, MAX_PROFILE_RATIO)

                    # Create timestamp with correct time-of-day for this period
                    timestamp_with_time = date_ts.replace(hour=period_hour, minute=period_minute)

                    profile_data.append(
                        {
                            "timestamp": timestamp_with_time,
                            "profile_ratio": profile_ratio,
                        }
                    )

            if len(profile_data) == 0:
                logger.warning("No valid profile data for period %d", period)
                empty_index = pd.DatetimeIndex([])
                return pd.DataFrame(index=empty_index), pd.Series(dtype=float, index=empty_index)

            # Create DataFrame from profile data
            profile_df = pd.DataFrame(profile_data)
            profile_df = profile_df.set_index("timestamp")  # Index by timestamp, not date

            # Create empty features (Holt-Winters uses time patterns)
            X_period = pd.DataFrame(index=profile_df.index)  # noqa: N806
            y_period = profile_df["profile_ratio"]

            logger.debug(
                "Prepared profile data for period %d: %d samples, mean ratio=%.3f, std=%.3f",
                period,
                len(y_period),
                y_period.mean() if len(y_period) > 0 else 0,
                y_period.std() if len(y_period) > 0 else 0,
            )

            return X_period, y_period

        # Prediction: create timestamps with correct time-of-day for this period
        # Period 0 = 00:00, Period 1 = 00:30, Period 2 = 01:00, etc.
        period_hour = period // 2
        period_minute = (period % 2) * 30

        # Get unique dates and create timestamps with correct time
        unique_dates = sorted(set(df_period.index.date))
        timestamps = []
        for date in unique_dates:
            date_ts = pd.Timestamp(date)
            timestamp_with_time = date_ts.replace(hour=period_hour, minute=period_minute)
            timestamps.append(timestamp_with_time)

        timestamp_index = pd.DatetimeIndex(timestamps)

        X_period = pd.DataFrame(index=timestamp_index)  # noqa: N806
        y_period = pd.Series(index=timestamp_index, dtype=float)

        logger.debug(
            "Prepared profile data for prediction (period %d): %d samples",
            period,
            len(X_period),
        )

        return X_period, y_period

    def _calculate_profile_ratios(
        self,
        y: pd.Series,
        demand_mean_daily: pd.Series,
    ) -> pd.Series:
        """Calculate profile ratios (load / demand_mean) for semi-hourly data.

        Maps each semi-hourly timestamp to its corresponding daily demand mean
        and computes the ratio. Ratios are clipped to reasonable bounds to
        avoid extreme values.

        Args:
            y: Semi-hourly load values with DatetimeIndex.
            demand_mean_daily: Daily demand mean values indexed by date.

        Returns:
            Series of profile ratios with same index as y.

        Raises:
            ValueError: If indices don't align or demand_mean contains invalid values.
        """
        logger.debug("Calculating profile ratios for %d semi-hourly samples", len(y))

        # Map each timestamp to its date
        dates = y.index.date

        # Expand daily demand mean to semi-hourly by mapping dates
        demand_mean_expanded = pd.Series(index=y.index, dtype=float)

        for i, date in enumerate(dates):
            try:
                # Convert date to Timestamp for lookup
                date_ts = pd.Timestamp(date)
                if date_ts in demand_mean_daily.index:
                    demand_mean_expanded.iloc[i] = demand_mean_daily.loc[date_ts]
                else:
                    # Date not found in demand_mean_daily
                    logger.warning(
                        "Date %s not found in demand_mean_daily, using mean of available values",
                        date,
                    )
                    demand_mean_expanded.iloc[i] = demand_mean_daily.mean()
            except Exception as e:
                logger.error("Error mapping date %s: %s", date, e)
                demand_mean_expanded.iloc[i] = demand_mean_daily.mean()

        # Calculate ratios with epsilon protection
        profile_ratios = y / (demand_mean_expanded + self._epsilon)

        # Clip to reasonable bounds
        profile_ratios = profile_ratios.clip(MIN_PROFILE_RATIO, MAX_PROFILE_RATIO)

        # Count clipped values
        n_clipped_low = (y / (demand_mean_expanded + self._epsilon) < MIN_PROFILE_RATIO).sum()
        n_clipped_high = (y / (demand_mean_expanded + self._epsilon) > MAX_PROFILE_RATIO).sum()

        if n_clipped_low > 0:
            logger.debug(
                "Clipped %d profile ratios below %.2f",
                n_clipped_low,
                MIN_PROFILE_RATIO,
            )

        if n_clipped_high > 0:
            logger.debug(
                "Clipped %d profile ratios above %.2f",
                n_clipped_high,
                MAX_PROFILE_RATIO,
            )

        logger.debug(
            "Profile ratios calculated: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
            profile_ratios.mean(),
            profile_ratios.std(),
            profile_ratios.min(),
            profile_ratios.max(),
        )

        return profile_ratios

    def get_seasonal_decomposition(self) -> dict[str, Any]:
        """Get comprehensive seasonal decomposition from both model levels.

        Extracts decomposition from the demand mean model (level, trend, seasonal)
        and optionally from profile models.

        Returns:
            Dictionary containing:
                - "demand_mean": Decomposition from demand mean model with keys:
                    - "level": Level component
                    - "trend": Trend component (if present)
                    - "seasonal": Seasonal component (if present)
                    - "fitted_values": In-sample fitted values
                    - "residuals": In-sample residuals

        Raises:
            ValueError: If model is not fitted.
        """
        self._check_is_fitted()

        logger.debug("Extracting seasonal decomposition")

        decomposition = {}

        # Get demand mean decomposition
        if self.demand_mean_model is not None:
            decomposition["demand_mean"] = self.demand_mean_model.get_seasonal_decomposition()
            logger.info("Extracted demand mean decomposition")

        # Optionally add profile decompositions (first few periods as samples)
        if len(self.profile_models) > 0:
            decomposition["profile_samples"] = {}
            sample_periods = sorted(self.profile_models.keys())[:3]  # First 3 periods

            for period in sample_periods:
                if period in self._profile_model._models:
                    fitted_model = self._profile_model._models[period]

                    # Extract components
                    period_decomp = {
                        "level": pd.Series(fitted_model.level) if hasattr(fitted_model, "level") else None,
                        "trend": pd.Series(fitted_model.trend) if hasattr(fitted_model, "trend") else None,
                        "seasonal": pd.Series(fitted_model.season) if hasattr(fitted_model, "season") else None,
                        "fitted_values": pd.Series(fitted_model.fittedvalues),
                        "residuals": pd.Series(fitted_model.resid),
                    }
                    decomposition["profile_samples"][period] = period_decomp

            logger.info("Extracted %d profile sample decompositions", len(sample_periods))

        return decomposition

    def __repr__(self) -> str:
        """Return string representation of the Holt-Winters model.

        Returns:
            String representation with model info and training status.
        """
        return (
            f"HoltWintersModel("
            f"name='{self.name}', "
            f"version='{self.version}', "
            f"fitted={self._is_fitted}, "
            f"n_profile_models={len(self.profile_models)}, "
            f"horizons={self.supported_horizons})"
        )
