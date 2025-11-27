"""RegDin+SVM hierarchical model for load forecasting.

This module implements the RegDinSVMModel class that combines ARIMA demand mean
forecasting with SVM profile models in a two-stage hierarchical approach:

1. Stage 1: ARIMA models daily demand mean (average load per day)
2. Stage 2: SVM models semi-hourly profile ratios (load / demand_mean) for each period
3. Combination: Semi-hourly load = demand_mean * profile_ratio

The RegDin methodology (Regression of Dynamic components) decomposes the load
forecasting problem into dynamic demand patterns (captured by ARIMA) and static
intraday profiles (captured by SVM).

Example:
    ```python
    from src.models.hierarchical.regdin_svm import RegDinSVMModel
    from src.models.config.arima_config import ARIMAConfig
    from src.models.config.svm_config import SVMProfileConfig
    import pandas as pd

    # Create model
    model = RegDinSVMModel()

    # Prepare configuration
    config = {
        "demand_mean_config": {
            "arima_config": ARIMAConfig(seasonal=True, season_length=7)
        },
        "profile_config": {
            "svm_config": SVMProfileConfig(kernel="rbf", optimize_hyperparams=True)
        }
    }

    # Train model (two-stage training)
    model.fit(X_train, y_train, config)

    # Generate forecasts for horizons D+0 to D+3
    predictions = model.predict(X_test, horizons=[0, 1, 2, 3])

    # Get decomposition for analysis
    decomp = model.get_forecast_decomposition(X_test)
    print(decomp["demand_mean"])     # Daily demand mean
    print(decomp["profiles"][0])     # Profile ratios for period 0
    print(decomp["combined_load"])   # Final semi-hourly load

    # Get diagnostics
    diagnostics = model.get_model_diagnostics()
    print(diagnostics["energy_conservation"])
    ```
"""

from typing import Any

import numpy as np
import pandas as pd

from src.models.demand_mean.arima_model import ARIMADemandMeanModel
from src.models.hierarchical.base_hierarchical import BaseHierarchicalModel
from src.models.profile.svm_profile import SVMProfileModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
EPSILON = 1e-8  # Small constant to avoid division by zero
MIN_PROFILE_RATIO = 0.01  # Minimum profile ratio (1%)
MAX_PROFILE_RATIO = 10.0  # Maximum profile ratio (1000%)
PERIODS_PER_DAY = 48  # Semi-hourly periods in a day
DEFAULT_ENERGY_TOLERANCE = 0.05  # 5% tolerance for energy conservation


class RegDinSVMModel(BaseHierarchicalModel):
    """RegDin+SVM hierarchical model for load forecasting.

    This model implements the RegDin (Regression of Dynamic components) methodology
    by combining ARIMA for dynamic demand mean forecasting with SVM for static
    profile ratio forecasting.

    Training Process:
        1. Aggregate semi-hourly data to daily demand mean
        2. Train ARIMA model on daily demand mean time series
        3. Calculate profile ratios (semi-hourly load / daily demand mean)
        4. Train 48 independent SVM models (one per semi-hourly period)

    Prediction Process:
        1. Predict daily demand mean using ARIMA
        2. Predict profile ratios for each period using SVM
        3. Combine: load[period] = demand_mean * profile_ratio[period]
        4. Validate energy conservation (daily sum ≈ demand_mean * 48)

    Attributes:
        _epsilon: Small constant to avoid division by zero in ratio calculation.
        _energy_tolerance: Tolerance for energy conservation validation.
    """

    def __init__(self) -> None:
        """Initialize RegDin+SVM hierarchical model."""
        super().__init__()
        self._epsilon = EPSILON
        self._energy_tolerance = DEFAULT_ENERGY_TOLERANCE
        self._profile_model: SVMProfileModel | None = None  # Single SVM for all periods
        logger.debug("Initialized RegDinSVMModel")

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series | pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        """Train the RegDin+SVM hierarchical model.

        Overrides base class fit() to train a SINGLE SVMProfileModel for all 48 periods
        rather than 48 separate models. SVMProfileModel internally handles all periods.

        Args:
            X: Training features DataFrame with semi-hourly resolution.
            y: Training target (load) as Series or single-column DataFrame.
            config: Training configuration with "demand_mean_config" and "profile_config".
        """
        logger.info("Starting RegDin+SVM two-stage hierarchical model training")

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
        logger.info("Stage 1: Training ARIMA demand mean model")
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

        # Stage 2: Train SINGLE SVMProfileModel on ALL profile ratio data
        logger.info("Stage 2: Training SVM profile model (all 48 periods)")
        profile_config = config.get("profile_config", {})

        # Calculate profile ratios for all semi-hourly data
        profile_ratios = self._calculate_profile_ratios(y, demand_mean_series)

        # Create empty X for SVM (it will generate temporal features)
        X_profile = pd.DataFrame(index=y.index)

        # Train single SVMProfileModel - it handles all 48 periods internally
        self._profile_model = SVMProfileModel()
        self._profile_model.fit(X_profile, profile_ratios, profile_config)

        # Map SVM's internal models to our profile_models dict for compatibility
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
            "RegDin+SVM training complete: demand mean trained on %d daily samples, "
            "%d profile models trained",
            len(y_daily),
            len(self.profile_models),
        )

    def get_feature_importance(self) -> dict[str, float]:
        """Return combined feature importance from demand mean and profile models.

        Note: Both ARIMA and SVM models don't provide traditional feature importance.
        ARIMA uses past values, and SVM with RBF kernel doesn't have feature importance.

        Returns:
            Empty dictionary (no feature importance available).
        """
        self._check_is_fitted()

        # ARIMA doesn't provide feature importance
        # SVM with RBF kernel doesn't provide feature importance
        logger.debug(
            "Feature importance not available for RegDin+SVM model "
            "(ARIMA uses past values, SVM with RBF doesn't have feature importance)"
        )
        return {}

    @property
    def name(self) -> str:
        """Return the model name identifier.

        Returns:
            Model name string.
        """
        return "regdin_svm"

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
    def demand_mean_model_class(self) -> type[ARIMADemandMeanModel]:
        """Return the class for demand mean forecasting.

        Returns:
            ARIMADemandMeanModel class for ARIMA-based demand mean forecasting.
        """
        return ARIMADemandMeanModel

    @property
    def profile_model_class(self) -> type[SVMProfileModel]:
        """Return the class for profile forecasting.

        Returns:
            SVMProfileModel class for SVM-based profile ratio forecasting.
        """
        return SVMProfileModel

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
            # ARIMA doesn't use features, only past values
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

            # Create empty features (SVM will generate temporal features)
            X_period = pd.DataFrame(index=profile_df.index)  # noqa: N806
            y_period = profile_df["profile_ratio"]

            logger.debug(
                "Prepared profile data for period %d: %d samples, " "mean ratio=%.3f, std=%.3f",
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

    def get_decomposition(
        self,
        X: pd.DataFrame,  # noqa: N803
    ) -> dict[str, Any]:
        """Get decomposition of predictions into demand mean and profiles.

        This override handles the ARIMA demand mean model which produces
        forecasts for horizons rather than per-date predictions. The method
        computes forecasts for as many horizons as there are unique dates
        in the input.

        Args:
            X: Features DataFrame with semi-hourly resolution.

        Returns:
            Dictionary containing:
                - "demand_mean": Series of demand mean predictions (one per day)
                - "profiles": Dictionary of profile predictions by period (0-47)
                - "combined_load": DataFrame of final semi-hourly predictions

        Raises:
            ValueError: If model is not fitted.
        """
        self._check_is_fitted()

        logger.debug("Computing decomposition for %d samples", len(X))

        # Prepare data - get unique dates
        df = X.copy()
        unique_dates = sorted(set(X.index.date))
        n_days = len(unique_dates)

        logger.debug("Input contains %d unique dates", n_days)

        # Predict demand mean using ARIMA for horizons 0 through n_days-1
        # ARIMA produces forecasts, not historical predictions
        horizons_needed = list(range(min(n_days, len(self.supported_horizons))))
        if len(horizons_needed) == 0:
            horizons_needed = [0]

        demand_mean_pred = self.demand_mean_model.predict(X, horizons=horizons_needed)

        # Convert ARIMA output (single row with h0, h1, ... columns) to Series
        # indexed by date
        demand_mean_values = []
        for h in horizons_needed:
            col_name = f"h{h}"
            if col_name in demand_mean_pred.columns:
                demand_mean_values.append(demand_mean_pred[col_name].iloc[0])
            else:
                # Fallback to first available value
                demand_mean_values.append(demand_mean_pred.iloc[0, 0])

        # Create Series indexed by dates
        date_index = pd.DatetimeIndex([pd.Timestamp(d) for d in unique_dates[:len(demand_mean_values)]])
        demand_mean = pd.Series(demand_mean_values, index=date_index)

        logger.debug(
            "Demand mean predictions: %d values, mean=%.2f",
            len(demand_mean),
            demand_mean.mean() if len(demand_mean) > 0 else 0,
        )

        # Predict profiles for each period
        profiles = {}
        for period in range(PERIODS_PER_DAY):
            if period not in self.profile_models:
                continue

            try:
                X_period, _ = self._prepare_profile_data(df, demand_mean, period)  # noqa: N806
                profile_pred = self._profile_model.predict(X_period)

                if isinstance(profile_pred, pd.DataFrame):
                    profile_values = profile_pred.iloc[:, 0]
                else:
                    profile_values = profile_pred

                # Re-index profiles to match demand_mean index (dates only)
                # Profile predictions come with timestamps, but we need date alignment
                profile_series = pd.Series(
                    profile_values.values[:len(demand_mean)],
                    index=demand_mean.index,
                )
                profiles[period] = profile_series

            except Exception as e:
                logger.warning(
                    "Failed to predict profile for period %d: %s",
                    period,
                    e,
                )
                continue

        # Combine demand mean and profiles
        combined_load = self.combiner.combine(demand_mean, profiles)

        decomposition = {
            "demand_mean": demand_mean,
            "profiles": profiles,
            "combined_load": combined_load,
        }

        logger.info(
            "Decomposition computed: demand_mean (%d samples), "
            "profiles (%d periods), combined_load (%dx%d)",
            len(demand_mean),
            len(profiles),
            combined_load.shape[0] if len(combined_load) > 0 else 0,
            combined_load.shape[1] if len(combined_load) > 0 else 0,
        )

        return decomposition

    def validate_energy_conservation(
        self,
        demand_mean: pd.Series,
        combined_load: pd.DataFrame,
        tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Validate that daily energy is conserved in combined predictions.

        The expected daily energy is demand_mean * 48 (48 semi-hourly periods).
        The actual daily energy is the sum of all 48 semi-hourly loads.
        This method checks that the relative error is within tolerance.

        Args:
            demand_mean: Series of demand mean predictions (daily average load).
            combined_load: DataFrame with semi-hourly load predictions (48 columns).
            tolerance: Maximum allowed relative error (default: 5% = 0.05).

        Returns:
            Dictionary with validation results:
                - "is_valid": bool, whether all errors are within tolerance
                - "mean_error": float, mean relative error across samples
                - "max_error": float, maximum relative error
                - "n_violations": int, number of samples exceeding tolerance
                - "tolerance": float, the tolerance threshold used

        Raises:
            ValueError: If model is not fitted.
        """
        self._check_is_fitted()

        if tolerance is None:
            tolerance = self._energy_tolerance

        logger.debug(
            "Validating energy conservation with tolerance %.1f%%",
            tolerance * 100,
        )

        # Use ProfileCombiner's validation method
        validation_result = self.combiner.validate_energy_conservation(
            demand_mean,
            combined_load,
            tolerance=tolerance,
        )

        if not validation_result["is_valid"]:
            logger.warning(
                "Energy conservation validation failed: %d/%d samples exceed "
                "%.1f%% tolerance (mean error: %.2f%%, max error: %.2f%%)",
                validation_result["n_violations"],
                len(demand_mean),
                tolerance * 100,
                validation_result["mean_error"] * 100,
                validation_result["max_error"] * 100,
            )
        else:
            logger.info(
                "Energy conservation validated: all samples within %.1f%% tolerance "
                "(mean error: %.2f%%, max error: %.2f%%)",
                tolerance * 100,
                validation_result["mean_error"] * 100,
                validation_result["max_error"] * 100,
            )

        return validation_result

    def get_forecast_decomposition(
        self,
        X: pd.DataFrame,  # noqa: N803
        horizons: list[int] | None = None,
    ) -> dict[str, Any]:
        """Get decomposition of forecasts into demand mean and profile components.

        This method returns the intermediate components of the hierarchical
        prediction, useful for debugging, analysis, and understanding model behavior.

        Args:
            X: Features DataFrame with semi-hourly resolution.
            horizons: List of forecast horizons. If None, uses first supported horizon.

        Returns:
            Dictionary containing:
                - "demand_mean": Series of demand mean predictions (daily average)
                - "profiles": Dictionary of profile predictions by period (0-47)
                - "combined_load": DataFrame of final semi-hourly predictions
                - "horizons": List of horizons used

        Raises:
            ValueError: If model is not fitted or input is invalid.
        """
        self._check_is_fitted()

        if horizons is None:
            horizons = [self.supported_horizons[0]]

        logger.debug(
            "Computing forecast decomposition for %d samples, horizons: %s",
            len(X),
            horizons,
        )

        # Use the base class get_decomposition method
        decomposition = self.get_decomposition(X)

        # Add horizons information
        decomposition["horizons"] = horizons

        logger.info(
            "Forecast decomposition computed: demand_mean (%d samples), "
            "profiles (%d periods), combined_load (%d x %d)",
            len(decomposition["demand_mean"]),
            len(decomposition["profiles"]),
            decomposition["combined_load"].shape[0],
            decomposition["combined_load"].shape[1],
        )

        return decomposition

    def get_model_diagnostics(self) -> dict[str, Any]:
        """Get comprehensive model diagnostics and statistics.

        Returns detailed information about both the demand mean (ARIMA) and
        profile (SVM) models, including training statistics, configuration,
        and performance metrics.

        Returns:
            Dictionary with diagnostic information:
                - "model_name": Name of the model ("regdin_svm")
                - "model_version": Version string
                - "supported_horizons": List of supported horizons
                - "demand_mean_model": Diagnostics from ARIMA demand mean model
                - "profile_models": Diagnostics from SVM profile models
                - "training_metadata": Training statistics and configuration
                - "n_profile_models": Number of trained profile models
                - "energy_tolerance": Tolerance for energy conservation

        Raises:
            ValueError: If model is not fitted.
        """
        self._check_is_fitted()

        logger.debug("Retrieving comprehensive model diagnostics")

        diagnostics: dict[str, Any] = {
            "model_name": self.name,
            "model_version": self.version,
            "supported_horizons": self.supported_horizons,
            "energy_tolerance": self._energy_tolerance,
            "n_profile_models": len(self.profile_models),
            "trained_profile_periods": sorted(self.profile_models.keys()),
            "training_metadata": self._training_metadata,
        }

        # Get demand mean model diagnostics
        try:
            if self.demand_mean_model is not None:
                diagnostics["demand_mean_model"] = self.demand_mean_model.get_diagnostics()
            else:
                diagnostics["demand_mean_model"] = None
        except Exception as e:
            logger.warning("Could not retrieve demand mean model diagnostics: %s", e)
            diagnostics["demand_mean_model"] = {"error": str(e)}

        # Get profile model diagnostics (summary)
        profile_diagnostics = {
            "n_models": len(self.profile_models),
            "periods_trained": sorted(self.profile_models.keys()),
        }

        # Get detailed diagnostics from first few profile models as samples
        sample_periods = sorted(self.profile_models.keys())[:3]  # First 3 periods
        profile_diagnostics["sample_models"] = {}

        for period in sample_periods:
            try:
                profile_model = self.profile_models[period]
                profile_diagnostics["sample_models"][period] = profile_model.get_diagnostics()
            except Exception as e:
                logger.warning(
                    "Could not retrieve diagnostics for profile model %d: %s",
                    period,
                    e,
                )
                profile_diagnostics["sample_models"][period] = {"error": str(e)}

        diagnostics["profile_models"] = profile_diagnostics

        logger.info(
            "Retrieved diagnostics: %d profile models, horizons %s",
            len(self.profile_models),
            self.supported_horizons,
        )

        return diagnostics

    def __repr__(self) -> str:
        """Return string representation of the RegDin+SVM model.

        Returns:
            String representation with model info and training status.
        """
        return (
            f"RegDinSVMModel("
            f"name='{self.name}', "
            f"version='{self.version}', "
            f"fitted={self._is_fitted}, "
            f"n_profile_models={len(self.profile_models)}, "
            f"horizons={self.supported_horizons})"
        )
