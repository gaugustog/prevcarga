"""BLF (Baseline Load Forecast) predictor for intraday corrections.

This module provides the BLFPredictor class that wraps a base forecasting model
and applies real-time corrections based on recent forecast errors. It supports:
- Updating predictions using real-time observations
- Period-specific correction factors with exponential decay weighting
- Statistical error modeling (Ridge/Lasso regression)
- Confidence bound adjustments based on recent accuracy
- State persistence for operational deployment
- Brazilian SIN semi-hourly (48 periods/day) support
- Timezone-aware timestamp handling (America/Sao_Paulo)
- Multiplicative and additive correction modes

Example:
    ```python
    from src.models.end_to_end.blf_predictor import BLFPredictor
    from src.models.end_to_end.lgbm_model import LGBMModel
    import pandas as pd

    # Train base model
    base_model = LGBMModel()
    base_model.fit(X_train, y_train, config)

    # Create BLF predictor for Brazilian SIN
    blf = BLFPredictor(base_model, config={
        "correction_window_hours": 6,
        "periods_per_day": 48,  # Semi-hourly
        "timezone": "America/Sao_Paulo",
        "subsystem": "SECO",
        "correction_mode": "multiplicative"
    })

    # Update with real-time observations
    observations = pd.DataFrame({
        "timestamp": [...],
        "load": [...],
        "base_prediction": [...]  # Optional: predictions at observation time
    })
    stats = blf.update_with_observations(observations)

    # Get corrected predictions
    corrected_preds = blf.predict_corrected(X_test, horizons=[0, 1])

    # Save/load state
    blf.save_state("blf_state.pkl")
    blf_loaded = BLFPredictor.load_state("blf_state.pkl", base_model)
    ```
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso, Ridge

try:
    import pytz

    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False

from src.models.base.model import BaseModel
from src.models.config.blf_config import BLFConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Constants for error model training
MIN_HISTORY_FOR_ERROR_MODEL = 10

# Default horizon scaling factors for confidence bounds (empirically calibrated)
DEFAULT_HORIZON_FACTORS = {
    0: 1.0,  # D+0 - baseline
    1: 1.5,  # D+1 - significant increase
    2: 1.8,  # D+2
    3: 2.0,  # D+3
    4: 2.1,  # D+4
    5: 2.2,  # D+5
    6: 2.2,  # D+6
    7: 2.2,  # D+7
    8: 2.2,  # D+8 - plateau
}


class BLFPredictor:
    """BLF predictor for intraday forecast corrections.

    This class wraps a base forecasting model and applies corrections based on
    recent forecast errors observed during the day. It calculates period-specific
    correction factors using exponential decay weighting and optionally trains
    a statistical error model to predict future errors.

    Features:
    - Semi-hourly (48 periods/day) support for Brazilian SIN grid
    - Timezone-aware timestamp handling (America/Sao_Paulo)
    - Multiplicative and additive correction modes
    - Subsystem-aware corrections (SECO, S, NE, N)
    - Horizon-specific decay factors

    The predictor maintains a correction history DataFrame with columns:
    - timestamp: Observation timestamp (normalized to configured timezone)
    - observed: Observed load value
    - predicted: Base model prediction at observation time
    - error: Forecast error (observed - predicted)
    - error_ratio: Ratio error for multiplicative corrections (observed/predicted - 1)
    - period: Period of day (0 to periods_per_day-1)
    - day_of_week: Day of week (0-6)
    - weight: Exponential decay weight for this observation
    - subsystem: Subsystem identifier (if configured)

    Attributes:
        base_model: Fitted BaseModel instance used for base predictions.
        config: BLFConfig instance with correction parameters.
        correction_history: DataFrame of past observations and errors.
        error_model: Trained Ridge/Lasso model for error prediction (if enabled).
        last_update: Timestamp of last observation update.
    """

    def __init__(
        self,
        base_model: BaseModel,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize BLF predictor with a base model.

        Args:
            base_model: Fitted BaseModel instance. Must be fitted before use.
            config: Configuration dictionary matching BLFConfig schema.
                   If None, uses default BLFConfig values.

        Raises:
            ValueError: If base_model is not fitted.

        Example:
            ```python
            base_model = LGBMModel()
            base_model.fit(X_train, y_train, config)
            # For Brazilian SIN with semi-hourly data
            blf = BLFPredictor(base_model, config={
                "periods_per_day": 48,
                "timezone": "America/Sao_Paulo",
                "correction_mode": "multiplicative"
            })
            ```
        """
        if not base_model.is_fitted():
            msg = "Base model must be fitted before creating BLFPredictor"
            raise ValueError(msg)

        self.base_model = base_model

        # Parse configuration
        if config is None:
            config = {}
        try:
            self.config = BLFConfig(**config)
        except Exception as e:
            msg = f"Invalid BLF configuration: {e}"
            raise ValueError(msg) from e

        # Initialize correction history with period-based columns
        self.correction_history = pd.DataFrame(
            columns=[
                "timestamp",
                "observed",
                "predicted",
                "error",
                "error_ratio",
                "period",
                "day_of_week",
                "weight",
                "subsystem",
            ]
        )

        # Initialize error model
        self.error_model: Ridge | Lasso | None = None

        # Track last update time
        self.last_update: datetime | None = None

        # Initialize timezone for timestamp normalization
        self._tz = None
        if HAS_PYTZ and self.config.timezone:
            try:
                self._tz = pytz.timezone(self.config.timezone)
            except pytz.UnknownTimeZoneError:
                logger.warning(
                    "Unknown timezone '%s', timestamps will not be normalized",
                    self.config.timezone,
                )

        logger.info(
            "Initialized BLFPredictor with base model: %s v%s, "
            "periods_per_day=%d, timezone=%s, correction_mode=%s, subsystem=%s",
            base_model.name,
            base_model.version,
            self.config.periods_per_day,
            self.config.timezone,
            self.config.correction_mode,
            self.config.subsystem,
        )

    def update_with_observations(
        self,
        observations: pd.DataFrame,
    ) -> dict[str, Any]:
        """Update correction history with new observations.

        This method:
        1. Validates observation data
        2. Calculates errors vs base predictions (if available) or stores for later
        3. Updates correction history with decay weights
        4. Trims history to max_history_days
        5. Optionally trains error model

        Args:
            observations: DataFrame with columns:
                - timestamp: pd.Timestamp or datetime
                - load: Observed load values
                - base_prediction: (Optional) Base model predictions at observation time.
                  If not provided, predictor will use stored predictions.

        Returns:
            Dictionary with update statistics:
                - n_observations: Number of observations added
                - n_total_history: Total observations in history
                - mean_abs_error: Mean absolute error of new observations
                - correction_applied: Whether corrections are active
                - error_model_trained: Whether error model was (re)trained

        Raises:
            ValueError: If observations DataFrame is invalid or missing required columns.

        Example:
            ```python
            obs = pd.DataFrame({
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "load": [100, 105, 110, ...],
                "base_prediction": [98, 103, 108, ...]
            })
            stats = blf.update_with_observations(obs)
            print(f"Updated with {stats['n_observations']} observations")
            ```
        """
        # Validate observations
        if observations.empty:
            logger.warning("Empty observations provided, no update performed")
            return {
                "n_observations": 0,
                "n_total_history": len(self.correction_history),
                "mean_abs_error": None,
                "correction_applied": False,
                "error_model_trained": False,
            }

        required_cols = {"timestamp", "load"}
        missing_cols = required_cols - set(observations.columns)
        if missing_cols:
            msg = f"Observations missing required columns: {missing_cols}"
            raise ValueError(msg)

        # Make a copy to avoid modifying input
        obs = observations.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(obs["timestamp"]):
            try:
                obs["timestamp"] = pd.to_datetime(obs["timestamp"], format="ISO8601")
            except ValueError:
                # Fall back to mixed format parsing if ISO8601 fails
                try:
                    obs["timestamp"] = pd.to_datetime(obs["timestamp"], format="mixed")
                except Exception as e:
                    msg = f"Failed to parse timestamp column: {e}"
                    raise ValueError(msg) from e

        # Calculate errors
        errors_df = self._calculate_recent_errors(obs)

        if errors_df.empty:
            logger.warning("No valid errors calculated from observations")
            return {
                "n_observations": 0,
                "n_total_history": len(self.correction_history),
                "mean_abs_error": None,
                "correction_applied": False,
                "error_model_trained": False,
            }

        # Add to correction history
        # Filter out empty/all-NA columns before concat to avoid FutureWarning
        if self.correction_history.empty:
            self.correction_history = errors_df.copy()
        else:
            self.correction_history = pd.concat(
                [self.correction_history, errors_df],
                ignore_index=True,
            )

        # Sort by timestamp
        self.correction_history = self.correction_history.sort_values("timestamp")

        # Trim to max history (based on most recent timestamp in history)
        if not self.correction_history.empty:
            most_recent = self.correction_history["timestamp"].max()
            cutoff_time = most_recent - timedelta(days=self.config.max_history_days)
            self.correction_history = self.correction_history[
                self.correction_history["timestamp"] >= cutoff_time
            ].reset_index(drop=True)

        # Update last update time
        self.last_update = datetime.now(UTC)

        # Optionally train error model
        error_model_trained = False
        if (
            self.config.use_error_model
            and len(self.correction_history) >= self.config.min_observations
        ):
            try:
                self._train_error_model()
                error_model_trained = True
            except Exception as e:
                logger.warning("Failed to train error model: %s", e)

        # Calculate statistics
        mean_abs_error = errors_df["error"].abs().mean()
        correction_applied = len(self.correction_history) >= self.config.min_observations

        logger.info(
            "Updated BLF with %d observations. Total history: %d. MAE: %.2f",
            len(errors_df),
            len(self.correction_history),
            mean_abs_error,
        )

        return {
            "n_observations": len(errors_df),
            "n_total_history": len(self.correction_history),
            "mean_abs_error": mean_abs_error,
            "correction_applied": correction_applied,
            "error_model_trained": error_model_trained,
        }

    def predict_corrected(
        self,
        X: pd.DataFrame,  # noqa: N803 - X is conventional for feature matrix
        horizons: list[int],
    ) -> pd.DataFrame:
        """Generate corrected predictions for specified horizons.

        This method:
        1. Gets base model predictions
        2. Applies period-specific correction factors
        3. Clips corrections to max_correction_pct
        4. Ensures non-negative predictions
        5. Adjusts confidence bounds based on recent accuracy

        Supports both additive and multiplicative correction modes.

        Args:
            X: Features DataFrame with shape (n_samples, n_features).
            horizons: List of forecast horizons to predict.

        Returns:
            DataFrame with corrected predictions and confidence bounds.
            Columns: h{n}, h{n}_lower, h{n}_upper for each horizon.

        Raises:
            ValueError: If base model prediction fails.

        Example:
            ```python
            # Get corrected predictions for D+0 and D+1
            corrected = blf.predict_corrected(X_test, horizons=[0, 1])
            # Returns columns: h0, h0_lower, h0_upper, h1, h1_lower, h1_upper
            ```
        """
        # Get base predictions
        try:
            base_preds = self.base_model.predict(X, horizons=horizons)
        except Exception as e:
            msg = f"Base model prediction failed: {e}"
            raise ValueError(msg) from e

        # If insufficient history, return base predictions with default bounds
        if len(self.correction_history) < self.config.min_observations:
            logger.debug(
                "Insufficient correction history (%d < %d), returning base predictions",
                len(self.correction_history),
                self.config.min_observations,
            )
            return self._add_default_confidence_bounds(base_preds, horizons)

        # Calculate correction factors by period
        correction_factors = self._calculate_correction_factors()

        # Apply corrections
        corrected_preds = pd.DataFrame(index=X.index)

        for horizon in horizons:
            base_col = f"h{horizon}"
            if base_col not in base_preds.columns:
                logger.warning("Horizon %d not in base predictions, skipping", horizon)
                continue

            base_values = base_preds[base_col].to_numpy()

            # Extract period from index or features
            periods = self._extract_periods(X)

            # Apply period-specific corrections
            corrected_values = base_values.copy()
            for i, (period, base_val) in enumerate(zip(periods, base_values, strict=False)):
                correction = correction_factors.get(period, 0.0)

                if self.config.correction_mode == "multiplicative":
                    # Multiplicative mode: correction is a ratio
                    # corrected = base * (1 + correction_ratio)
                    # Clip correction ratio to max_correction_pct
                    max_ratio = self.config.max_correction_pct / 100.0
                    correction = np.clip(correction, -max_ratio, max_ratio)
                    corrected_values[i] = base_val * (1.0 + correction)
                else:
                    # Additive mode: correction is in load units
                    # Clip correction to max_correction_pct of base value
                    max_correction = base_val * (self.config.max_correction_pct / 100.0)
                    correction = np.clip(correction, -max_correction, max_correction)
                    corrected_values[i] = base_val + correction

            # Ensure non-negative
            corrected_values = np.maximum(corrected_values, 0.0)

            corrected_preds[base_col] = corrected_values

        # Add confidence bounds
        corrected_preds = self._adjust_confidence_bounds(corrected_preds, horizons)

        logger.debug(
            "Generated %d corrected predictions for horizons %s (mode=%s)",
            len(corrected_preds),
            horizons,
            self.config.correction_mode,
        )

        return corrected_preds

    def _normalize_timestamp(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        """Normalize timestamp to configured timezone.

        Args:
            timestamp: Input timestamp (may be timezone-aware or naive).

        Returns:
            Timestamp normalized to configured timezone, or unchanged if
            no timezone configured.
        """
        if self._tz is None:
            return timestamp

        # If timestamp is naive, assume it's already in local time
        if timestamp.tzinfo is None:
            return timestamp

        # Convert to configured timezone
        try:
            if hasattr(timestamp, "tz_convert"):
                return timestamp.tz_convert(self._tz)
            return timestamp
        except Exception:
            return timestamp

    def _timestamp_to_period(self, timestamp: pd.Timestamp) -> int:
        """Convert timestamp to period of day (0 to periods_per_day-1).

        For 48 periods/day (semi-hourly), period = hour * 2 + (1 if minute >= 30 else 0).
        For 24 periods/day (hourly), period = hour.

        Args:
            timestamp: Input timestamp (should be normalized to local timezone).

        Returns:
            Period of day (0 to periods_per_day-1).
        """
        normalized = self._normalize_timestamp(timestamp)
        hour = normalized.hour
        minute = normalized.minute

        if self.config.periods_per_day == 48:
            # Semi-hourly: 0-47
            return hour * 2 + (1 if minute >= 30 else 0)
        elif self.config.periods_per_day == 24:
            # Hourly: 0-23
            return hour
        else:
            # Generic: distribute periods evenly
            minutes_per_period = (24 * 60) // self.config.periods_per_day
            total_minutes = hour * 60 + minute
            return total_minutes // minutes_per_period

    def _calculate_recent_errors(
        self,
        observations: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate errors from observations.

        Args:
            observations: DataFrame with timestamp, load, and optionally base_prediction.

        Returns:
            DataFrame with columns: timestamp, observed, predicted, error, error_ratio,
            period, day_of_week, weight, subsystem.
        """
        errors_list = []

        for _, row in observations.iterrows():
            timestamp = row["timestamp"]
            observed = row["load"]

            # Get base prediction
            if "base_prediction" in observations.columns and pd.notna(row["base_prediction"]):
                predicted = row["base_prediction"]
            else:
                # No base prediction available, skip this observation
                logger.debug(
                    "No base prediction for timestamp %s, skipping error calculation",
                    timestamp,
                )
                continue

            # Calculate additive error
            error = observed - predicted

            # Calculate multiplicative error ratio (for multiplicative corrections)
            # error_ratio = (observed / predicted) - 1, so corrected = predicted * (1 + error_ratio)
            if predicted != 0:
                error_ratio = (observed / predicted) - 1.0
                # Clip extreme ratios to prevent instability
                error_ratio = np.clip(error_ratio, -0.5, 0.5)
            else:
                error_ratio = 0.0

            # Normalize timestamp and extract time features
            normalized_ts = self._normalize_timestamp(timestamp)
            period = self._timestamp_to_period(normalized_ts)
            day_of_week = normalized_ts.dayofweek

            errors_list.append(
                {
                    "timestamp": timestamp,
                    "observed": observed,
                    "predicted": predicted,
                    "error": error,
                    "error_ratio": error_ratio,
                    "period": period,
                    "day_of_week": day_of_week,
                    "weight": 1.0,  # Initial weight, will be recalculated later
                    "subsystem": self.config.subsystem,
                }
            )

        if not errors_list:
            return pd.DataFrame()

        errors_df = pd.DataFrame(errors_list)

        # Recalculate weights relative to the most recent observation
        if not errors_df.empty:
            most_recent = errors_df["timestamp"].max()
            errors_df["weight"] = errors_df["timestamp"].apply(
                lambda ts: self.config.error_decay_factor
                ** ((most_recent - ts).total_seconds() / 3600.0)
            )

        return errors_df

    def _calculate_correction_factors(self) -> dict[int, float]:
        """Calculate period-specific correction factors from history.

        Returns:
            Dictionary mapping period (0 to periods_per_day-1) to correction factor.
            For additive mode: correction in load units.
            For multiplicative mode: correction ratio (to multiply by base prediction).
        """
        if self.correction_history.empty:
            return {}

        # Get recent history within correction window (relative to most recent observation)
        if not self.correction_history.empty:
            most_recent = self.correction_history["timestamp"].max()
            cutoff_time = most_recent - timedelta(hours=self.config.correction_window_hours)
            recent = self.correction_history[
                self.correction_history["timestamp"] >= cutoff_time
            ].copy()
        else:
            recent = self.correction_history.copy()

        if recent.empty or len(recent) < self.config.min_observations:
            return {}

        # Select error column based on correction mode
        if self.config.correction_mode == "multiplicative":
            error_col = "error_ratio"
        else:
            error_col = "error"

        # Handle legacy data that might not have error_ratio column
        if error_col not in recent.columns:
            error_col = "error"
            if self.config.correction_mode == "multiplicative":
                logger.warning(
                    "error_ratio column not found in history, falling back to additive corrections"
                )

        # Calculate weighted mean error by period
        correction_factors = {}
        for period in range(self.config.periods_per_day):
            # Use 'period' column if available, fall back to 'hour' for backward compatibility
            if "period" in recent.columns:
                period_data = recent[recent["period"] == period]
            elif "hour" in recent.columns:
                # Legacy hourly data - map to periods
                if self.config.periods_per_day == 48:
                    # For semi-hourly, use hour * 2 for :00 periods, hour * 2 + 1 for :30
                    hour = period // 2
                    period_data = recent[recent["hour"] == hour]
                else:
                    period_data = recent[recent["hour"] == period]
            else:
                period_data = pd.DataFrame()

            if period_data.empty:
                # No data for this period, use overall weighted mean
                total_weight = recent["weight"].sum()
                if total_weight > 0:
                    correction_factors[period] = (
                        recent[error_col] * recent["weight"]
                    ).sum() / total_weight
                else:
                    correction_factors[period] = 0.0
            else:
                # Weighted mean for this specific period
                total_weight = period_data["weight"].sum()
                if total_weight > 0:
                    correction_factors[period] = (
                        period_data[error_col] * period_data["weight"]
                    ).sum() / total_weight
                else:
                    correction_factors[period] = 0.0

        logger.debug(
            "Calculated correction factors for %d periods from %d recent observations (mode=%s)",
            len(correction_factors),
            len(recent),
            self.config.correction_mode,
        )

        return correction_factors

    def _train_error_model(self) -> None:
        """Train statistical error model (Ridge/Lasso) on correction history.

        This model learns patterns in forecast errors based on time features
        (period, day of week) to improve correction accuracy.
        """
        if len(self.correction_history) < MIN_HISTORY_FOR_ERROR_MODEL:
            logger.debug("Insufficient history to train error model")
            return

        # Prepare features: period and day_of_week
        # Use 'period' column if available, fall back to 'hour' for backward compatibility
        if "period" in self.correction_history.columns:
            feature_cols = ["period", "day_of_week"]
        elif "hour" in self.correction_history.columns:
            feature_cols = ["hour", "day_of_week"]
        else:
            logger.warning("No period or hour column found in history, skipping error model training")
            return

        x_error = self.correction_history[feature_cols].to_numpy()
        y_error = self.correction_history["error"].to_numpy()
        weights = self.correction_history["weight"].to_numpy()

        # Create error model
        if self.config.error_model_type == "ridge":
            self.error_model = Ridge(alpha=self.config.error_model_alpha)
        else:
            self.error_model = Lasso(alpha=self.config.error_model_alpha)

        # Train with sample weights
        try:
            self.error_model.fit(x_error, y_error, sample_weight=weights)
            logger.info(
                "Trained %s error model on %d observations",
                self.config.error_model_type,
                len(self.correction_history),
            )
        except Exception as e:
            logger.warning("Error model training failed: %s", e)
            self.error_model = None

    def _adjust_confidence_bounds(
        self,
        predictions: pd.DataFrame,
        horizons: list[int],
    ) -> pd.DataFrame:
        """Add confidence bounds to predictions based on recent error statistics.

        Uses empirically-calibrated horizon scaling factors that reflect actual
        load forecast error growth patterns (rapid increase D+0 to D+1, plateau after D+4).

        Args:
            predictions: DataFrame with prediction columns h{n}.
            horizons: List of horizons.

        Returns:
            DataFrame with added columns h{n}_lower and h{n}_upper.
        """
        if self.correction_history.empty:
            return self._add_default_confidence_bounds(predictions, horizons)

        # Calculate empirical error standard deviation
        error_std = self.correction_history["error"].std()

        # Calculate z-score for confidence level
        z_score = stats.norm.ppf((1 + self.config.confidence_level) / 2)

        # Add bounds for each horizon
        for horizon in horizons:
            pred_col = f"h{horizon}"
            if pred_col not in predictions.columns:
                continue

            # Use empirically-calibrated horizon scaling factors
            # These reflect actual load forecast error growth: rapid D+0→D+1, plateau after D+4
            horizon_factor = DEFAULT_HORIZON_FACTORS.get(horizon, 2.2)
            interval_width = z_score * error_std * horizon_factor

            predictions[f"{pred_col}_lower"] = np.maximum(
                predictions[pred_col] - interval_width,
                0.0,  # Load cannot be negative
            )
            predictions[f"{pred_col}_upper"] = predictions[pred_col] + interval_width

        return predictions

    def _add_default_confidence_bounds(
        self,
        predictions: pd.DataFrame,
        horizons: list[int],
    ) -> pd.DataFrame:
        """Add default confidence bounds (10% of prediction) when no history available.

        Args:
            predictions: DataFrame with prediction columns.
            horizons: List of horizons.

        Returns:
            DataFrame with added confidence bound columns.
        """
        for horizon in horizons:
            pred_col = f"h{horizon}"
            if pred_col not in predictions.columns:
                continue

            # Default: ±10% confidence interval
            default_interval = predictions[pred_col] * 0.10
            predictions[f"{pred_col}_lower"] = np.maximum(
                predictions[pred_col] - default_interval,
                0.0,
            )
            predictions[f"{pred_col}_upper"] = predictions[pred_col] + default_interval

        return predictions

    def _extract_periods(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Extract period of day from features or index.

        For semi-hourly data (48 periods/day), period = hour * 2 + (1 if minute >= 30 else 0).
        For hourly data (24 periods/day), period = hour.

        Args:
            X: Features DataFrame.

        Returns:
            Array of periods (0 to periods_per_day-1).
        """
        # Try to extract from index if it's a DatetimeIndex
        if isinstance(X.index, pd.DatetimeIndex):
            if self.config.periods_per_day == 48:
                # Semi-hourly: convert to period
                hours = X.index.hour.to_numpy()
                minutes = X.index.minute.to_numpy()
                return hours * 2 + (minutes >= 30).astype(int)
            else:
                return X.index.hour.to_numpy()

        # Try to find period in columns
        period_cols = [col for col in X.columns if "period" in col.lower()]
        if period_cols:
            return X[period_cols[0]].to_numpy().astype(int)

        # Try to find hour in columns
        hour_cols = [col for col in X.columns if "hour" in col.lower()]
        if hour_cols:
            hours = X[hour_cols[0]].to_numpy()
            if self.config.periods_per_day == 48:
                # Assume :00 minutes, so period = hour * 2
                return (hours * 2).astype(int)
            return hours.astype(int)

        # Default: assume period 24 (midday for 48 periods) or 12 (for 24 periods)
        default_period = self.config.periods_per_day // 2
        logger.warning(
            "Could not extract period from features, using default period=%d",
            default_period,
        )
        return np.full(len(X), default_period)

    def get_correction_summary(self) -> pd.DataFrame:
        """Get summary of correction factors by period.

        Returns:
            DataFrame with columns: period, mean_error, weighted_mean_error,
            n_observations, correction_factor. For semi-hourly data, includes
            periods 0-47.

        Example:
            ```python
            summary = blf.get_correction_summary()
            print(summary)
            #    period  mean_error  weighted_mean_error  n_observations  correction_factor
            # 0       0       -2.5                 -2.3               10               -2.3
            # 1       1       -1.8                 -1.7                8               -1.7
            # ...
            ```
        """
        if self.correction_history.empty:
            return pd.DataFrame(
                columns=[
                    "period",
                    "mean_error",
                    "weighted_mean_error",
                    "n_observations",
                    "correction_factor",
                ]
            )

        # Get correction factors
        correction_factors = self._calculate_correction_factors()

        # Determine period column (use 'period' if available, fall back to 'hour')
        period_col = "period" if "period" in self.correction_history.columns else "hour"

        # Calculate statistics by period
        summary_list = []
        for period in range(self.config.periods_per_day):
            if period_col == "period":
                period_data = self.correction_history[self.correction_history["period"] == period]
            else:
                # Legacy hourly data - map to periods
                if self.config.periods_per_day == 48:
                    hour = period // 2
                    period_data = self.correction_history[self.correction_history["hour"] == hour]
                else:
                    period_data = self.correction_history[self.correction_history["hour"] == period]

            if period_data.empty:
                summary_list.append(
                    {
                        "period": period,
                        "mean_error": 0.0,
                        "weighted_mean_error": 0.0,
                        "n_observations": 0,
                        "correction_factor": correction_factors.get(period, 0.0),
                    }
                )
            else:
                mean_error = period_data["error"].mean()
                total_weight = period_data["weight"].sum()
                weighted_mean = (
                    (period_data["error"] * period_data["weight"]).sum() / total_weight
                    if total_weight > 0
                    else 0.0
                )

                summary_list.append(
                    {
                        "period": period,
                        "mean_error": mean_error,
                        "weighted_mean_error": weighted_mean,
                        "n_observations": len(period_data),
                        "correction_factor": correction_factors.get(period, 0.0),
                    }
                )

        return pd.DataFrame(summary_list)

    def save_state(self, path: str | Path) -> None:
        """Save BLF predictor state to disk.

        This saves correction history, error model, and configuration.
        The base model is NOT saved and must be provided when loading.

        Args:
            path: File path to save state. Parent directory created if needed.

        Raises:
            OSError: If file cannot be written.

        Example:
            ```python
            blf.save_state("blf_state.pkl")
            ```
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config.model_dump(),
            "correction_history": self.correction_history,
            "error_model": self.error_model,
            "last_update": self.last_update,
            "base_model_name": self.base_model.name,
            "base_model_version": self.base_model.version,
        }

        try:
            joblib.dump(state, path)
            logger.info("Saved BLF state to %s", path)
        except Exception as e:
            msg = f"Failed to save BLF state to {path}: {e}"
            raise OSError(msg) from e

    @classmethod
    def load_state(
        cls,
        path: str | Path,
        base_model: BaseModel,
    ) -> "BLFPredictor":
        """Load BLF predictor state from disk.

        Args:
            path: File path to load state from.
            base_model: Fitted base model to use with loaded state.
                       Must match the model used when state was saved.

        Returns:
            BLFPredictor instance with restored state.

        Raises:
            FileNotFoundError: If state file doesn't exist.
            ValueError: If base model doesn't match saved state.
            OSError: If file cannot be read.

        Example:
            ```python
            base_model = LGBMModel.load("base_model.joblib")
            blf = BLFPredictor.load_state("blf_state.pkl", base_model)
            ```
        """
        path = Path(path)

        if not path.exists():
            msg = f"BLF state file not found: {path}"
            raise FileNotFoundError(msg)

        try:
            state = joblib.load(path)
        except Exception as e:
            msg = f"Failed to load BLF state from {path}: {e}"
            raise OSError(msg) from e

        # Validate base model matches
        if base_model.name != state.get("base_model_name"):
            msg = (
                f"Base model name mismatch: expected {state.get('base_model_name')}, "
                f"got {base_model.name}"
            )
            raise ValueError(msg)

        if base_model.version != state.get("base_model_version"):
            logger.warning(
                "Base model version mismatch: expected %s, got %s",
                state.get("base_model_version"),
                base_model.version,
            )

        # Create predictor instance
        predictor = cls(base_model, config=state["config"])

        # Restore state
        predictor.correction_history = state["correction_history"]
        predictor.error_model = state["error_model"]
        predictor.last_update = state["last_update"]

        logger.info("Loaded BLF state from %s", path)
        return predictor

    def __repr__(self) -> str:
        """Return string representation of BLF predictor.

        Returns:
            String with base model info and correction history size.
        """
        return (
            f"BLFPredictor("
            f"base_model={self.base_model.name}, "
            f"history_size={len(self.correction_history)}, "
            f"last_update={self.last_update})"
        )
