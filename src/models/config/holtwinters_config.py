"""Holt-Winters exponential smoothing model configuration schema.

This module provides Pydantic models for Holt-Winters exponential smoothing
configuration, including trend, seasonal, and damping parameters for both
demand mean and profile forecasting.

Example:
    ```python
    from src.models.config.holtwinters_config import (
        HoltWintersConfig,
        HoltWintersProfileConfig
    )

    # Configuration for demand mean model
    demand_config = HoltWintersConfig(
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        damped_trend=False
    )

    # Configuration for profile models
    profile_config = HoltWintersProfileConfig(
        trend=None,
        seasonal="add",
        seasonal_periods=7
    )
    ```
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HoltWintersConfig(BaseModel):
    """Configuration for Holt-Winters demand mean forecasting.

    This class defines all hyperparameters for Holt-Winters exponential
    smoothing models used for demand mean forecasting, including:
    - Trend component configuration (additive, multiplicative, or none)
    - Seasonal component configuration (additive, multiplicative, or none)
    - Seasonal period length (e.g., 7 for weekly patterns)
    - Damped trend option to prevent over-forecasting
    - Box-Cox transformation for variance stabilization
    - Initialization method for starting values

    Attributes:
        trend: Type of trend component. "add" for additive trend (linear),
            "mul" for multiplicative trend (exponential), or None for no trend.
        seasonal: Type of seasonal component. "add" for additive seasonality
            (constant seasonal effect), "mul" for multiplicative seasonality
            (proportional seasonal effect), or None for no seasonality.
        seasonal_periods: Number of periods in one seasonal cycle. Default 7
            for weekly patterns in daily demand data.
        damped_trend: Whether to use damped trend. Damping reduces the trend
            effect over forecast horizon, preventing unrealistic long-term
            extrapolation.
        use_boxcox: Whether to apply Box-Cox transformation to stabilize
            variance. Can be True (auto-estimate lambda), False (no transform),
            or a float value for lambda.
        initialization_method: Method for initializing level, trend, and
            seasonal components. "estimated" uses optimization, "heuristic"
            uses simple averages.
        min_training_days: Minimum number of daily observations required for
            training. Must be at least 2 * seasonal_periods.
        prediction_intervals: Confidence levels for prediction intervals.
            Default [80, 95] for 80% and 95% intervals.
    """

    # Component types
    trend: Literal["add", "mul", None] = Field(
        default="add",
        description="Trend component: 'add', 'mul', or None"
    )
    seasonal: Literal["add", "mul", None] = Field(
        default="add",
        description="Seasonal component: 'add', 'mul', or None"
    )

    # Seasonal parameters
    seasonal_periods: int = Field(
        default=7,
        ge=2,
        le=365,
        description="Seasonal period length (7 for weekly pattern)"
    )

    # Trend configuration
    damped_trend: bool = Field(
        default=False,
        description="Whether to use damped trend"
    )

    # Transformation
    use_boxcox: bool | float | Literal["log"] = Field(
        default=False,
        description="Box-Cox transformation: False, True, 'log', or lambda value"
    )

    # Initialization
    initialization_method: Literal["estimated", "heuristic", "legacy-heuristic"] = Field(
        default="estimated",
        description="Initialization method for state components"
    )

    # Training constraints
    min_training_days: int = Field(
        default=14,
        ge=7,
        le=365,
        description="Minimum daily observations required for training"
    )

    # Prediction configuration
    prediction_intervals: list[int] = Field(
        default=[80, 95],
        description="Confidence levels for prediction intervals"
    )

    @field_validator("use_boxcox")
    @classmethod
    def validate_boxcox(cls, v: bool | float | str) -> bool | float | str:
        """Validate Box-Cox parameter.

        Args:
            v: Box-Cox parameter value.

        Returns:
            Validated Box-Cox parameter.

        Raises:
            ValueError: If parameter is invalid.
        """
        if isinstance(v, (bool, str)):
            if isinstance(v, str) and v != "log":
                msg = f"use_boxcox string must be 'log', got '{v}'"
                raise ValueError(msg)
        elif isinstance(v, (int, float)):
            # Lambda parameter for Box-Cox must be reasonable
            if not -2.0 <= v <= 2.0:
                logger.warning("Box-Cox lambda %.2f is outside typical range [-2, 2]", v)
        else:
            msg = f"use_boxcox must be bool, float, or 'log', got {type(v)}"
            raise ValueError(msg)

        return v

    @field_validator("prediction_intervals")
    @classmethod
    def validate_prediction_intervals(cls, v: list[int]) -> list[int]:
        """Validate prediction intervals are valid percentages.

        Args:
            v: List of prediction interval confidence levels.

        Returns:
            Validated list of intervals.

        Raises:
            ValueError: If any interval is not between 1 and 99.
        """
        min_interval = 1
        max_interval = 99
        for interval in v:
            if not min_interval <= interval <= max_interval:
                msg = f"Prediction interval {interval} must be between 1 and 99"
                raise ValueError(msg)

        # Sort intervals in ascending order
        return sorted(v)

    @model_validator(mode="after")
    def validate_seasonal_params(self) -> "HoltWintersConfig":
        """Validate seasonal parameters are reasonable.

        Returns:
            Validated configuration.

        Raises:
            ValueError: If seasonal parameters are invalid.
        """
        weekly_season_length = 7

        if self.seasonal is not None:
            if self.seasonal_periods < 2:
                msg = f"seasonal_periods must be >= 2 when seasonal is set, got {self.seasonal_periods}"
                raise ValueError(msg)

            # Warn if season_length doesn't match weekly pattern for daily data
            if self.seasonal_periods != weekly_season_length:
                logger.warning(
                    "seasonal_periods is %d (expected 7 for weekly pattern in daily data)",
                    self.seasonal_periods,
                )
        else:
            # Seasonality disabled - seasonal_periods is ignored
            pass

        return self

    @model_validator(mode="after")
    def validate_min_training_days(self) -> "HoltWintersConfig":
        """Validate minimum training days is sufficient for model.

        Returns:
            Validated configuration.

        Raises:
            ValueError: If min_training_days is too small.
        """
        # Need at least 2 complete seasonal cycles for reliable estimation
        if self.seasonal is not None:
            min_required = self.seasonal_periods * 2
            if self.min_training_days < min_required:
                logger.warning(
                    "min_training_days (%d) should be at least 2*seasonal_periods (%d) "
                    "for reliable seasonal estimation. Recommended: %d days.",
                    self.min_training_days,
                    self.seasonal_periods,
                    min_required,
                )

        return self

    @model_validator(mode="after")
    def validate_damped_trend(self) -> "HoltWintersConfig":
        """Validate damped trend is only used with trend.

        Returns:
            Validated configuration.

        Raises:
            ValueError: If damped_trend=True but trend=None.
        """
        if self.damped_trend and self.trend is None:
            msg = "damped_trend=True requires trend to be set ('add' or 'mul')"
            raise ValueError(msg)

        return self

    def to_statsmodels_params(self) -> dict[str, Any]:
        """Convert configuration to statsmodels ExponentialSmoothing parameters.

        Returns:
            Dictionary of parameters ready for statsmodels ExponentialSmoothing.

        Example:
            ```python
            config = HoltWintersConfig(trend="add", seasonal="add", seasonal_periods=7)
            params = config.to_statsmodels_params()
            # Use with ExponentialSmoothing
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(data, **params)
            ```
        """
        params = {
            "trend": self.trend,
            "seasonal": self.seasonal,
            "seasonal_periods": self.seasonal_periods if self.seasonal else None,
            "damped_trend": self.damped_trend,
            "use_boxcox": self.use_boxcox,
            "initialization_method": self.initialization_method,
        }

        logger.debug(
            "Generated ExponentialSmoothing parameters: trend=%s, seasonal=%s, "
            "seasonal_periods=%s, damped_trend=%s",
            self.trend,
            self.seasonal,
            self.seasonal_periods if self.seasonal else None,
            self.damped_trend,
        )
        return params

    model_config = {
        "frozen": False,  # Allow modification after creation
        "validate_assignment": True,  # Validate on attribute assignment
        "extra": "forbid",  # Raise error on extra fields
    }


class HoltWintersProfileConfig(BaseModel):
    """Configuration for Holt-Winters profile models.

    This class defines hyperparameters for the 48 independent Holt-Winters
    models used for profile ratio forecasting. Profile models typically:
    - Have no trend (profiles are stationary)
    - Have seasonal component for weekly patterns
    - Use simpler initialization since data is already normalized

    Attributes:
        trend: Type of trend component. Usually None for profiles since
            profile ratios are stationary. Can be "add" or "mul" if trend
            is detected in profile patterns.
        seasonal: Type of seasonal component. "add" for additive seasonality,
            "mul" for multiplicative seasonality, or None for no seasonality.
            Default "add" to capture day-of-week patterns.
        seasonal_periods: Number of periods in one seasonal cycle. Default 7
            for weekly patterns in daily profile data.
        damped_trend: Whether to use damped trend. Only used if trend is set.
        use_boxcox: Whether to apply Box-Cox transformation. Usually False
            for profile ratios which are already normalized.
        initialization_method: Method for initializing components.
        min_samples_per_period: Minimum number of training samples required
            per semi-hourly period (0-47). Periods with fewer samples will
            be skipped during training.
        min_profile_ratio: Minimum allowable profile ratio for clipping
            predictions. Default 0.1 (10% of daily mean).
        max_profile_ratio: Maximum allowable profile ratio for clipping
            predictions. Default 5.0 (500% of daily mean).
    """

    # Component types (profiles typically have no trend)
    trend: Literal["add", "mul", None] = Field(
        default=None,
        description="Trend component: 'add', 'mul', or None"
    )
    seasonal: Literal["add", "mul", None] = Field(
        default="add",
        description="Seasonal component: 'add', 'mul', or None"
    )

    # Seasonal parameters
    seasonal_periods: int = Field(
        default=7,
        ge=2,
        le=365,
        description="Seasonal period length (7 for weekly pattern)"
    )

    # Trend configuration
    damped_trend: bool = Field(
        default=False,
        description="Whether to use damped trend"
    )

    # Transformation (usually not needed for normalized profile ratios)
    use_boxcox: bool | float | Literal["log"] = Field(
        default=False,
        description="Box-Cox transformation: False, True, 'log', or lambda value"
    )

    # Initialization
    initialization_method: Literal["estimated", "heuristic", "legacy-heuristic"] = Field(
        default="estimated",
        description="Initialization method for state components"
    )

    # Training constraints
    min_samples_per_period: int = Field(
        default=14,
        ge=7,
        le=500,
        description="Minimum samples required per semi-hourly period"
    )

    # Profile ratio bounds
    min_profile_ratio: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Minimum allowable profile ratio"
    )
    max_profile_ratio: float = Field(
        default=5.0,
        ge=1.0,
        le=10.0,
        description="Maximum allowable profile ratio"
    )

    @field_validator("use_boxcox")
    @classmethod
    def validate_boxcox(cls, v: bool | float | str) -> bool | float | str:
        """Validate Box-Cox parameter.

        Args:
            v: Box-Cox parameter value.

        Returns:
            Validated Box-Cox parameter.

        Raises:
            ValueError: If parameter is invalid.
        """
        if isinstance(v, (bool, str)):
            if isinstance(v, str) and v != "log":
                msg = f"use_boxcox string must be 'log', got '{v}'"
                raise ValueError(msg)
        elif isinstance(v, (int, float)):
            if not -2.0 <= v <= 2.0:
                logger.warning("Box-Cox lambda %.2f is outside typical range [-2, 2]", v)
        else:
            msg = f"use_boxcox must be bool, float, or 'log', got {type(v)}"
            raise ValueError(msg)

        return v

    @model_validator(mode="after")
    def validate_damped_trend(self) -> "HoltWintersProfileConfig":
        """Validate damped trend is only used with trend.

        Returns:
            Validated configuration.

        Raises:
            ValueError: If damped_trend=True but trend=None.
        """
        if self.damped_trend and self.trend is None:
            msg = "damped_trend=True requires trend to be set ('add' or 'mul')"
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_min_samples(self) -> "HoltWintersProfileConfig":
        """Validate minimum samples is sufficient.

        Returns:
            Validated configuration.
        """
        if self.seasonal is not None:
            min_required = self.seasonal_periods * 2
            if self.min_samples_per_period < min_required:
                logger.warning(
                    "min_samples_per_period (%d) should be at least 2*seasonal_periods (%d) "
                    "for reliable seasonal estimation.",
                    self.min_samples_per_period,
                    min_required,
                )

        return self

    def to_statsmodels_params(self) -> dict[str, Any]:
        """Convert configuration to statsmodels ExponentialSmoothing parameters.

        Returns:
            Dictionary of parameters ready for statsmodels ExponentialSmoothing.

        Example:
            ```python
            config = HoltWintersProfileConfig(seasonal="add", seasonal_periods=7)
            params = config.to_statsmodels_params()
            # Use with ExponentialSmoothing
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(data, **params)
            ```
        """
        params = {
            "trend": self.trend,
            "seasonal": self.seasonal,
            "seasonal_periods": self.seasonal_periods if self.seasonal else None,
            "damped_trend": self.damped_trend,
            "use_boxcox": self.use_boxcox,
            "initialization_method": self.initialization_method,
        }

        logger.debug(
            "Generated ExponentialSmoothing parameters for profile: trend=%s, "
            "seasonal=%s, seasonal_periods=%s",
            self.trend,
            self.seasonal,
            self.seasonal_periods if self.seasonal else None,
        )
        return params

    model_config = {
        "frozen": False,  # Allow modification after creation
        "validate_assignment": True,  # Validate on attribute assignment
        "extra": "forbid",  # Raise error on extra fields
    }
